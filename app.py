"""
app.py
The Streamlit dashboard. It connects all the other files together.

Run:  streamlit run app.py      (then open http://localhost:8501)
Colours are set in .streamlit/config.toml, extra styling in style.css.
"""
import hashlib
import html
import os
import time

import cv2
import streamlit as st

from database import dispense, get_inventory, get_log, init_db
from dosage import recommend
from extract import extract_medicines, plain_language
from matcher import match_medicine
from ocr import available_engines, read_text, text_to_result
from preprocess import load_image, preprocess

st.set_page_config(page_title="Smart Prescription Reader", layout="wide")
init_db()

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

STATUS_CLASS = {"Matched": "ok", "Not found": "bad"}


def shown_status(match):
    """A medicine counts as matched when a stock medicine is at least 60% likely to be it."""
    return "Matched" if match["likely"] else "Not found"


def stat_tile(label, value, note=""):
    """A small card showing one number (used in the rows of statistics)."""
    return (f"<div class='tile'><div class='tile-label'>{label}</div>"
            f"<div class='tile-value'>{value}</div><div class='tile-note'>{note}</div></div>")


def tiles_row(tiles):
    st.markdown("<div class='tiles'>" + "".join(tiles) + "</div>", unsafe_allow_html=True)


def mark_medicines(image, medicines, matches):
    """
    Draw a numbered box around each identified medicine on the processed image
    (green = matched in stock, red = not found). The numbers match the medicine cards.
    """
    marked = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    height, width = marked.shape[:2]
    thick = max(2, width // 400)
    for number, (med, match) in enumerate(zip(medicines, matches), start=1):
        colour = (79, 111, 47) if shown_status(match) == "Matched" else (40, 40, 185)  # BGR
        x1 = min(b[0] for b in med["boxes"]) * width
        y1 = min(b[1] for b in med["boxes"]) * height
        x2 = max(b[0] + b[2] for b in med["boxes"]) * width
        y2 = max(b[1] + b[3] for b in med["boxes"]) * height
        pad = 0.006 * width
        x1, y1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
        x2, y2 = int(min(width - 1, x2 + pad)), int(min(height - 1, y2 + pad))
        cv2.rectangle(marked, (x1, y1), (x2, y2), colour, thick)
        # Number tag just left of the box (inside it if there is no room), same number as the card.
        scale = max(0.7, width / 1000)
        (tw, th), _ = cv2.getTextSize(str(number), cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        tag_w, tag_h = tw + 12, th + 12
        tx = x1 - tag_w - 4 if x1 - tag_w - 4 >= 0 else x1
        ty = max(0, min(height - tag_h, (y1 + y2) // 2 - tag_h // 2))
        cv2.rectangle(marked, (tx, ty), (tx + tag_w, ty + tag_h), colour, -1)
        cv2.putText(marked, str(number), (tx + 6, ty + th + 6), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)
    return marked


def availability(stock):
    if stock == 0:
        return "bad", "Out of stock"
    return ("warn", "Low stock") if stock < 30 else ("ok", "In stock")


def plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def medicine_card(number, med, match, advice, products):
    """
    A compact card for one scanned medicine: name, dosage, best stock match and how many strips to give.
    Clicking the arrow opens the details: matching medicines in stock, the strip calculation and warnings.
    """
    status = shown_status(match)
    css = STATUS_CLASS[status]
    title = html.escape(f"{number}. {med['name'] or 'Unknown medicine'} {med['strength'] or ''}")
    sentence = html.escape(plain_language(med).split(": ", 1)[1])
    best = match["likely"][0] if match["likely"] else None
    stock_line = (f"{html.escape(best['label'])} &middot; Shelf {html.escape(str(best['shelf']))} &middot; "
                  f"{availability(best['stock'])[1]}" if best else "No match in stock")
    notes = (f"<span class='note-count'>{plural(len(match['flags']), 'note')}</span>"
             if match["flags"] else "")
    summary = (f"<summary><div class='sum-main'><div><span class='card-title'>{title}</span>"
               f"<span class='card-dose'>{sentence[0].upper() + sentence[1:]}</span></div>"
               f"<span class='card-stock'>{stock_line}</span></div>"
               f"<div class='sum-side'>{notes}<span class='strips'>{html.escape(advice['text'])}</span>"
               f"<span class='badge {css}'>{status}</span></div>"
               f"<span class='chevron'></span></summary>")

    use_text = products.get(best["med_id"], {}).get("use") if best else None
    use = f"<div class='card-use'>Used for: {html.escape(use_text)}</div>" if use_text else ""
    quantity = (f"<div class='match-title'>Recommended quantity</div>"
                f"<div class='quantity'><b>{html.escape(advice['text'])}</b>: {html.escape(advice['detail'])}</div>")
    rows = ""
    for c in match["likely"]:
        stock_css, stock_text = availability(c["stock"])
        pack = products.get(c["med_id"], {}).get("pack", "pack")
        rows += (f"<tr><td>{html.escape(c['label'])}</td>"
                 f"<td><div class='bar'><span style='width:{c['score']}%'></span></div>{c['score']}%</td>"
                 f"<td><span class='badge {stock_css}'>{stock_text}</span> {plural(c['stock'], pack)}</td>"
                 f"<td>Shelf {html.escape(str(c['shelf']))}</td></tr>")
    if rows:
        stock_table = ("<div class='match-title'>Matching medicines in stock (60% likely or more)</div>"
                       "<table class='match-table'><tr><th>Medicine</th><th>Likelihood</th><th>Stock</th>"
                       f"<th>Location</th></tr>{rows}</table>")
    else:
        stock_table = "<div class='match-none'>No medicine in stock is at least 60% likely to match.</div>"
    flags = "".join(f"<li>{html.escape(f)}</li>" for f in match["flags"])
    flags = f"<div class='match-title'>Notes</div><ul class='card-flags'>{flags}</ul>" if flags else ""
    return (f"<details class='med-card {css}'>{summary}"
            f"<div class='card-body'>{use}{quantity}{stock_table}{flags}</div></details>")


@st.cache_data(show_spinner="Scanning the prescription...")
def scan(file_bytes, engine):
    """Image -> text. Cached, so editing the text below does not scan the image again."""
    start = time.time()
    image = load_image(file_bytes)
    if image is None:
        return None, None, None, 0
    clean = preprocess(image)
    # Apple Vision reads the original photo best; Tesseract needs the cleaned black & white image.
    ocr_result = read_text(image if engine == "vision" else clean, engine)
    return image, clean, ocr_result, time.time() - start


def suggested_id(match):
    """Medicine to pre-select in the pharmacist form: the best stock match (60% likely or more)."""
    return match["likely"][0]["med_id"] if match["likely"] else None


# ---------------- Header ----------------
st.markdown("""
<div class='hero'>
  <h1>Smart Prescription Reader</h1>
  <p>Scan a prescription, understand every medicine.</p>
  <div class='hero-note'>Pharmacist verification required before dispensing.</div>
</div>""", unsafe_allow_html=True)

# ---------------- Sidebar: choose a prescription ----------------
with st.sidebar:
    st.markdown("<div class='side-title'>Prescription</div>", unsafe_allow_html=True)
    source = st.radio("Source", ["Sample prescription", "Upload an image"], label_visibility="collapsed")
    file_bytes = None
    if source == "Upload an image":
        upload = st.file_uploader("Prescription photo", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
                                  help="A clear, well-lit photo or scan. iPhone HEIC photos: export as JPG first.")
        if upload:
            file_bytes = upload.getvalue()
    else:
        samples = sorted(f for f in os.listdir("data/samples") if f.endswith(".png"))
        chosen = st.selectbox("Sample", samples, help="rx_001-030 printed, rx_031-060 handwriting style, rx_061-100 messy handwritten")
        with open(os.path.join("data/samples", chosen), "rb") as f:
            file_bytes = f.read()

    # The engine choice is folded away: the default (Apple Vision when available) suits most photos.
    ENGINE_LABELS = {"vision": "Apple Vision (handwriting and photos)", "tesseract": "Tesseract (printed text)"}
    with st.expander("Text recognition"):
        engine = st.radio("Engine", available_engines(), format_func=ENGINE_LABELS.get, label_visibility="collapsed")

tab_read, tab_inventory, tab_log = st.tabs(["Read prescription", "Inventory", "Dispense log"])

# ---------------- Tab 1: read a prescription ----------------
with tab_read:
    if not file_bytes:
        st.info("Upload a prescription image from the sidebar to begin.")
    elif scan(file_bytes, engine)[0] is None:
        st.error("This file could not be opened as an image. Please upload a JPG or PNG photo.")
    else:
        image, clean, ocr_result, scan_seconds = scan(file_bytes, engine)
        rx_id = hashlib.md5(file_bytes).hexdigest()[:8]  # new prescription -> fresh form and text box
        tiles_slot = st.empty()  # filled in below, once the medicines are known
        start = time.time()
        scanned_medicines = extract_medicines(ocr_result)
        # Only the medicine lines and their dosage (no doctor, patient or address details).
        medicine_text = "\n".join(m["line"] for m in scanned_medicines)

        # ---- Image to text: the photo next to its scanned medicine lines ----
        col1, col2 = st.columns(2)
        with col1, st.container(border=True, key="panel_original"):
            st.markdown("<div class='panel-title'>Original prescription</div>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")
        with col2, st.container(border=True, key="panel_text"):
            st.markdown("<div class='panel-title'>Scanned medicines and dosage</div>", unsafe_allow_html=True)
            text = st.text_area("Scanned medicines and dosage", medicine_text, height=230,
                                key=f"text_{rx_id}_{engine}", label_visibility="collapsed",
                                placeholder="No medicine lines were found. Type them here, one per line, "
                                            "e.g. Tab Paracetamol 500mg - 1 tab - BD - 5 days",
                                help="Fix any misread word here; the medicines below update automatically.")
            st.download_button("Download text", text, file_name=f"prescription_{rx_id}.txt")
            edited = text.strip() != medicine_text.strip()
            if edited:
                st.caption("Using your corrected text.")
        with st.expander("Processed image with identified medicines", expanded=True):
            processed_slot = st.empty()  # filled in below with the medicines marked in boxes

        medicines = extract_medicines(text_to_result(text)) if edited else scanned_medicines
        inventory = get_inventory()
        matches = [match_medicine(m, inventory) for m in medicines]
        products = inventory.set_index("med_id").to_dict("index")  # med_id -> pack size, form, ...
        # Strips to give, worked out from the dosage and the best-matching product's pack size.
        advice = [recommend(m, products.get(r["likely"][0]["med_id"]) if r["likely"] else None)
                  for m, r in zip(medicines, matches)]
        seconds = scan_seconds + time.time() - start

        # Boxes come from the scan, so they are only drawn while the scanned text is unchanged.
        processed = clean if edited else mark_medicines(clean, medicines, matches)
        processed_slot.image(processed, channels="BGR" if processed.ndim == 3 else "RGB", width="stretch",
                             caption=None if edited else "Green = matched, red = not found")

        statuses = [shown_status(m) for m in matches]
        with tiles_slot.container():
            tiles_row([
                stat_tile("Medicines read", len(medicines), "lines detected on the page"),
                stat_tile("Matched", statuses.count("Matched"), "found in the inventory"),
                stat_tile("Not found", statuses.count("Not found"), "not in the inventory"),
                stat_tile("Processing time", f"{seconds:.2f}s", "scan and analysis"),
            ])

        st.markdown("<h3 class='section'>Medicines and dosage</h3>", unsafe_allow_html=True)
        if not medicines:
            st.error("No medicines could be read. Please check the original prescription manually.")
        cards = [medicine_card(i, m, r, a, products)
                 for i, (m, r, a) in enumerate(zip(medicines, matches, advice), start=1)]
        st.markdown("<div class='card-list'>" + "".join(cards) + "</div>", unsafe_allow_html=True)

        # ---- Pharmacist verification ----
        st.markdown("<h3 class='section'>Pharmacist verification</h3>", unsafe_allow_html=True)
        inventory = get_inventory()
        options = ["(do not dispense)"] + [f"{r.med_id}: {r.name} {r.strength} (shelf {r.shelf}, stock {r.stock})"
                                           for r in inventory.itertuples()]
        with st.form("verify", border=False), st.container(border=True, key="panel_verify"):
            pharmacist = st.text_input("Pharmacist name / registration no.")
            choices = []
            for i, (med, match) in enumerate(zip(medicines, matches)):
                # Pre-select the suggested medicine; the pharmacist must still tick "checked" for every line.
                default = 0
                suggestion = suggested_id(match)
                if suggestion is not None:
                    default = next(j for j, o in enumerate(options) if o.startswith(f"{suggestion}:"))
                c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="bottom")
                pick = c1.selectbox(f"Line {i + 1}: {med['line']}", options, index=default, key=f"pick{i}_{rx_id}")
                qty = c2.number_input("Strips / bottles", min_value=1, value=advice[i]["packs"] or 1,
                                      key=f"qty{i}_{rx_id}", help=advice[i]["detail"])
                ok = c3.checkbox("Checked against original", key=f"ok{i}_{rx_id}")
                choices.append((pick, qty, ok))
            confirm = st.checkbox("I confirm I have reviewed this prescription and the selected medicines.")
            submitted = st.form_submit_button("Verify and dispense", type="primary")

        if submitted:
            if not pharmacist.strip() or not confirm:
                st.error("Enter the pharmacist name and tick the final confirmation.")
            else:
                for pick, qty, ok in choices:
                    if pick == options[0]:
                        continue
                    if not ok:
                        st.error(f"Skipped {pick.split(': ', 1)[1].split(' (')[0]}: not ticked as checked.")
                        continue
                    med_id = int(pick.split(":")[0])
                    row = inventory[inventory["med_id"] == med_id].iloc[0]
                    try:
                        dispense(med_id, int(qty), pharmacist.strip())
                        st.success(f"Dispensed {plural(int(qty), row['pack'])} of {row['name']} {row['strength']} "
                                   f"from shelf {row['shelf']}")
                    except ValueError as error:
                        st.error(f"{row['name']} {row['strength']}: {error}")

# ---------------- Tab 2: inventory ----------------
with tab_inventory:
    inventory = get_inventory()
    tiles_row([
        stat_tile("Products", len(inventory), "medicine and strength pairs"),
        stat_tile("Categories", inventory["category"].nunique(), "therapeutic groups"),
        stat_tile("Packs in stock", f"{inventory['stock'].sum():,}", "strips, bottles and tubes"),
        stat_tile("Low stock", int(((inventory["stock"] > 0) & (inventory["stock"] < 30)).sum()), "under 30 units"),
        stat_tile("Out of stock", int((inventory["stock"] == 0).sum()), "needs reordering"),
    ])
    f1, f2 = st.columns([2, 1])
    search = f1.text_input("Search by name", placeholder="e.g. Paracetamol")
    categories = f2.multiselect("Category", sorted(inventory["category"].unique()))
    shown = inventory[inventory["name"].str.contains(search, case=False)] if search else inventory
    if categories:
        shown = shown[shown["category"].isin(categories)]
    st.dataframe(shown.style.apply(
        lambda r: ["background-color: #fee2e2" if r["stock"] == 0 else
                   "background-color: #fef3c7" if r["stock"] < 30 else ""] * len(r), axis=1),
        width="stretch", hide_index=True, height=520)
    st.caption("Red rows are out of stock, amber rows are low on stock.")
    if st.button("Reset inventory to the original CSV"):
        init_db(reset=True)
        st.rerun()

# ---------------- Tab 3: dispense log ----------------
with tab_log:
    log = get_log()
    if log.empty:
        st.info("Nothing dispensed yet.")
    else:
        st.dataframe(log, width="stretch", hide_index=True)
