"""
make_samples.py
Creates SYNTHETIC (fake) prescription images for testing, plus the correct answers.

Three styles:
  printed      rx_001-030  typed text with camera noise
  handwritten  rx_031-060  handwriting-style fonts, blur, tilt, low resolution
  messy        rx_061-100  like a real phone photo of a doctor's note: blue/black ink on paper,
                           wobbly words, cursive fonts, brand names, syrups, dosing written on the
                           line below the medicine, diagnosis notes, JPEG blur

Run:  python make_samples.py

Output:
  data/samples/rx_001.png ...   -> prescription images
  data/samples/rx_001.txt ...   -> the exact text written on each image
  data/ground_truth.csv         -> the medicines on each image (answer key)
"""
import csv
import io
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

random.seed(42)  # same "random" samples every time, so results can be repeated
np.random.seed(42)

SAMPLES_DIR = "data/samples"
GROUND_TRUTH = "data/ground_truth.csv"
STYLES = ["printed"] * 30 + ["handwritten"] * 30 + ["messy"] * 40

# Fonts that ship with macOS.
FONT_DIR = "/System/Library/Fonts/"
FONTS = {
    "printed": [FONT_DIR + "Supplemental/Arial.ttf", FONT_DIR + "Supplemental/Courier New.ttf"],
    "handwritten": [FONT_DIR + "Supplemental/Bradley Hand Bold.ttf", FONT_DIR + "Noteworthy.ttc",
                    FONT_DIR + "MarkerFelt.ttc", FONT_DIR + "Supplemental/Chalkboard.ttc"],
    "messy": [FONT_DIR + "Supplemental/Bradley Hand Bold.ttf", FONT_DIR + "Noteworthy.ttc",
              FONT_DIR + "Supplemental/Apple Chancery.ttf", FONT_DIR + "Supplemental/SnellRoundhand.ttc",
              FONT_DIR + "Supplemental/Comic Sans MS.ttf", FONT_DIR + "MarkerFelt.ttc"],
}

# Medicines that ARE in the inventory (tablets, capsules and syrups), read from data/medicines.csv
with open("data/medicines.csv") as f:
    IN_STOCK = [row for row in csv.DictReader(f) if row["form"] in ("Tablet", "Capsule", "Syrup")]
# Medicines that are NOT in the inventory (the system should say "Not found")
NOT_IN_STOCK = [{"name": n, "strength": s, "form": "Tablet", "brands": ""} for n, s in
                [("Bilastine", "20mg"), ("Nitazoxanide", "500mg"), ("Faropenem", "200mg"),
                 ("Ethamsylate", "500mg"), ("Mirabegron", "50mg"), ("Rupatadine", "10mg")]]

FORM_WORD = {"Tablet": ["Tab", "Tab", "Tab."], "Capsule": ["Cap", "Cap."], "Syrup": ["Syp", "Syrup"]}
DOSE = {"Tablet": ["1 tab", "1 tab", "2 tab"], "Capsule": ["1 cap"], "Syrup": ["5 ml", "10 ml", "2.5 ml"]}
FREQUENCIES = ["twice daily", "BD", "once daily", "OD", "thrice daily", "TDS", "at night", "1-0-1", "SOS"]
SCHEDULES = ["1 - 0 - 1", "1 - 1 - 1", "0 - 0 - 1", "1 - 0 - 0", "x - x - 1"]  # written below the medicine
DOCTORS = ["Dr. A. Sharma, MBBS", "Dr. R. Iyer, MD", "Dr. P. Banerjee, MBBS", "Dr. S. Khan, MD"]
PATIENTS = ["Ravi Kumar", "Anita Das", "Rahul Sen", "Priya Nair", "Arjun Rao", "Meera Joshi"]
NOTES = ["c/o fever 3 days", "c/o cough and cold", "Dx: URTI", "c/o acidity", "BP 130/80",
         "Adv: plenty of fluids", "Review after 1 week", "c/o headache"]


def load_font(path, size):
    """Load a font file; fall back to Pillow's default font if it is missing."""
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)


def written_name(med, style):
    """Messy prescriptions often use a brand name (Dolo, Serta) instead of the generic name."""
    brands = [b for b in med["brands"].split(";") if b]
    if style == "messy" and brands and random.random() < 0.4:
        return random.choice(brands)
    return med["name"]


def make_prescription(style):
    """Return (lines_of_text, list_of_medicines) for one random prescription."""
    meds = random.sample(IN_STOCK, random.randint(2, 4))
    if random.random() < 0.3:  # about 1 in 3 prescriptions has a medicine we don't stock
        meds[-1] = random.choice(NOT_IN_STOCK)

    lines = [random.choice(DOCTORS), "City Health Clinic",
             f"Patient: {random.choice(PATIENTS)}   Age: {random.randint(18, 80)}",
             f"Date: {random.randint(1, 28):02d}/{random.randint(1, 12):02d}/2026"]
    if style == "messy":
        lines += random.sample(NOTES, 2)
    lines.append("Rx")

    medicines = []
    for i, med in enumerate(meds, start=1):
        name = written_name(med, style)
        strength = med["strength"].split("/")[0]  # a syrup "125mg/5ml" is written "125mg"
        form = random.choice(FORM_WORD[med["form"]])
        dose = random.choice(DOSE[med["form"]])
        days = random.choice([3, 5, 7, 10])
        if style == "messy" and random.random() < 0.5:
            freq = random.choice(SCHEDULES)  # dosing on its own line, under the medicine
            lines += [f"{i}) {form} {name} {strength}", f"      {freq}   x {days} days"]
        else:
            freq = random.choice(FREQUENCIES)
            lines.append(f"{i}. {form} {name} {strength} - {dose} - {freq} - {days} days")
        medicines.append({"name": med["name"], "written": name, "strength": med["strength"],
                          "form": med["form"], "frequency": freq, "in_inventory": med in IN_STOCK})
    lines.append("Signature")
    return lines, medicines


def draw_clean(lines, style):
    """Printed and handwriting-font styles (grayscale page)."""
    font = load_font(random.choice(FONTS[style]), 34 if style == "printed" else 40)
    img = Image.new("L", (1500, 300 + 75 * len(lines)), color=255)  # L = grayscale
    draw = ImageDraw.Draw(img)
    y = 60
    for line in lines:
        x = 70 + (random.randint(-10, 10) if style == "handwritten" else 0)
        draw.text((x, y), line, fill=random.randint(0, 50), font=font)
        y += 75
    if style == "handwritten":
        img = img.rotate(random.uniform(-3, 3), fillcolor=255, expand=True)  # slightly tilted photo
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 1.8)))
        # Low-quality phone camera: shrink the photo, so fine letter details are lost.
        img = img.resize((int(img.width * 0.55), int(img.height * 0.55)), Image.BILINEAR)
    return img.convert("RGB")


def draw_messy(lines):
    """A phone photo of a handwritten note: coloured ink, paper tint, wobbly words."""
    font_path = random.choice(FONTS["messy"])
    ink = random.choice([(25, 45, 140), (20, 20, 30), (40, 60, 170)])  # blue or black pen
    paper = tuple(random.randint(232, 250) for _ in range(3))
    img = Image.new("RGB", (1500, 300 + 85 * len(lines)), color=paper)
    draw = ImageDraw.Draw(img)
    y = 60
    for line in lines:
        x = 70 + random.randint(-15, 25)
        size = random.randint(38, 48)
        for word in line.split(" "):  # each word slightly up/down and a different size
            if not word:
                x += size // 3
                continue
            font = load_font(font_path, size + random.randint(-3, 3))
            draw.text((x, y + random.randint(-5, 5)), word, fill=ink, font=font)
            x += int(draw.textlength(word + " ", font=font))
        y += 85
    img = img.rotate(random.uniform(-4, 4), fillcolor=paper, expand=True)
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.8, 1.6)))
    scale = random.uniform(0.5, 0.7)
    return img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)


def add_camera_effects(img, jpeg):
    pixels = np.array(img).astype(np.float32)
    # Uneven lighting: one side of the page is darker (like a phone photo).
    shadow = np.linspace(1.0, random.uniform(0.65, 0.9), pixels.shape[1])
    pixels = pixels * shadow[np.newaxis, :, np.newaxis]
    pixels += np.random.normal(0, 10, pixels.shape)  # grain
    img = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    if jpeg:  # phone photos are JPEG-compressed, which smudges fine strokes
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=random.randint(45, 70))
        img = Image.open(buffer).convert("RGB")
    return img


def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    for old in os.listdir(SAMPLES_DIR):  # remove the previous set
        os.remove(os.path.join(SAMPLES_DIR, old))
    rows = []
    for n, style in enumerate(STYLES, start=1):
        lines, medicines = make_prescription(style)
        name = f"rx_{n:03d}"
        img = draw_messy(lines) if style == "messy" else draw_clean(lines, style)
        add_camera_effects(img, jpeg=style == "messy").save(f"{SAMPLES_DIR}/{name}.png")
        with open(f"{SAMPLES_DIR}/{name}.txt", "w") as f:
            f.write("\n".join(line.strip() for line in lines))
        for med in medicines:
            rows.append({"image": f"{name}.png", "style": style, **med})

    with open(GROUND_TRUTH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "style", "name", "written", "strength", "form",
                                               "frequency", "in_inventory"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {len(STYLES)} prescriptions in {SAMPLES_DIR}/ and {len(rows)} "
          f"answer rows in {GROUND_TRUTH}")


if __name__ == "__main__":
    main()
