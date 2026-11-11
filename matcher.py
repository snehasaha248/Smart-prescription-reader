"""
matcher.py
Matches a medicine name read by OCR to a medicine in the inventory.

Two methods (we compare them in the research evaluation):
  "exact" - baseline: the name must be spelled exactly right
  "fuzzy" - proposed: allows small OCR spelling mistakes ("Paracetmol" -> "Paracetamol"),
            using a similarity score from 0 to 100 (RapidFuzz library)
Doctors often write BRAND names (Serta, Dolo, Ativan). Each inventory row lists its brands,
so a brand is matched to its generic medicine (Serta -> Sertraline).
"""
import re

from rapidfuzz import fuzz

MATCH_THRESHOLD = 90   # score >= 90: confident match
REVIEW_THRESHOLD = 75  # score 75-89: possible match, a human must check
DETECT_MIN_SCORE = 60  # likelihood needed before a stock medicine is shown as a possible match
SHORT_NAME = 5         # names shorter than this (Dolo, Pan) must be spelled exactly

# Words on a prescription line that are never medicine names.
NOT_NAMES = {"tab", "tabs", "tablet", "cap", "caps", "capsule", "syp", "syrup", "inj", "daily", "days",
             "once", "twice", "thrice", "night", "week", "weeks", "morning", "when", "needed",
             "plan", "times", "ointment", "cream", "drops", "plus"}


def base_strength(strength):
    """Syrups are stored as '125mg/5ml'; a prescription usually just says '125mg'."""
    return str(strength or "").lower().replace(" ", "").split("/")[0]


def is_syrup_line(med):
    return bool(re.search(r"\b(syp|syr|syrup|susp|suspension)\b", med.get("line", ""), re.IGNORECASE))


def pick_form(rows, med):
    """Same name and strength in two forms (Cetirizine 5mg tablet / 5mg/5ml syrup): use the line's form."""
    if len(rows) > 1:
        syrup = rows["form"] == "Syrup"
        preferred = rows[syrup] if is_syrup_line(med) else rows[~syrup]
        if not preferred.empty:
            return preferred
    return rows


def aliases(inventory):
    """Every name a medicine can be written as: [(alias, generic name, is_brand), ...]."""
    result = []
    for name, brands in inventory[["name", "brands"]].drop_duplicates("name").itertuples(index=False):
        result.append((name, name, False))
        for brand in str(brands or "").split(";"):
            if brand.strip():
                result.append((brand.strip(), name, True))
    return result


def similarity(word, alias):
    """0-100 similarity between a written word and a medicine name. Short names must match exactly."""
    word, alias = word.lower(), alias.lower()
    if len(alias) < SHORT_NAME or len(word) < SHORT_NAME:
        return 100 if word == alias else 0
    return fuzz.ratio(word, alias)


def detect_medicine(med, inventory, top=3):
    """
    Detection for unclear (e.g. handwritten) lines: compare EVERY word on the line with every
    inventory medicine and brand name, then use the strength as a tie-breaker.
    Returns up to `top` candidates, best first: [{"med_id", "label", "score", ...}, ...].
    These are suggestions for the pharmacist, never automatic matches.
    """
    words = {w.lower() for w in re.findall(r"[A-Za-z]{3,}", med.get("line", med["name"] or ""))}
    words -= NOT_NAMES
    if med["name"]:
        words.add(med["name"].lower())
    if not words:
        return []

    best_name = {}  # generic name -> best similarity of any word to its name or brands
    for alias, name, _ in aliases(inventory):
        score = max(similarity(w, alias) for w in words)
        best_name[name] = max(best_name.get(name, 0), score)

    candidates = []
    for row in inventory.itertuples():
        name_score = best_name.get(row.name, 0)
        if name_score < DETECT_MIN_SCORE:
            continue
        # The likelihood shown is the NAME similarity; a different strength lowers it,
        # the same strength only puts this row first among the medicine's strengths.
        same_strength = bool(med["strength"]) and base_strength(med["strength"]) == base_strength(row.strength)
        same_form = (row.form == "Syrup") == is_syrup_line(med)
        score = name_score - (10 if med["strength"] and not same_strength else 0)
        if score < DETECT_MIN_SCORE:
            continue
        label = f"{row.name} {row.strength}" + (" syrup" if row.form == "Syrup" else "")
        candidates.append({"med_id": int(row.med_id), "label": label,
                           "score": int(max(0, round(score))), "same_strength": same_strength,
                           "same_form": same_form, "stock": int(row.stock), "shelf": row.shelf, "use": row.use})
    candidates.sort(key=lambda c: (c["score"], c["same_strength"], c["same_form"]), reverse=True)
    return candidates[:top]


def match_medicine(med, inventory, method="fuzzy"):
    """
    med:       one dict from extract.extract_medicines
    inventory: DataFrame from database.get_inventory
    Returns a dict with status "Matched", "Needs review" or "Not found", and "likely":
    up to 3 stock medicines that are at least 60% likely, best first (shown on the dashboard).
    """
    result = _match(med, inventory, method)
    likely = []
    if result["med_id"] is not None:  # the direct match comes first
        row = inventory[inventory["med_id"] == result["med_id"]].iloc[0]
        likely.append({"med_id": result["med_id"], "label": result["matched_name"], "score": result["score"],
                       "stock": int(row["stock"]), "shelf": row["shelf"]})
    likely += [c for c in result["detected"] if c["med_id"] != result["med_id"]]
    result["likely"] = [c for c in likely if c["score"] >= DETECT_MIN_SCORE][:3]
    return result


def _match(med, inventory, method):
    result = {"med_id": None, "matched_name": None, "score": 0, "stock": None,
              "shelf": None, "use": None, "status": "Not found", "flags": list(med["flags"]),
              "detected": detect_medicine(med, inventory) if method == "fuzzy" else []}
    if not med["name"]:
        return result

    written = med["name"].lower()
    best, score, is_brand = None, 0, False
    for alias, name, brand in aliases(inventory):
        s = (100 if written == alias.lower() else 0) if method == "exact" else similarity(written, alias)
        if s > score:
            best, score, is_brand = name, round(s), brand

    result["score"] = score
    if best is None or score < REVIEW_THRESHOLD:
        result["flags"].append("Not in inventory (or name unreadable)")
        return result

    # Same name can exist in several strengths (Paracetamol 500mg / 650mg): pick the right one.
    # Without a readable strength we do NOT guess one: the pharmacist must choose.
    rows = inventory[inventory["name"] == best]
    result["use"] = rows.iloc[0]["use"]  # what the medicine is for, shown to the patient
    if is_brand:
        result["flags"].append(f"Brand name '{med['name']}' read as {best}: please confirm")
    if med["strength"]:
        rows = pick_form(rows[rows["strength"].map(base_strength) == base_strength(med["strength"])], med)
    if rows.empty or not med["strength"]:
        if med["strength"]:
            result["flags"].append(f"{best} found, but not in strength {med['strength']}")
        result["matched_name"] = best
        result["status"] = "Needs review"
        return result

    row = rows.iloc[0]
    result.update(med_id=int(row["med_id"]), matched_name=f"{row['name']} {row['strength']}" + (" syrup" if row["form"] == "Syrup" else ""),
                  stock=int(row["stock"]), shelf=row["shelf"])

    if score < MATCH_THRESHOLD:
        result["flags"].append(f"Spelling differs from '{best}' (similarity {score}%)")
    if row["stock"] == 0:
        result["flags"].append("Out of stock")
    result["status"] = "Matched" if not result["flags"] else "Needs review"
    return result


if __name__ == "__main__":
    # Quick test: python matcher.py
    from database import get_inventory, init_db
    init_db()
    inventory = get_inventory()
    tests = [("Paracetamol", "500mg"), ("Paracetmol", "500mg"), ("Amoxycilin", "500mg"),
             ("Rabeprazole", "20mg"), ("SERTA", "50mg"), ("Dolo", "650mg")]
    for name, strength in tests:
        med = {"name": name, "strength": strength, "flags": []}
        r = match_medicine(med, inventory)
        print(f"{name:12} -> {r['status']:12} {r['matched_name']}  score={r['score']}  {r['flags']}")
    # Detection on a badly read handwritten line ("Tab" read as "Tub", so the name was missed)
    line = {"name": "Tub", "strength": "2mg", "flags": [], "line": "2. Tub Loperamide 2mg - 1 tab - once daily"}
    print("Detected:", [(c["label"], c["score"]) for c in detect_medicine(line, inventory)])
