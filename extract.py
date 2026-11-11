"""
extract.py
Finds medicine details in the OCR text and FLAGS anything uncertain instead of guessing.

For every medicine line we try to find:
  name       e.g. Paracetamol
  strength   e.g. 500mg
  dose       e.g. 1 tab
  frequency  e.g. BD -> "Twice a day"
  duration   e.g. 5 days
Doctors often write the instructions on the line BELOW the medicine ("1 - x - 1", "Night"),
so those lines are attached to the medicine above them.
"""
import re

from rapidfuzz import fuzz

# Words below this OCR confidence (0-100) are treated as "not sure".
CONFIDENCE_THRESHOLD = 70

# Words that describe the form of a medicine, not its name ("Ta", "Td", "Tu" = handwritten "Tab").
FORM_WORDS = {"tab", "tabs", "tablet", "cap", "caps", "capsule", "syp", "syrup", "susp", "inj",
              "injection", "ta", "td", "tu", "oint", "ointment", "cream", "gel", "drop", "drops", "lotion"}
# Words that can come before the medicine name but are not names.
NOT_NAMES = {"plan", "rx", "adv", "advice", "take"}

# Strength: a number followed by a unit, e.g. 500mg, 2.5 mg, 50mcg ("ml" is a dose, e.g. "5 ml" of syrup;
# a syrup strength like "125mg/5ml" is read as 125mg)
STRENGTH_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|iu)\b", re.IGNORECASE)
# Dose: how much to take each time, e.g. "1 tab", "2 caps", "5 ml" (the separator before
# the number stops "500mg - 1 tab" from being read as "0 tab").
DOSE_PATTERN = re.compile(r"(?:^|[\s\-~])(\d+)\s*(tabs?|caps?|tablets?|capsules?|ml|tsp)\b", re.IGNORECASE)
DURATION_PATTERN = re.compile(r"(\d+)\s*(days?|weeks?|months?)\b", re.IGNORECASE)
# Dose pattern like 1-0-1 or x-x-1 (morning-afternoon-night; "x" means none). Handwriting is often
# read with look-alikes ("o" for 0, "l" for 1) or a missing dash ("1-0 0"), so those are allowed too.
_SLOT = r"([0-2xXoOlI|])"
_SEP = r"(?:\s*[-—–~._]+\s*|\s+)"  # dash, tilde, dot or underscore ("o_0 - 1"), or just spaces
MAN_PATTERN = re.compile(rf"(?<![A-Za-z0-9]){_SLOT}{_SEP}{_SLOT}{_SEP}{_SLOT}(?![A-Za-z0-9])")
TIMES_A_DAY = re.compile(r"\b(\d)\s*times?\s*a\s*day", re.IGNORECASE)

# OCR often confuses these look-alike characters inside numbers and units.
DIGIT_LOOKALIKES = str.maketrans({"S": "5", "s": "5", "O": "0", "o": "0", "c": "0", "I": "1",
                                  "l": "1", "T": "1", "|": "1", "B": "8", ",": "."})
UNIT_LOOKALIKES = {"mq": "mg", "ma": "mg", "m9": "mg", "m@": "mg", "rng": "mg", "ms": "mg", "me": "mg",
                   "ng": "mg", "meg": "mcg", "mcq": "mcg", "mcy": "mcg", "mog": "mcg"}
LOOSE_STRENGTH = re.compile(r"^([0-9SsOoIlTBc|]{1,5}(?:[.,][0-9OoSs]{1,2})?)"
                            r"(mq|ma|m9|m@|rng|ms|me|ng|mg|meg|mcg|mcq|mcy|mog)$", re.IGNORECASE)

# Doctor's shorthand -> plain English (checked in this order)
FREQUENCIES = [
    (r"\b(thrice daily|three times|tds|tid)\b", "Three times a day"),
    (r"\b(twice daily|two times|bd[ax]?|bid)\b", "Twice a day"),
    (r"\b(four times|qid|qds)\b", "Four times a day"),
    (r"\b(once daily|once a day|od|qd)\b", "Once a day"),
    (r"\b(at night|bedtime|hs)\b", "At bedtime"),
    (r"\b(sos|prn|when needed|as needed)\b", "Only when needed"),
]


def clean_word(word):
    """Remove symbols around a word: '(Lorazepam)' -> 'Lorazepam', 'Plan→' -> 'Plan'."""
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", word)


def find_frequency(text):
    """Return the frequency in plain English, or None if we can't tell."""
    lower = text.lower()
    for pattern, meaning in FREQUENCIES:
        if re.search(pattern, lower):
            return meaning
    times = TIMES_A_DAY.search(text)
    if times:
        n = int(times.group(1))
        return {1: "Once a day", 2: "Twice a day", 3: "Three times a day"}.get(n, f"{n} times a day")
    man = MAN_PATTERN.search(text)
    if man:
        slots = [n.translate(str.maketrans("lI|", "111")) for n in man.groups()]
        parts = [when for when, n in zip(["in the morning", "in the afternoon", "at night"], slots) if n in "12"]
        return (" and ".join(parts)).capitalize() if parts else None
    # Handwritten "night" is often misread ("nigw", "INighL"), so allow a close spelling.
    if any(fuzz.ratio(clean_word(w).lower(), "night") >= 65 for w in text.split()):
        return "At night"
    return None


def schedule_units(text):
    """Tablets per day from a schedule like '1-0-1' (= 2) or '2-x-1' (= 3). None if there is no schedule."""
    man = MAN_PATTERN.search(text)
    if not man:
        return None
    slots = [n.translate(str.maketrans("lI|", "111")) for n in man.groups()]
    return sum(int(n) for n in slots if n in "12")


def has_form_word(text):
    words = [clean_word(w).lower() for w in text.split()]
    words = [w for w in words if w and not w.isdigit()]  # skip "1." or "(3)" numbering
    return any(w in FORM_WORDS for w in words[:4])  # e.g. "Tab X", "Moxicip eye drops"


def find_name(words):
    """
    The medicine name is the first real word after the number/form word.
    words = [(word, confidence), ...]. Returns (name, confidence) or (None, 0).
    """
    for word, conf in words:
        clean = clean_word(word)
        if not clean or clean.lower() in FORM_WORDS or clean.lower() in NOT_NAMES:
            continue
        if clean.isdigit():  # skip "1." numbering
            continue
        if STRENGTH_PATTERN.fullmatch(clean) or loose_strength(clean):  # reached the strength: no name
            return None, 0
        if len(clean) >= 3 and re.search(r"[A-Za-z]", clean):
            return clean, conf
    return None, 0


def loose_strength(word):
    """
    Repair a strength that OCR misread, e.g. 'S00m@' -> '500mg', '4mq' -> '4mg', '0.5ms' -> '0.5mg'.
    Only look-alike characters are swapped. Returns the strength or None.
    """
    found = LOOSE_STRENGTH.match(clean_word(word))
    if not found:
        return None
    raw_number, raw_unit = found.group(1), found.group(2).lower()
    # With a look-alike unit we also need a real digit, otherwise "Some" would become "50mg".
    if raw_unit in UNIT_LOOKALIKES and not re.search(r"\d", raw_number):
        return None
    try:
        value = float(raw_number.translate(DIGIT_LOOKALIKES))
    except ValueError:
        return None
    return f"{value:g}{UNIT_LOOKALIKES.get(raw_unit, raw_unit)}" if value > 0 else None


def find_strength(text, correct):
    """Returns (strength, corrected_from_word). corrected_from_word is None when read cleanly."""
    found = STRENGTH_PATTERN.search(text)
    if found:
        return found.group(1) + found.group(2).lower(), None
    if correct:
        for word in text.split():
            strength = loose_strength(word)
            if strength:
                return strength, clean_word(word)
    return None, None


def is_below(line, med_line):
    """True if `line` sits just under `med_line` on the page and overlaps it horizontally."""
    x, y, w, h = line["box"]
    mx, my, mw, mh = med_line["box"]
    centre = x + w / 2
    return y > my and y - my < max(4 * mh, 0.08) and mx - 0.05 <= centre <= mx + mw + 0.05


def extract_medicines(ocr_result, correct=True):
    """
    Turn OCR output (from ocr.read_text) into a list of medicine dicts.
    Each dict has a "flags" list explaining anything that needs a human to check.
    correct=False switches off the strength correction (used for the baseline in evaluate.py).
    """
    lines = ocr_result["lines"]
    medicines, used = [], set()

    # Pass 1: find the medicine lines.
    for i, line in enumerate(lines):
        text = line["text"]
        name, name_conf = find_name(line["words"])
        strength, corrected = find_strength(text, correct)
        if name is None and not has_form_word(text):
            continue  # e.g. a strength written alone under a medicine: handled in pass 2
        if not (has_form_word(text) or strength):
            continue
        dose = DOSE_PATTERN.search(text)
        duration = DURATION_PATTERN.search(text)
        scores = [c for w, c in line["words"] if re.search(r"[A-Za-z0-9]", w)]  # ignore "-", "~"
        medicines.append({
            "name": name, "name_confidence": name_conf, "strength": strength, "corrected_from": corrected,
            "dose": f"{dose.group(1)} {dose.group(2).lower()}" if dose else None,
            "frequency": find_frequency(text), "schedule_units": schedule_units(text),
            "duration": f"{duration.group(1)} {duration.group(2).lower()}" if duration else None,
            "line": text, "line_conf": min(scores) if scores else 0, "_line": line,
            "boxes": [line["box"]],  # where it is on the page, to draw a box around it
        })
        used.add(i)

    # Pass 2: attach instructions written on the lines just below a medicine.
    for i, line in enumerate(lines):
        if i in used:
            continue
        owners = [m for m in medicines if is_below(line, m["_line"])]
        if not owners:
            continue
        med = max(owners, key=lambda m: m["_line"]["box"][1])  # the nearest medicine above
        text = line["text"]
        combined = med["line"] + " " + text  # e.g. "5 Times a" + "day"
        useful = (find_frequency(text) or DURATION_PATTERN.search(text) or DOSE_PATTERN.search(text)
                  or find_strength(text, correct)[0] or (med["frequency"] is None and find_frequency(combined)))
        if not useful:  # e.g. "Signature" or a note: not part of the medicine
            continue
        if med["frequency"] is None:
            med["frequency"] = find_frequency(combined)
        if med["schedule_units"] is None:
            med["schedule_units"] = schedule_units(text)
        if med["duration"] is None and DURATION_PATTERN.search(text):
            found = DURATION_PATTERN.search(text)
            med["duration"] = f"{found.group(1)} {found.group(2).lower()}"
        if med["strength"] is None:
            med["strength"], med["corrected_from"] = find_strength(text, correct)
        med["line"] += " / " + text
        med["boxes"].append(line["box"])

    # Pass 3: flag everything uncertain.
    for med in medicines:
        flags = []
        if med["name"] is None:
            flags.append("Medicine name not found")
        elif med["name_confidence"] < CONFIDENCE_THRESHOLD:
            flags.append(f"Name unclear (OCR confidence {med['name_confidence']:.0f}%)")
        if med["corrected_from"]:
            flags.append(f"Strength auto-corrected from '{med['corrected_from']}' to {med['strength']}: please confirm")
        if med["strength"] is None:
            flags.append("Strength missing or unreadable")
        if med["frequency"] is None:
            flags.append("Frequency missing or unreadable")
        if med["line_conf"] < CONFIDENCE_THRESHOLD and not flags:
            flags.append(f"Part of this line is unclear (lowest confidence {med['line_conf']:.0f}%)")
        med["flags"] = flags
        del med["_line"], med["corrected_from"], med["line_conf"]
    return medicines


def plain_language(med):
    """A short, easy-to-read sentence for patients, e.g. 'Paracetamol 500mg: take 1 tab, twice a day, for 5 days.'"""
    if med["dose"]:
        dose = f"take {med['dose']}"
    elif med.get("schedule_units"):  # a schedule like "1 - 0 - 1" already gives the amount
        dose = f"take {med['schedule_units']} a day"
    elif re.search(r"\b(cream|ointment|oint|gel|lotion)\b", med["line"], re.IGNORECASE):
        dose = "apply"
    elif re.search(r"\bdrops?\b", med["line"], re.IGNORECASE):
        dose = "use as drops"
    else:
        dose = "dose not clear"
    parts = [dose, med["frequency"].lower() if med["frequency"] else "how often is not clear"]
    if med["duration"]:
        parts.append(f"for {med['duration']}")
    title = f"{med['name'] or 'Unknown medicine'} {med['strength'] or ''}".strip()
    return f"{title}: " + ", ".join(parts) + "."


if __name__ == "__main__":
    # Quick test: python extract.py [image path]
    import sys

    import cv2
    from ocr import available_engines, read_text
    path = sys.argv[1] if len(sys.argv) > 1 else "data/samples/rx_47.png"
    ocr_result = read_text(cv2.imread(path), available_engines()[0])
    for med in extract_medicines(ocr_result):
        print(plain_language(med), "| flags:", med["flags"] or "none")
