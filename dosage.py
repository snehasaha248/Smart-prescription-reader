"""
dosage.py
Recommends how many strips (or bottles) to give, from the dosage on the prescription.

  total needed = amount per dose x doses per day x number of days
  strips       = total needed / tablets per strip, rounded UP

Example: Paracetamol 1 tab, twice a day, for 7 days -> 1 x 2 x 7 = 14 tablets
         -> 2 strips of 10 (20 tablets).
This is only a suggestion; the pharmacist decides the final quantity.
"""
import math
import re

# Plain-language frequency (from extract.py) -> doses per day
DOSES_PER_DAY = {"once a day": 1, "twice a day": 2, "three times a day": 3, "four times a day": 4,
                 "at bedtime": 1, "at night": 1}


def doses_per_day(frequency):
    if not frequency:
        return None
    text = frequency.lower()
    if text in DOSES_PER_DAY:
        return DOSES_PER_DAY[text]
    times = re.match(r"(\d+) times a day", text)
    if times:
        return int(times.group(1))
    if text.startswith("in the") or text.startswith("at night"):  # "In the morning and at night"
        return text.count(" and ") + 1
    return None  # e.g. "Only when needed"


def number_of_days(duration):
    found = re.match(r"(\d+) (day|week|month)", duration or "")
    if not found:
        return None
    return int(found.group(1)) * {"day": 1, "week": 7, "month": 30}[found.group(2)]


def amount(n, unit):
    """amount(14, "tablet") -> '14 tablets', amount(75, "ml") -> '75 ml'."""
    return f"{n:g} ml" if unit == "ml" else f"{n:g} {unit}{'' if n == 1 else 's'}"


def recommend(med, product=None):
    """
    med:     one dict from extract.extract_medicines
    product: the matching inventory row (dict with "form", "pack", "pack_size"), or None
    Returns {"packs": number or None, "pack": "strip"/"bottle"/"tube", "text": short summary,
             "detail": how it was worked out}.
    """
    form = product["form"] if product else ("Syrup" if "ml" in (med["dose"] or "") else "Tablet")
    pack = product["pack"] if product else ("bottle" if form == "Syrup" else "strip")
    size = int(product["pack_size"]) if product else (60 if form == "Syrup" else 10)

    def result(packs, detail):
        word = pack if packs == 1 else pack + "s"
        return {"packs": packs, "pack": pack, "text": f"{packs} {word}", "detail": detail}

    if form in ("Cream", "Ointment", "Drops"):
        return result(1, f"Creams, ointments and drops: usually 1 {pack} is enough.")

    unit = "ml" if form == "Syrup" else ("tablet" if form == "Tablet" else "capsule")
    days = number_of_days(med["duration"])
    if med["frequency"] == "Only when needed":
        return result(1, f"Taken only when needed, so the exact amount is unknown: 1 {pack} suggested.")
    if days is None:
        return {"packs": None, "pack": pack, "text": "Duration not written",
                "detail": "The number of days is missing, so the quantity cannot be worked out. "
                          "Please check with the prescriber."}

    written = re.match(r"(\d+(?:\.\d+)?)", med["dose"] or "")
    assumed = "" if written else " (assumed, not written)"
    if med.get("schedule_units") and form != "Syrup":  # "2 - x - 1" = 3 tablets a day
        per_day, how = med["schedule_units"], f"{amount(med['schedule_units'], unit)} a day (from the schedule)"
    elif med.get("schedule_units"):  # syrup "0 - 0 - 1" = 1 dose a day, each dose in ml
        per_dose = float(written.group(1)) if written else 5
        per_day = per_dose * med["schedule_units"]
        how = f"{amount(per_dose, unit)} per dose{assumed} x {med['schedule_units']} a day (from the schedule)"
    else:
        times = doses_per_day(med["frequency"])
        if times is None:
            return {"packs": None, "pack": pack, "text": "Frequency unclear",
                    "detail": "How often to take it is not clear, so the quantity cannot be worked out."}
        per_dose = float(written.group(1)) if written else (5 if form == "Syrup" else 1)
        per_day = per_dose * times
        how = f"{amount(per_dose, unit)} per dose{assumed} x {times} a day"

    total = per_day * days
    packs = max(1, math.ceil(total / size))
    return result(packs, f"{how} x {days} days = {amount(total, unit)}. One {pack} has "
                         f"{amount(size, unit)}, so {packs} {pack}{'' if packs == 1 else 's'} "
                         f"({amount(packs * size, unit)}).")


if __name__ == "__main__":
    # Quick test: python dosage.py
    tests = [
        ({"dose": "1 tab", "frequency": "Twice a day", "duration": "7 days"}, {"form": "Tablet", "pack": "strip", "pack_size": 10}),
        ({"dose": None, "frequency": "In the morning and at night", "duration": "10 days", "schedule_units": 3},
         {"form": "Tablet", "pack": "strip", "pack_size": 15}),
        ({"dose": "5 ml", "frequency": "Three times a day", "duration": "5 days"}, {"form": "Syrup", "pack": "bottle", "pack_size": 60}),
        ({"dose": "1 tab", "frequency": "Only when needed", "duration": "5 days"}, None),
        ({"dose": "1 tab", "frequency": "Once a day", "duration": None}, None),
    ]
    for med, product in tests:
        r = recommend(med, product)
        print(f"{r['text']:22} | {r['detail']}")
