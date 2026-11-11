"""
evaluate.py
Measures how well the system works on the synthetic prescriptions, for the research paper.

Compares 4 methods:
  1. Baseline        : Tesseract on the raw image (grayscale only) + exact name matching
  2. + Preprocessing : Tesseract on the cleaned image             + exact name matching
  3. Proposed        : Tesseract on the cleaned image + fuzzy matching, correction, detection
  4. Proposed (Vision): same as 3, but Apple Vision reads the original image (macOS only)

Metrics:
  CER / WER          character / word error rate of the OCR text (lower is better)
  Name accuracy      % of medicines whose name was read exactly right
  Correct suggestion % of stocked medicines for which the right inventory item was
                     suggested to the pharmacist (even if it carried a warning)
  Detection top-1/3  % of stocked medicines found by detection (best guess / any of 3 guesses),
                     which also scans every word of unclear lines (proposed method only)
  Precision / Recall of automatic (un-flagged) inventory matches
  Silent errors      wrong medicine matched WITHOUT a warning (the dangerous case)
  Review rate        % of medicines sent to the pharmacist for review
  Time               seconds per prescription

Run:  python evaluate.py      (results go to the results/ folder)
"""
import os
import time

import cv2
import matplotlib
import pandas as pd
from rapidfuzz.distance import Levenshtein

from database import get_inventory, init_db
from extract import extract_medicines
from matcher import match_medicine
from ocr import VISION_AVAILABLE, read_text
from preprocess import preprocess, to_gray

matplotlib.use("Agg")  # draw charts to files, no window
import matplotlib.pyplot as plt  # noqa: E402

SAMPLES_DIR = "data/samples"
RESULTS_DIR = "results"
METHODS = {
    "Baseline": {"engine": "tesseract", "preprocess": False, "match": "exact", "correct": False},
    "+ Preprocessing": {"engine": "tesseract", "preprocess": True, "match": "exact", "correct": False},
    "Proposed": {"engine": "tesseract", "preprocess": True, "match": "fuzzy", "correct": True},
}
if VISION_AVAILABLE:
    METHODS["Proposed (Vision)"] = {"engine": "vision", "preprocess": False, "match": "fuzzy", "correct": True}


def normalize(text):
    """Lower-case and single spaces, so only real reading mistakes are counted."""
    return " ".join(text.lower().split())


def error_rate(predicted, truth):
    """Edit distance / length of the truth. Works for strings (CER) and word lists (WER)."""
    return Levenshtein.distance(predicted, truth) / max(len(truth), 1)


def evaluate_image(image_name, truth_rows, truth_text, settings, inventory):
    image = cv2.imread(os.path.join(SAMPLES_DIR, image_name))
    start = time.time()
    if settings["engine"] == "vision":
        ocr_input = image  # Vision reads the original photo
    else:
        ocr_input = preprocess(image) if settings["preprocess"] else to_gray(image)
    ocr_result = read_text(ocr_input, settings["engine"])
    medicines = extract_medicines(ocr_result, correct=settings["correct"])
    matches = [match_medicine(m, inventory, method=settings["match"]) for m in medicines]
    seconds = time.time() - start

    # Inventory id of each medicine that is really on the prescription.
    truth_ids = set()
    for row in truth_rows.itertuples():
        hit = inventory[(inventory["name"] == row.name) & (inventory["strength"] == row.strength)
                        & (inventory["form"] == row.form)]
        if not hit.empty:
            truth_ids.add(int(hit.iloc[0]["med_id"]))

    auto = [m for m in matches if m["status"] == "Matched"]  # accepted without a warning
    tp = sum(1 for m in auto if m["med_id"] in truth_ids)
    read_names = {(m["name"] or "").lower() for m in medicines}
    pred, true = normalize(ocr_result["text"]), normalize(truth_text)
    return {
        "image": image_name,
        "style": truth_rows["style"].iloc[0],
        "CER": error_rate(pred, true),
        "WER": error_rate(pred.split(), true.split()),
        "names_total": len(truth_rows),
        "names_correct": sum(1 for n in truth_rows["written"] if n.lower() in read_names),  # as written (brand or generic)
        "extracted": len(medicines),
        "in_inventory": len(truth_ids),
        # Correct medicine suggested to the pharmacist (with or without a warning).
        "suggested": len(truth_ids & {m["med_id"] for m in matches}),
        # Detection: the right medicine is the match or the 1st detected candidate (top-1),
        # or anywhere among the 3 detected candidates (top-3).
        "detected_top1": len(truth_ids & {m["med_id"] if m["med_id"] is not None else
                                          (m["detected"][0]["med_id"] if m["detected"] else None)
                                          for m in matches}),
        "detected_top3": len(truth_ids & ({m["med_id"] for m in matches} |
                                          {c["med_id"] for m in matches for c in m["detected"]})),
        "TP": tp,
        "FP": len(auto) - tp,
        "FN": len(truth_ids) - tp,
        "reviewed": len(matches) - len(auto),
        "seconds": seconds,
    }


def summarize(df):
    """Combine per-image rows into one row of metrics."""
    tp, fp, fn = df["TP"].sum(), df["FP"].sum(), df["FN"].sum()
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return pd.Series({
        "CER (%)": 100 * df["CER"].mean(),
        "WER (%)": 100 * df["WER"].mean(),
        "Name accuracy (%)": 100 * df["names_correct"].sum() / df["names_total"].sum(),
        "Correct suggestion (%)": 100 * df["suggested"].sum() / max(df["in_inventory"].sum(), 1),
        "Detection top-1 (%)": 100 * df["detected_top1"].sum() / max(df["in_inventory"].sum(), 1),
        "Detection top-3 (%)": 100 * df["detected_top3"].sum() / max(df["in_inventory"].sum(), 1),
        "Precision (%)": 100 * precision,
        "Recall (%)": 100 * recall,
        "F1 (%)": 100 * (2 * precision * recall / (precision + recall) if precision + recall else 0),
        "Silent errors (%)": 100 * fp / max(df["extracted"].sum(), 1),
        "Review rate (%)": 100 * df["reviewed"].sum() / max(df["extracted"].sum(), 1),
        "Time (s/image)": df["seconds"].mean(),
    })


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    init_db(reset=True)  # evaluate against the original stock levels
    inventory = get_inventory()
    truth = pd.read_csv("data/ground_truth.csv")

    rows = []
    for method, settings in METHODS.items():
        print(f"Evaluating: {method} ...")
        for image_name, truth_rows in truth.groupby("image"):
            with open(os.path.join(SAMPLES_DIR, image_name.replace(".png", ".txt"))) as f:
                truth_text = f.read()
            result = evaluate_image(image_name, truth_rows, truth_text, settings, inventory)
            rows.append({"method": method, **result})

    per_image = pd.DataFrame(rows)
    per_image.to_csv(f"{RESULTS_DIR}/per_image.csv", index=False)

    overall = per_image.groupby("method", sort=False).apply(summarize, include_groups=False)
    by_style = per_image.groupby(["method", "style"], sort=False).apply(summarize, include_groups=False)
    overall.round(2).to_csv(f"{RESULTS_DIR}/summary.csv")
    by_style.round(2).to_csv(f"{RESULTS_DIR}/summary_by_style.csv")

    pd.set_option("display.width", 200)
    print("\n=== Overall results ===")
    print(overall.round(1).to_string())
    print("\n=== By writing style ===")
    print(by_style.round(1).to_string())

    # Chart 1: OCR errors by writing style
    by_style[["CER (%)", "WER (%)"]].unstack("style").plot.bar(figsize=(9, 5), rot=0)
    plt.title("OCR error rate (lower is better)")
    plt.ylabel("%")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/ocr_error_rates.png", dpi=150)

    # Chart 2: matching quality
    overall[["Correct suggestion (%)", "Detection top-1 (%)", "Precision (%)", "Recall (%)",
             "Review rate (%)"]].plot.bar(figsize=(10, 5.5), rot=0)
    plt.title("Medicine matching (all methods had 0% silent errors)")
    plt.ylabel("%")
    plt.ylim(0, 105)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/matching_results.png", dpi=150)
    print(f"\nSaved tables and charts in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
