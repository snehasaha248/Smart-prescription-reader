# Smart Prescription Reader and Accessible Medicine Information System

B.Tech final-year project. The app reads a prescription image, extracts medicine details,
**flags anything uncertain instead of guessing**, matches medicines to a pharmacy inventory,
and requires a pharmacist to verify everything before dispensing.

> **Note:** For education and demonstration only. Uses synthetic prescriptions and dummy inventory data.
> It does not make medical decisions.

## How it works
```
Upload image → ocr.py (image to text) → extract.py → editable medicine text → matcher.py (+ database.py)
            → detection of unclear lines → app.py dashboard → pharmacist verifies and dispenses
```
The dashboard shows the original photo and, below it, every identified medicine **marked in a numbered box on the processed image** (green = matched,
red = not found; the numbers match the medicine list), next to the **scanned medicines and dosage** only (doctor, patient and address
details are left out). The text can be corrected by hand and downloaded as a `.txt` file; the medicine list
updates from the corrected text.

Below that, every scanned medicine gets a **compact card**: name, dosage in plain language, the best match in
stock with its shelf and stock level, and the **recommended number of strips (or bottles)**. A medicine is shown
as **Matched** when a stock medicine is at least 60% likely to be it, otherwise **Not found**. Clicking the arrow on a card expands it to show:
- what the medicine is used for,
- how the quantity was worked out (e.g. 2 tablets x 2 a day x 10 days = 40 tablets = 4 strips of 10),
- the **matching medicines in stock** that are at least **60% likely** (up to 3, best first), with stock level and shelf,
- any notes that need the pharmacist's attention.

The recommended strips are pre-filled in the pharmacist form; the pharmacist confirms or changes them before dispensing.

| File | Job |
|---|---|
| `app.py` | Streamlit dashboard that connects everything |
| `style.css` | Dashboard styling (cards, header, tiles) |
| `.streamlit/config.toml` | Dashboard colour theme |
| `preprocess.py` | OpenCV image cleanup: grayscale, lighting fix, denoise, threshold, deskew |
| `ocr.py` | Image-to-text converter with two engines: **Apple Vision** (built into macOS, reads handwriting and phone photos) and **Tesseract** (any computer, printed text). Gives text with confidence scores |
| `extract.py` | Finds name, strength, dose, frequency and duration; attaches instructions written on the line below; flags uncertain items; corrects OCR look-alike characters in strengths (`50me` → 50mg) |
| `database.py` | SQLite inventory and dispense log |
| `dosage.py` | Recommends how many strips/bottles to give: amount per dose x doses per day x days, divided by the pack size and rounded up |
| `matcher.py` | Exact (baseline) or fuzzy (proposed) matching against generic **and brand** names (Serta → Sertraline), plus **detection** of unclear lines: every word on the line is compared with the inventory |
| `make_samples.py` | Generates 100 synthetic prescriptions (printed, handwriting-style and messy handwritten) and the answer key |
| `evaluate.py` | Research evaluation: CER, WER, accuracy, precision/recall, time |
| `run.sh` | Shortcut that starts the dashboard |

## Dataset
- `data/medicines.csv` has **208 inventory items** across **21 categories** (pain relief, antibiotics, diabetes,
  blood pressure, heart, mental health, cough and cold, skin, eye and ear and more): 157 tablets, 23 capsules,
  **20 syrups** (e.g. Paracetamol 125mg/5ml, Ambroxol 15mg/5ml) on a syrup rack (S shelves), and 8 creams,
  ointments and eye drops on a topical rack (T shelves). Each item has its common Indian **brand names**, its
  **pack** (strip of 10 or 15, bottle in ml, tube), a plain-language "used for" description, a stock level and a shelf location.
  A prescription line like `Syp Calpol 125mg - 5 ml` is matched to the syrup (not the tablet), and `5 ml` is read as the dose.
- `data/samples/` has **100 synthetic prescriptions** (300 medicines):
  - rx_001–030 **printed**
  - rx_031–060 **handwriting-style** fonts with blur, tilt, shadows and low resolution
  - rx_061–100 **messy handwritten**: blue/black ink on paper, wobbly words, cursive fonts, brand names
    (Dolo, Serta), syrups, dosing written on the line below ("1 - 0 - 1 x 5 days"), diagnosis notes, JPEG blur
  About 1 in 3 prescriptions includes a medicine that is not in the inventory.
- `data/ground_truth.csv` is the answer key: the generic name, the name as written (brand or generic), strength and form.

## Setup
1. Install **Tesseract OCR** and **Python 3.11** (Apple Vision needs no install: it is part of macOS).
   The original setup used micromamba:
   ```bash
   micromamba create -n rx -c conda-forge python=3.11 tesseract
   micromamba activate rx
   ```
   On macOS with Homebrew, `brew install tesseract` also works. On Windows, use the UB-Mannheim Tesseract installer.
2. Install the Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Run
```bash
streamlit run app.py          # dashboard at http://localhost:8501  (or ./run.sh)
python make_samples.py        # (re)create synthetic prescriptions
python evaluate.py            # research results -> results/
```
Each module can also be tested on its own, e.g. `python ocr.py` or `python matcher.py`.

## Results (100 synthetic prescriptions, 300 medicines)
| Method | CER % | WER % | Name acc. % | Correct suggestion % | Detection top-1 % | Detection top-3 % | Precision % | Recall % | Silent errors % | Review rate % |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (Tesseract, raw image + exact match) | 39.8 | 65.7 | 49.3 | 43.6 | 43.6 | 43.6 | 100 | 17.3 | 0 | 78.4 |
| + Preprocessing | 10.8 | 25.4 | 77.0 | 61.3 | 61.3 | 61.3 | 100 | 37.2 | 0 | 63.3 |
| **Proposed** (+ fuzzy/brand match, strength correction, detection) | 10.8 | 25.4 | 77.0 | 70.7 | 86.1 | 87.6 | 100 | 37.6 | 0 | 63.0 |
| **Proposed (Apple Vision OCR)** | **2.2** | **8.5** | **93.7** | **94.0** | **98.9** | **100** | 100 | **71.0** | 0 | **36.8** |

Messy handwritten prescriptions only:
| Method | CER % | Correct suggestion % | Detection top-1 % | Review rate % |
|---|---|---|---|---|
| Baseline | 47.7 | 25.0 | 25.0 | 86.2 |
| Proposed (Tesseract) | 24.3 | 45.4 | 67.6 | 93.0 |
| **Proposed (Apple Vision)** | **5.4** | **85.2** | **97.2** | **60.9** |

No method ever matched a wrong medicine without a warning (0% silent errors). Messy prescriptions still go to
review often, mostly by design: brand names are always confirmed by the pharmacist, and some items are out of stock.
Detected medicines are always shown as suggestions for the pharmacist, never dispensed automatically.

**Real handwritten prescriptions (qualitative check, images not included for privacy):** on two real
handwritten prescriptions with 7 medicines, the Apple Vision pipeline identified all 7 medicines, including
brand names (Serta → Sertraline, Rivotril → Clonazepam, Ativan → Lorazepam). Tesseract read almost none of them.
Full per-style results are in `results/summary_by_style.csv`. Charts are `results/ocr_error_rates.png` and `results/matching_results.png`.

## Limitations
- The evaluation data is synthetic: handwriting-style fonts, not real doctors' handwriting. Tesseract performs poorly on real handwriting.
- Apple Vision only runs on macOS; on other computers the app falls back to Tesseract.
- Very messy handwriting can still be misread (e.g. a strength read as 900mg instead of 200mg); such items are flagged for the pharmacist.
- The dataset is small (100 synthetic images, plus 2 real prescriptions), so results are indicative only.
- No novelty is claimed without a literature review.
