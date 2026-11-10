# Insurance Document Intelligence for Automated Policy and Claim Intake

**Course:** Digital Image Processing (DIP)  
**Project:** 004 — Insurance Document Intelligence  
**Implementation:** Python, OpenCV, Tesseract OCR, TF-IDF, LinearSVC, local web demo

## Quick start (Ubuntu)

### 1. Install system OCR

```bash
sudo apt update
sudo apt install -y tesseract-ocr
```

### 2. Start the web application

From the extracted project folder:

```bash
chmod +x start_demo.sh
./start_demo.sh
```

Open **http://127.0.0.1:8000** in your browser. The project now includes a responsive dashboard UI with drag-and-drop upload, file selection, processing pipeline overview, extracted-field cards, human-review status and OCR text viewer.

Upload a PNG/JPG/TIFF document image. The browser sends it to the local `/api/analyze` endpoint and displays the JSON result.

The first launch creates a Python virtual environment and installs the dependencies. Internet is needed for the initial pip installation.

### 3. Run the tests

In another terminal:

```bash
source .venv/bin/activate
cd backend
pytest -q
```

### 4. Train the ML baseline (optional; pre-trained demo model and metrics are included)

From the project root:

```bash
source .venv/bin/activate
python backend/scripts/train_phase8.py
```

Classify a new document image using the trained model:

```bash
python backend/scripts/predict_phase8.py --image dataset/synthetic/images/motor_claim_form_11.png
```

## What the system demonstrates

1. **Image enhancement:** grayscale conversion, histogram equalization and CLAHE.
2. **Denoising and thresholding:** Gaussian, median and bilateral filtering; global, Otsu and adaptive thresholding.
3. **Morphology:** erosion, dilation, opening, closing and morphological gradient with multiple structuring elements.
4. **Segmentation:** connected components, text regions, lines, words and character candidates.
5. **OCR:** Tesseract text recognition, word confidence and bounding boxes.
6. **Document classification:** keyword baseline plus TF-IDF + LinearSVC ML baseline.
7. **Information extraction:** evidence-backed extraction of common policy/claim fields.
8. **Validation:** basic format checks and human-review flags.
9. **Evaluation:** OCR CER/WER utilities, classification metrics and extraction precision/recall/F1 utilities.

## Included dataset

`dataset/synthetic/` contains 60 fictional document images (12 per class), transcripts, a CSV manifest, JSONL annotations and fixed train/validation/test split metadata. Classes:
- motor_claim_form
- policy_document
- repair_invoice
- vehicle_registration
- medical_bill

All sample data is synthetic and explicitly marked as an academic demonstration. It is not real customer/insurance data.

## Useful commands

Run the full OCR + extraction + validation pipeline:

```bash
python backend/scripts/run_phase6.py \
  --input dataset/original/sample_synthetic_claim_form.png \
  --output results/intelligence/sample_analysis.json
```

Run Phase 2–5 experiments using the scripts in `backend/scripts/`. See the phase sections later in this README for details.

## Important limitations (state these in your presentation)

- The supplied corpus is small and synthetic. The ML test split has only 10 images (2 per class); a high score on this split is not evidence of real-world performance.
- The generated transcripts are clean ground truth for the synthetic examples. Real OCR output can be worse.
- Rule-based extraction supports the demonstrated field patterns and may fail on new layouts, handwriting, or OCR errors.
- Format checks do not verify Aadhaar/PAN, vehicle ownership, insurer records, policy validity, fraud, or claim eligibility.
- Human review is required before any real-world decision. This is an academic prototype, not a production insurance system.

## Project structure

- `backend/app/dip/` — DIP algorithms
- `backend/app/ocr/` — Tesseract OCR and CER/WER
- `backend/app/intelligence/` — classification and field extraction
- `backend/app/evaluation/` — validation and metrics
- `backend/scripts/` — reproducible phase runners and ML scripts
- `dataset/original/` — synthetic starter sample used by earlier phases
- `dataset/synthetic/` — 60-document dataset
- `results/` — saved experiment outputs, OCR results, trained model and metrics
- `demo_server.py` — local browser-based upload and analysis demo


## Browser UI

The frontend is a responsive, self-contained HTML/CSS/JavaScript application in `frontend/index.html`. The local Python server serves this UI and exposes:
- `GET /` — dashboard
- `GET /health` — service health
- `POST /api/analyze` — multipart image upload and OCR/intelligence JSON response

No Node/npm build step is needed for the demo UI. The Python server serves the frontend directly.
