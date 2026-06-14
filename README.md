# Non-Invasive Neonatal Jaundice Detection

Smartphone-based screening system for neonatal jaundice using baby skin and eye sclera photos. The project targets rural Indian hospitals and clinics where transcutaneous bilirubinometers or rapid laboratory access may not be available.

Clinical disclaimer: this system is for screening and research only. It must not replace serum bilirubin testing, pediatric assessment, or local neonatal jaundice treatment protocols.

## Problem Statement

Neonatal jaundice is common in the first week of life and can become dangerous when bilirubin rises unchecked. Several clinical sources report jaundice in roughly 60% of term newborns and 80% of preterm newborns. An Indian study on maternal detection reported any jaundice in 79% of neonates and significant jaundice in 13.4%. Rural facilities may lack bilirubinometers, trained staff, or reliable referral workflows, so a phone-based screening tool can help flag babies who need urgent confirmation and treatment.

Sources:

- Indian neonatal jaundice detection study: https://pmc.ncbi.nlm.nih.gov/articles/PMC6985939/
- Global severe neonatal jaundice meta-analysis: https://pubmed.ncbi.nlm.nih.gov/37297932/
- Eastern India hospital profile noting common term/preterm burden: https://imsear.searo.who.int/handle/123456789/242137

## Features

- Dual-input model using skin and sclera images.
- EfficientNet-B0 backbone for lightweight deployment.
- Multi-task output: severity class and estimated bilirubin in mg/dL.
- OpenCV preprocessing with YCbCr color features.
- Skin-tone normalization for Indian neonatal skin-tone diversity.
- Synthetic jaundice generation in YCbCr/YCrCb space.
- Flask mobile-first web app with SQLite patient records.
- Offline operation with local records and sync queue.
- REST API for Android or hospital server integration.
- ONNX export and INT8 quantization for mobile CPU inference.
- Docker deployment for hospital LAN servers.

## Project Structure

```text
data/
  normal/
  jaundiced/
    mild/
    moderate/
    severe/
models/
synthetic/
app/
reports/
static/generated/
templates/
```

Key files:

- `dataset.py`: image loading, augmentation, YCbCr features, train/val/test split.
- `skin_tone_normalizer.py`: skin-tone and illumination normalization.
- `model.py`: dual-branch EfficientNet-B0 model.
- `train.py`: multi-task training.
- `evaluate.py`: confusion matrix and medical metrics.
- `synthetic_jaundice_generator.py`: synthetic augmentation and ablation tooling.
- `research_results.py`: paper tables and figures.
- `region_detector.py`: offline skin/sclera detection.
- `mobile_optimize.py`: ONNX export, INT8 quantization, CPU benchmark.
- `app.py`: Flask web app and REST API.

## Installation

Use Python 3.10.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Setup

Place images in:

```text
data/normal/
data/jaundiced/mild/
data/jaundiced/moderate/
data/jaundiced/severe/
```

Recommended filename patterns for paired samples:

```text
baby001_skin.jpg
baby001_sclera.jpg
baby002-skin.jpg
baby002-sclera.jpg
```

Unpaired images are still accepted, but paired skin/sclera images are preferred.

## Training

```powershell
python train.py
```

Training uses:

- Classification loss: CrossEntropy
- Regression loss: MSE
- Combined loss: `0.7 * classification + 0.3 * regression`
- Optimizer: Adam, learning rate `1e-4`
- Best checkpoint: `models/best_model.pth`

Evaluate:

```powershell
python evaluate.py
```

Outputs:

- `reports/training_curves.png`
- `reports/confusion_matrix.png`
- `reports/auc_roc_curve.png`
- `reports/training_report.json`
- `reports/evaluation_report.json`

## Synthetic Data and Research Ablation

Generate synthetic jaundice images from normal baby skin images:

```powershell
python synthetic_jaundice_generator.py generate --variants 2
```

Create ablation manifests:

```powershell
python synthetic_jaundice_generator.py prepare-ablation --synthetic-ratio 1.0
```

Compare prediction CSVs:

```powershell
python synthetic_jaundice_generator.py ablation --baseline-predictions baseline.csv --synthetic-predictions synthetic.csv
```

Generate paper tables and figures:

```powershell
python research_results.py
```

## Results Table

Current placeholder table for manuscript layout. Replace with real validation results before publication.

| Method | Overall accuracy | Dark skin accuracy | Rare severe accuracy |
| --- | ---: | ---: | ---: |
| Without synthetic data | 0.812 | 0.716 | 0.641 |
| With synthetic data | 0.884 | 0.832 | 0.781 |

## Mobile Optimization

Export ONNX, quantize INT8, and benchmark CPU inference:

```powershell
python mobile_optimize.py --checkpoint models/best_model.pth
```

Outputs:

- `models/jaundice_model.onnx`
- `models/jaundice_model_int8.onnx`
- `models/mobile_optimization_report.json`

Targets:

- Quantized model under 20 MB.
- CPU p95 latency under 3 seconds.

Always verify on the target Android phone because desktop CPU latency is not a guarantee of mobile performance.

## Flask App

Run locally:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000/
```

The app supports:

- Skin photo upload or phone camera capture.
- Eye/sclera photo upload or phone camera capture.
- Baby name, age in hours, birth weight, gestational age.
- Processing preview for skin/sclera detection.
- Result page with severity, bilirubin estimate, risk meter, threshold comparison, advice, GradCAM-style overlay.
- Patient records and PDF export.

## REST API

Endpoint:

```text
POST /api/predict
```

Request JSON:

```json
{
  "skin_image_base64": "...",
  "eye_image_base64": "...",
  "age_hours": 36
}
```

Response JSON:

```json
{
  "severity": "Mild",
  "bilirubin_estimate": 8.5,
  "urgency": "low",
  "advice": "Increase feeding frequency",
  "threshold_status": "Below age-specific action threshold",
  "probabilities": [0.1, 0.7, 0.15, 0.05],
  "prediction_source": "model",
  "gradcam_base64": "...",
  "disclaimer": "For screening only, confirm with blood test"
}
```

If the skin or sclera region is unclear, the API returns HTTP `422` with retake guidance.

## Offline Mode

The app works fully offline:

- Model inference runs locally.
- OpenCV region detection runs locally.
- SQLite records are stored in `app/patients.db`.
- Uploads and generated visuals stay on disk.
- New records are added to `sync_queue`.

When Wi-Fi/server access is available, configure:

```powershell
$env:SYNC_SERVER_URL="https://hospital-server.example/api/sync"
```

Then trigger:

```text
POST /offline/sync
```

If no sync URL is configured, records remain local.

## Docker Deployment

Build and run:

```powershell
docker compose up --build
```

Open:

```text
http://localhost:5000/
```

Persistent folders are mounted through `docker-compose.yml`:

- `app/`
- `models/`
- `reports/`
- `static/generated/`
- `data/`

## Clinical Validation Requirements

Before real deployment:

- Validate against serum bilirubin measurements.
- Stratify by age in hours, gestational age, birth weight, sex, lighting, phone model, and Fitzpatrick skin tone.
- Report sensitivity and specificity for clinically significant jaundice.
- Evaluate false negatives separately for severe cases.
- Obtain ethics approval and data privacy review.
- Follow local pediatric jaundice treatment nomograms and referral protocols.

## Future Work

- Replace heuristic GradCAM-style overlay with true GradCAM hooks for the trained model.
- Train a dedicated segmentation model for newborn face, forehead, chest, and sclera.
- Add Android-native ONNX Runtime inference.
- Add calibrated age-hour nomograms from local clinical guidelines.
- Add secure encrypted sync to district hospital servers.
- Add multilingual UI for rural staff.
- Conduct prospective clinical validation across multiple Indian states.
