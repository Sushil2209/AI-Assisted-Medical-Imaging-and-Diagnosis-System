# AI-Assisted Medical Imaging and Diagnosis System

> **WARNING: For education only, not for clinical use.**
> This is an internship capstone project. It must not be used to diagnose, treat or make decisions about real patients.

## 1. Overview and Objectives

This project explores how deep learning can support medical imaging workflows. It has two parts:

- **Classification:** an EfficientNet-B0 model predicts pneumonia from chest X-rays (RSNA Pneumonia dataset).
- **Segmentation:** a MONAI 3D U-Net segments the spleen in CT volumes (Medical Segmentation Decathlon, Task09).

A live pipeline connects an **Orthanc** DICOM server, a **FastAPI** inference service with **Grad-CAM** explanations, and a **React + TypeScript** dashboard.

Objectives:

1. Train and evaluate a chest X-ray classifier with patient-level data splits.
2. Train and evaluate a 3D spleen segmentation model.
3. Explain classifier predictions with Grad-CAM heatmaps.
4. Serve predictions for DICOM images stored in Orthanc through a REST API.
5. Show studies, predictions and heatmaps in a web dashboard.

## 2. Features and Implemented Functionality

| Area | Status |
|------|--------|
| Patient-level RSNA splits (70/15/15, stratified, fixed seed) | Implemented (`scripts/make_rsna_splits.py`) |
| EfficientNet-B0 training and evaluation | Implemented (`scripts/train_classifier.py`) |
| Grad-CAM heatmaps | Implemented (`src/explain/gradcam.py`) |
| DICOM to image conversion (MONOCHROME1/2, rescale slope/intercept, 8/16-bit) | Implemented in the API |
| Orthanc integration (list studies, fetch DICOM instances) | Implemented |
| FastAPI inference service (`/health`, `/orthanc/studies`, `/predict/{instance_id}`) | Implemented |
| React + TypeScript dashboard | Implemented (health check, study list, preview, prediction, Grad-CAM) |
| MONAI 3D U-Net training and evaluation | Implemented as a **training script** (`scripts/train_spleen.py`) |
| Pytest test suite | Implemented (21 tests) |
| DVC tracking of the MSD dataset | Metadata committed (no remote configured) |

Not implemented (verified by inspecting the source code):

- The segmentation model is **not** part of the inference API. The API only serves the classifier.
- There is **no** approve/reject workflow and **no** audit log in the code.
- There is **no** Docker deployment (no Dockerfile or compose file).
- There is no model card file in the repository.

## 3. System Architecture and Data Flow

```
 RSNA PNG --> scripts/upload_test_dicom.py --> Orthanc (localhost:8042)
                                                  |
                      +---------------------------+
                      | REST (studies, instances, previews)
                      v
        React dashboard (Vite dev server)  --/api-->  FastAPI service (127.0.0.1:8000)
        (/orthanc-proxy goes to Orthanc)               1. download DICOM from Orthanc
                                                       2. convert to 8-bit RGB image
                                                       3. EfficientNet-B0 (CPU) -> probability
                                                       4. optional Grad-CAM overlay (PNG, base64)
```

Flow in the dashboard:

1. The dashboard calls `GET /api/health` to show backend status.
2. It lists studies with `GET /api/orthanc/studies`.
3. For a selected study it asks Orthanc (through the Vite proxy) for the first instance and its PNG preview.
4. It calls `POST /api/predict/{instance_id}?gradcam=true` and shows the class, probability and heatmap.

Project layout:

```
configs/config.yaml          Paths and hyperparameters
data/splits/                 RSNA split CSVs (train/val/test) and MSD split JSON
data/raw/msd.dvc             DVC metadata for the MSD dataset
docs/                        Metrics JSON files and Grad-CAM / Orthanc figures
frontend/                    React + TypeScript dashboard (Vite)
scripts/                     Training, split, demo and Orthanc helper scripts
services/inference_api/      FastAPI application (main.py)
src/data, src/explain        Dataset/transforms and Grad-CAM code
tests/                       Pytest tests
```

## 4. Datasets

### RSNA Pneumonia (classification)

- Chest X-rays exported as PNG, with a label file (`stage2_train_metadata.csv`) containing `patientId` and `Target` (1 = pneumonia, 0 = no pneumonia).
- Labels are aggregated per patient (`max` of `Target`), giving one row per patient.
- Splits in `data/splits/*.csv` (columns `patientId, Target, filename`) are created with a stratified 70/15/15 split **by patient, never by image**. The test set has 4,003 patients.
- `tests/test_splits.py` checks that no `patientId` appears in two splits.
- Raw images are not committed. Paths are set in `configs/config.yaml` (`data/raw/rsna/...`).

### MSD Task09 Spleen (segmentation)

- 3D abdominal CT volumes with spleen labels (Medical Segmentation Decathlon).
- `scripts/train_spleen.py` shuffles the cases with the configured seed and uses 31 train / 5 validation / 5 test cases (`data/splits/msd_split.json`).
- Preprocessing: RAS orientation, spacing 1.5 x 1.5 x 2.0 mm (from `pixdim` in the config), intensity window [-57, 164] scaled to [0, 1], foreground crop, random 96 x 96 x 96 patches for training.

## 5. Classification Model: EfficientNet-B0

- Architecture: torchvision EfficientNet-B0 with a single-logit head; sigmoid gives the pneumonia probability. Input size 224 x 224, ImageNet normalisation.
- Training config: 10 epochs, batch size 16, learning rate 1e-4, weight decay 0.01, threshold 0.5. Augmentation: small rotation and brightness/contrast jitter (no flips).
- Best epoch (by validation AUC): 3.

Test results on 4,003 patients (from [docs/classifier_metrics.json](docs/classifier_metrics.json)):

| Metric | Value |
|--------|-------|
| AUC | 0.887 |
| Sensitivity | 0.828 |
| Specificity | 0.784 |

Confusion matrix (rows = true class, columns = predicted): `[[2430, 671], [155, 747]]`.

## 6. Segmentation Model: MONAI 3D U-Net

- MONAI `UNet` (3D), channels `[16, 32, 64, 128, 256]`, strides `[2, 2, 2, 2]`, 2 residual units, `DiceCELoss`, AdamW, 50 epochs, sliding-window inference.
- **Test mean Dice: 0.7055** (result from the author's training run).
- **Final training result vs committed file:** the final result is the test mean Dice of 0.7055 reported above. The `docs/spleen_metrics.json` file committed in the repository (`test_mean_dice` about 0.004, `best_val_dice` about 0.009, `n_test` = 2) comes from a short debug run (2 test cases) and must **not** be read as the final result. The final metrics file and checkpoint are written to `models_ckpt/`, which is git-ignored, so they are not in the repository.
- The model is trained and evaluated by script only. It is **not** served by the API or shown in the dashboard.

## 7. Grad-CAM Explainability

`src/explain/gradcam.py` uses the `pytorch-grad-cam` library on the last feature block of EfficientNet-B0 (`model.features[-1]`) and overlays the heatmap on the input image. The API returns the overlay as a base64 PNG when `gradcam=true`.

Examples in `docs/`:

- [gradcam_examples.png](docs/gradcam_examples.png): 3 pneumonia and 3 normal test images.
- [gradcam_orthanc_comparison.png](docs/gradcam_orthanc_comparison.png): original vs Grad-CAM for an image served from Orthanc.

Grad-CAM shows where the model looked. It is not proof that the model is correct.

## 8. Orthanc DICOM Server Integration

- Orthanc is expected at `http://localhost:8042` (setting `orthanc.url` in `configs/config.yaml`). Install and start Orthanc yourself; the repository does not include a Docker setup.
- `scripts/upload_test_dicom.py` converts one RSNA PNG to a DICOM Secondary Capture and uploads it to Orthanc.
- The API reads studies (`/studies?expand`) and DICOM files (`/instances/{id}/file`) from Orthanc. Errors map to HTTP 404 (not found) or 502 (Orthanc unreachable/error).
- Only single-frame, single-channel grayscale DICOM images are supported.

## 9. FastAPI Inference Endpoints

Source: [services/inference_api/main.py](services/inference_api/main.py). Interactive docs are at `http://127.0.0.1:8000/docs`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status |
| GET | `/orthanc/studies` | List studies from Orthanc |
| POST | `/predict/{instance_id}?gradcam=false` | Classify an Orthanc instance; `gradcam=true` adds a heatmap |

**Health**

```powershell
curl http://127.0.0.1:8000/health
```

Response fields: `status`, `model_loaded`, `device`, `disclaimer`.

**List studies**

```powershell
curl http://127.0.0.1:8000/orthanc/studies
```

Response fields: `studies` (each with `id`, `patient_id`, `study_date`, `description`) and `disclaimer`.

**Predict**

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict/<ORTHANC_INSTANCE_ID>"
curl.exe -X POST "http://127.0.0.1:8000/predict/<ORTHANC_INSTANCE_ID>?gradcam=true"
```

Response fields: `instance_id`, `predicted_class` (`"No pneumonia"` or `"Pneumonia"`), `pneumonia_probability`, `threshold`, `disclaimer`, and `gradcam_png_base64` (only when `gradcam=true`).

Error codes: 404 (not in Orthanc), 422 (unsupported or undecodable DICOM), 500 (Grad-CAM failure), 502 (Orthanc problem).

## 10. React + TypeScript Dashboard

Located in `frontend/` (React 19, TypeScript, Vite). The dev server proxies `/api` to `http://127.0.0.1:8000` and `/orthanc-proxy` to `http://localhost:8042`, so the backend and Orthanc must be running. The footer shows an education-only notice.

```powershell
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite (usually `http://localhost:5173`). Other scripts: `npm run build`, `npm run lint`.

## 11. Python Environment Setup (Windows)

Requirements: Python 3.12 (the project was developed with 3.12), Node.js, Git.

```powershell
git clone https://github.com/Sushil2209/AI-Assisted-Medical-Imaging-and-Diagnosis-System.git
cd "AI-Assisted-Medical-Imaging-and-Diagnosis-System"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\check_setup.py
```

`requirements.txt` is a full pinned environment export. Datasets and model checkpoints are not in the repository.

**Common commands**

```powershell
# Create patient-level RSNA splits (needs the RSNA metadata file)
python scripts\make_rsna_splits.py

# Train the classifier (--debug runs a tiny smoke test)
python scripts\train_classifier.py

# Train the spleen model
python scripts\train_spleen.py

# Grad-CAM example grid
python scripts\demo_gradcam.py

# Upload a test DICOM to Orthanc
python scripts\upload_test_dicom.py
```

Training writes checkpoints to `models_ckpt/` (git-ignored). The API expects `models_ckpt/efficientnet_b0.pth`.

**Run the backend** (from the project root, with the venv active, Orthanc running):

```powershell
uvicorn services.inference_api.main:app --host 127.0.0.1 --port 8000
```

Then run the frontend as described in section 10.

## 12. Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Actual result from the latest run: **21 passed, 1 warning** (about 38 s).

| File | Covers |
|------|--------|
| `tests/test_splits.py` | No `patientId` shared between train/val/test |
| `tests/test_dicom_to_image.py` | DICOM decoding: photometric inversion, rescale, bit depth, invalid input |
| `tests/test_predict_gradcam.py` | `/predict` with and without Grad-CAM, Grad-CAM failure |

Run pytest inside the project virtual environment; the global Python may lack packages such as `pydicom`.

## 13. DVC Dataset Tracking

- `data/raw/msd.dvc` records the MSD Spleen dataset (103 files, about 6.5 GB). The `.dvc` metadata is committed to Git; the data itself is git-ignored.
- **No DVC remote is configured** (`.dvc/config` is empty), so `dvc pull` and `dvc push` will not work until a remote is added.

```powershell
dvc status
dvc remote add -d <name> <path-or-url>   # one-time setup
dvc push
dvc pull
```

## 14. Limitations and Future Improvements

Limitations:

- Educational prototype; not validated clinically or on external data.
- The classifier uses a single dataset and a fixed 0.5 threshold; no calibration or fairness analysis.
- The API runs on CPU and classifies only the first instance of a study in the dashboard.
- The segmentation model is not integrated into the API or dashboard.
- No authentication, approve/reject workflow, or audit log.
- No Docker deployment and no DVC remote.
- The committed `docs/spleen_metrics.json` is from a debug run, not the final run.

Future improvements:

- Add the spleen model to the API with NIfTI/DICOM series input and a 3D viewer.
- Add a clinician review step (approve/reject) with an append-only audit log.
- Add Docker packaging, a DVC remote, and CI for tests.
- Evaluate on external data; add calibration and uncertainty estimates.
- Add a model card.

## 15. Medical Disclaimer

**For education only, not for clinical use.** The models, outputs and heatmaps in this project are for learning purposes. They are not medical devices, have not been clinically validated, and must not replace the judgement of qualified healthcare professionals.
