# AIML-02: AI-Assisted Medical Imaging and Diagnosis System
Internship capstone project, reviewed by mentors. Education only, not for clinical use.

## Stack
- Python, PyTorch, MONAI, FastAPI, pydicom, Orthanc, DVC, React, Docker
- Classification: RSNA chest X-rays (PNG), splits in data/splits/*.csv, one row per patient
- Segmentation: MSD Task09_Spleen, MONAI 3D U-Net

## Rules
- Keep code simple and readable: small functions, short comments, no extra abstractions
- Paths and hyperparameters go in configs/config.yaml, not in code
- Never split by image, only by patientId
- Report AUC, sensitivity and specificity (classification), Dice (segmentation)
- Only report real results from my runs, never invented numbers
- Add a short pytest test for the important logic (splits, shapes, API endpoints)
- Never commit raw data or model checkpoints
- Log every Approve/Reject decision to an append-only audit log
- Show "For education only, not for clinical use" in the UI, README and model card