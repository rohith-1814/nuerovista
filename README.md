# NeuroVista — Alzheimer's MRI research prototype

A local Streamlit application for four-class brain MRI image classification research. It supports ResNet50 and EfficientNet-B0, prediction probabilities, Grad-CAM overlays, SHAP superpixel attribution, basic image checks, held-out evaluation metrics, local study history, and PDF reports.

> **Research use only.** This project is not a medical device and does not diagnose Alzheimer's disease. Do not use its outputs for clinical decisions. Predictions and visual explanations require independent validation and clinician review.

## Requirements

- Python 3.10 or later
- A licensed, appropriately consented brain MRI image dataset
- Optional NVIDIA GPU for faster training; CPU is supported
- Internet access to download the default Hugging Face dataset and pretrained torchvision initialization

## Install and run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Default: Hugging Face Alzheimer MRI dataset

Training defaults to [`Falah/Alzheimer_MRI`](https://huggingface.co/datasets/Falah/Alzheimer_MRI), a four-class brain MRI image dataset. Its published card lists 5,120 training images and 1,280 test images, with labels `Mild_Demented`, `Moderate_Demented`, `Non_Demented`, and `Very_Mild_Demented`. The trainer maps these to this app's class order. The first run downloads and caches the dataset.

```bash
python train.py --models resnet50 efficientnet_b0 --epochs 10
```

The training split is stratified into 90% training and 10% validation data; the dataset's provided test split is used for reported metrics. **The dataset does not document patient identifiers or guarantee patient-disjoint splits.** Therefore, these metrics may be affected by image-level leakage and must not be interpreted as independent clinical validation.

The dataset card declares Apache-2.0; check its current terms, provenance, image quality, and suitability for your use before downloading or redistributing it. The dataset author requests this citation:

```text
@dataset{alzheimer_mri_dataset,
  author = {Falah.G.Salieh},
  title = {Alzheimer MRI Dataset},
  year = {2023},
  publisher = {Hugging Face},
  version = {1.0},
  url = {https://huggingface.co/datasets/Falah/Alzheimer_MRI}
}
```

### Alternative: local ImageFolder data

For your own dataset, place images in `train`, `val`, and (recommended) `test` directories. Each split must contain exactly the following folders:

```text
data/
  train/
    non_demented/
    very_mild/
    mild/
    moderate/
  val/
    non_demented/
    very_mild/
    mild/
    moderate/
  test/                         # separate, held-out patients
    non_demented/
    very_mild/
    mild/
    moderate/
```

Supported training images are those readable by torchvision/Pillow. The app itself accepts PNG, JPEG, BMP, and TIFF 2D images. It does not process DICOM series or 3D MRI volumes. Check that the source dataset labels genuinely correspond to these categories before using them.

Train using the local folders instead of the default Hugging Face source:

```bash
python train.py --source local --data data --models resnet50 efficientnet_b0 --epochs 10 --batch-size 16
```

Training saves the best validation-accuracy checkpoint to `models/<model>.pth`. When a test split exists, it evaluates that checkpoint and writes `models/<model>_metrics.json`. No held-out metrics are displayed unless a test split was evaluated. Adjust epochs, batch size, and learning rate for your compute and dataset. Checkpoints and source data are intentionally ignored by Git.

Start the dashboard:

```bash
streamlit run app.py
```

For a trained checkpoint in another location, select the matching classifier in the sidebar and upload its `.pth` file. Checkpoints must come from this project's training script and include all four exact class labels.

## Features

- **Multi-model classification:** ResNet50 and EfficientNet-B0, fine-tuned from torchvision ImageNet weights.
- **Image quality checks:** image readability, minimum dimensions, aspect ratio, contrast, and intensity variation. These do not verify MRI modality or clinical quality.
- **Explainability:** Grad-CAM for the predicted class and Kernel SHAP over a 4×4 image superpixel grid. Both are model-behavior visualizations, not lesion segmentation or causal evidence.
- **Uncertainty caution:** warnings for top-class scores below 60%; the threshold is a simple caution and does not calibrate confidence.
- **Evaluation:** accuracy, macro precision/recall/F1 and confusion matrix from a separate test split.
- **History and reporting:** local CSV prediction history, CSV export, and a downloadable PDF report.

## Responsible use and privacy

The app runs locally by default. Prediction history is written to `patient_history.csv` as plain text, without encryption or access controls. Do not enter identifiable health data; use synthetic or de-identified data and protect the machine and files according to applicable policies. Uploaded scans are not written to disk by the app. Follow dataset licenses, consent requirements, institutional review, and local regulations.

Patient-level splits are essential: images from one patient must not appear across training, validation, and test sets. The default Hugging Face dataset provides no patient IDs to verify this. The included augmentation and evaluation do not establish clinical validity, robustness, calibration, fairness, or generalizability. Validate independently on representative populations and acquisition protocols before any research claims.

GitHub repository: [rohith-1814/nuerovista](https://github.com/rohith-1814/nuerovista).
