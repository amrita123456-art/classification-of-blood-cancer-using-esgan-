# BloodAI — ESRGAN-Enhanced Blood Smear Image Analysis & Leukemia Subtype Classification Using MobileNetV2

An end-to-end Python, Deep Learning, and Flask application for automated leukemia subtype classification from microscopic blood smears. The system addresses low-resolution optical degradation using **ESRGAN (Residual-in-Residual Dense Blocks)** super-resolution (128×128 → 512×512) prior to transfer learning classification with **MobileNetV2**.

---

## 🎯 Target Classes (5 Subtypes)

| Class Code | Meaning | Clinical Overview |
|:---|:---|:---|
| `ALL` | Acute Lymphoblastic Leukemia | Rapid proliferation of immature lymphoid blasts. |
| `AML` | Acute Myeloid Leukemia | Excessive accumulation of abnormal myeloid blasts in bone marrow. |
| `CLL` | Chronic Lymphocytic Leukemia | Indolent accumulation of mature-appearing yet dysfunctional lymphocytes. |
| `CML` | Chronic Myeloid Leukemia | Clonal myeloproliferative disorder linked to Philadelphia chromosome. |
| `NORMAL` | Normal Blood Smear | Normal leukocyte morphology without malignant blast populations. |

---

## 🔬 System Pipeline Architecture

```text
                  USER
                   │
                   ▼
         ┌───────────────────┐
         │ Upload Blood Smear│
         │       Slide       │
         └─────────┬─────────┘
                   │
                   ▼
         ┌───────────────────┐
         │ Image Validation  │
         │  JPG / PNG / JPEG │
         └─────────┬─────────┘
                   │
                   ▼
         ┌───────────────────┐
         │  Preprocessing    │
         │  Resize 128×128   │
         │  Normalization    │
         └─────────┬─────────┘
                   │
                   ▼
         ┌───────────────────┐
         │      ESRGAN       │
         │   RRDB Generator  │
         │ 128×128 → 512×512 │
         └─────────┬─────────┘
                   │
                   ▼
         ┌───────────────────┐
         │    MobileNetV2    │
         │ Feature Extraction│
         │ Transfer Learning │
         └─────────┬─────────┘
                   │
                   ▼
         ┌───────────────────┐
         │ 5-Class Prediction│
         │  Softmax Output   │
         └─────────┬─────────┘
                   │
                   ▼
     ┌───────────────────────────────┐
     │        Flask Dashboard        │
     │                               │
     │ Predicted Subtype & Confidence│
     │ Probability Distribution Bar  │
     │ Original vs Enhanced (Side/By)│
     │ Grad-CAM Attention Heatmap    │
     └───────────────────────────────┘
```

---

## 🎨 Clinical AI Design Theme

The web interface is styled using a tailored clinical and technology-focused palette:
- **Primary Navy:** `#17324D`
- **Clinical Teal:** `#2E7D7A`
- **Research Blue:** `#4F7CAC`
- **Main Background:** `#F6F8FB`
- **Card Background:** `#FFFFFF`
- **Dark Slate Text:** `#263746`
- **Soft Gray Border:** `#DCE3EA`

---

## 🚀 Installation & Quick Start

### 1. Environment Setup
Create and activate a Python 3.10+ virtual environment:
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Initialize Scaffolding Folders
```bash
python preprocessing/dataset_split.py
```

### 4. Run the Flask Web Application
```bash
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 🧠 Training & Evaluation Workflow

### Phase 1 — Dataset Preparation
Organize your blood smear images under `dataset/` following patient/slide-level separation:
- `dataset/train/{ALL,AML,CLL,CML,NORMAL}`
- `dataset/validation/{ALL,AML,CLL,CML,NORMAL}`
- `dataset/test/{ALL,AML,CLL,CML,NORMAL}`

### Phase 2 — Train / Fine-tune ESRGAN
Train the RRDBNet Generator on paired LR/HR blood smear images:
```bash
python esrgan/train.py --epochs 15 --batch_size 4
```

### Phase 3 — Train MobileNetV2 Classifier
Run two-stage transfer learning (Frozen Base → Fine-Tuning Top Conv Layers):
```bash
python classification/train.py --epochs1 20 --epochs2 15 --batch_size 16
```

### Phase 4 — Comparative Evaluation (Baseline vs Proposed)
Evaluate both Model A (Baseline MobileNetV2 on LR) and Model B (Proposed ESRGAN + MobileNetV2):
```bash
python classification/evaluate.py
```
This generates:
- `reports/metrics.json`
- `reports/confusion_matrix.png`
- `reports/model_comparison.png`

---

## 📁 Repository Structure

```text
blood-smear-specific Python application/
│
├── app.py                          # Flask application entry point
├── config.py                       # Global paths, constants, and hyperparameters
├── requirements.txt                # Pinned dependencies
├── README.md                       # Documentation
│
├── database/
│   └── db.py                       # SQLite persistence for scans
│
├── preprocessing/
│   ├── preprocess.py               # Downsampling, normalization, PSNR
│   ├── augmentation.py             # Keras conservative data augmentation
│   └── dataset_split.py            # Dataset verification & audit
│
├── esrgan/
│   ├── rrdb.py                     # Residual-in-Residual Dense Block in PyTorch
│   ├── generator.py                # 4x RRDBNet super-resolution generator
│   ├── discriminator.py            # VGG relativistic discriminator
│   ├── losses.py                   # Perceptual VGG, L1, and RaGAN losses
│   ├── train.py                    # Training & fine-tuning script
│   └── inference.py                # Super-resolution inference engine
│
├── classification/
│   ├── train.py                    # Two-stage MobileNetV2 training script
│   ├── predict.py                  # 5-class prediction engine
│   ├── gradcam.py                  # Explainable AI Grad-CAM generation
│   └── evaluate.py                 # Baseline vs Proposed comparative benchmark
│
├── evaluation/
│   ├── metrics.py                  # Accuracy, Precision, Recall, F1, AUC
│   └── plots.py                    # Confusion matrix & comparative bar charts
│
├── routes/
│   ├── main.py                     # Dashboard & About routes
│   ├── analysis.py                 # Image upload & inference routes
│   ├── history.py                  # Patient scan history routes
│   └── metrics.py                  # Model benchmarks route
│
├── templates/
│   ├── base.html                   # Master layout with sidebar
│   ├── index.html                  # Dashboard & pipeline overview
│   ├── upload.html                 # Drag & drop upload interface
│   ├── result.html                 # Side-by-side images, probabilities & Grad-CAM
│   ├── history.html                # Past scan history & deletion
│   ├── metrics.html                # Model comparison & confusion matrix
│   └── about.html                  # Mathematical formulations & cytology
│
├── static/
│   ├── css/main.css                # Clinical AI Navy Teal styling
│   └── js/
│       ├── upload.js               # Interactive upload & animated pipeline
│       └── result.js               # Probability bar animations & export
│
└── dataset/
    └── README.md                   # Dataset guidelines & references
```

---

## ⚖️ Research Protocol & Disclaimer
BloodAI is engineered for research and academic validation purposes. Predictions generated by neural network architectures must be corroborated by standard hematological assays, peripheral blood flow cytometry, cytogenetics, and examination by a qualified hematopathologist.
