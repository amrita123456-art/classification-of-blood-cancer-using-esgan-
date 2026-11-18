import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Database Settings
DATABASE_PATH = BASE_DIR / "database" / "predictions.db"

# Storage Settings
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
PROCESSED_FOLDER = BASE_DIR / "static" / "uploads" / "processed"
REPORTS_FOLDER = BASE_DIR / "reports"

# Ensure directories exist
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, REPORTS_FOLDER, BASE_DIR / "database", BASE_DIR / "models" / "esrgan", BASE_DIR / "models" / "mobilenet"]:
    folder.mkdir(parents=True, exist_ok=True)

# Image Dimensions
LR_SIZE = (128, 128)
HR_SIZE = (512, 512)
MOBILENET_INPUT_SHAPE = (224, 224, 3)

# Allowed File Extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload

# Classes Definition
CLASSES = ["ALL", "AML", "CLL", "CML", "NORMAL"]
CLASS_DETAILS = {
    "ALL": {
        "name": "Acute Lymphoblastic Leukemia",
        "description": "Rapidly progressing cancer of the lymphoid line of blood cells, characterized by large numbers of immature lymphocytes."
    },
    "AML": {
        "name": "Acute Myeloid Leukemia",
        "description": "Cancer of the myeloid line of blood cells, characterized by rapid growth of abnormal white blood cells that build up in bone marrow."
    },
    "CLL": {
        "name": "Chronic Lymphocytic Leukemia",
        "description": "Slower growing leukemia affecting lymphoid cells, leading to an overproduction of mature-appearing yet abnormal lymphocytes."
    },
    "CML": {
        "name": "Chronic Myeloid Leukemia",
        "description": "Slowly progressing leukemia originating in bone marrow myeloid cells, frequently linked to the Philadelphia chromosome."
    },
    "NORMAL": {
        "name": "Normal Blood Smear",
        "description": "Normal cellular morphology without malignant blast populations or severe cytological abnormalities."
    }
}

# Model Paths
ESRGAN_MODEL_PATH = BASE_DIR / "models" / "esrgan" / "generator.pth"
MOBILENET_MODEL_PATH = BASE_DIR / "models" / "mobilenet" / "blood_cancer_mobilenetv2.h5"
METRICS_JSON_PATH = BASE_DIR / "reports" / "metrics.json"

# Dataset Directories
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

# Flask Secret Key
SECRET_KEY = os.environ.get("SECRET_KEY", "bloodai-super-secret-clinical-key-2026")
