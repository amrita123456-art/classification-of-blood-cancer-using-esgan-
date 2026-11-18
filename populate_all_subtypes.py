import random
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TRAIN_DIR, VAL_DIR, TEST_DIR, CLASSES

def populate_from_folder(src_folder, target_class, split_ratios=(0.70, 0.15, 0.15)):
    src_folder = Path(src_folder)
    if not src_folder.exists():
        print(f"Folder does not exist yet: {src_folder}")
        return False

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [f for f in src_folder.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts]
    
    # Exclude system files or non-image thumbnails
    files = [f for f in files if "thumb" not in f.name.lower()]

    print(f"\nProcessing {target_class}: found {len(files)} image files in {src_folder}")
    if not files:
        return False

    random.seed(42)
    random.shuffle(files)

    n_total = len(files)
    train_r, val_r, _ = split_ratios
    n_train = max(1, int(n_total * train_r))
    n_val = max(1, int(n_total * val_r))

    splits = {
        TRAIN_DIR / target_class: files[:n_train],
        VAL_DIR / target_class: files[n_train:n_train + n_val],
        TEST_DIR / target_class: files[n_train + n_val:]
    }

    for split_dir, split_files in splits.items():
        split_dir.mkdir(parents=True, exist_ok=True)
        for sf in split_files:
            dst = split_dir / sf.name
            if dst.exists():
                dst = split_dir / f"{sf.stem}_{random.randint(1000, 9999)}{sf.suffix}"
            shutil.copy2(str(sf), str(dst))
        print(f"  {split_dir.parent.name}/{split_dir.name} -> {len(list(split_dir.glob('*')))} images")

    return True

if __name__ == "__main__":
    # Check AML
    aml_dirs = [
        Path("dataset/raw_download/aml_download"),
        Path("dataset/raw_download/eugeneshenderov_acute-promyelocytic-leukemia-apl")
    ]
    for d in aml_dirs:
        if d.exists() and any(d.glob("**/*.*")):
            populate_from_folder(d, "AML")
            break

    # Check CML
    cml_dirs = [
        Path("dataset/raw_download/cml_download"),
        Path("dataset/raw_download/nuraishahafzainizam_chronic-myeloid-leukemia")
    ]
    for d in cml_dirs:
        if d.exists() and any(d.glob("**/*.*")):
            populate_from_folder(d, "CML")
            break

    from preprocessing.dataset_split import verify_dataset_structure
    verify_dataset_structure()
