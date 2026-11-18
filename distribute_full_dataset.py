import random
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TRAIN_DIR, VAL_DIR, TEST_DIR, CLASSES

def distribute_files(file_list, target_class, split_ratios=(0.70, 0.15, 0.15)):
    random.seed(42)
    random.shuffle(file_list)

    n_total = len(file_list)
    train_r, val_r, _ = split_ratios
    n_train = max(1, int(n_total * train_r))
    n_val = max(1, int(n_total * val_r))

    splits = {
        TRAIN_DIR / target_class: file_list[:n_train],
        VAL_DIR / target_class: file_list[n_train:n_train + n_val],
        TEST_DIR / target_class: file_list[n_train + n_val:]
    }

    for split_dir, split_files in splits.items():
        split_dir.mkdir(parents=True, exist_ok=True)
        for sf in split_files:
            dst = split_dir / sf.name
            if dst.exists():
                dst = split_dir / f"{sf.stem}_{random.randint(1000, 9999)}{sf.suffix}"
            shutil.copy2(str(sf), str(dst))
        print(f"  {split_dir.parent.name}/{split_dir.name} -> {len(list(split_dir.glob('*')))} images")

def main():
    print("=" * 65)
    print("DISTRIBUTING AML AND CML IMAGES INTO TRAIN, VAL, TEST SPLITS")
    print("=" * 65)

    # 1. Distribute AML (Blasts and Promyelocytes from APL dataset)
    aml_root = Path("dataset/raw_download/aml_download/All/All")
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    aml_files = [
        f for f in aml_root.rglob("*")
        if f.is_file() and f.suffix.lower() in valid_exts and ("blast" in f.parent.name.lower() or "promyelocyte" in f.parent.name.lower())
    ]
    print(f"\nDistributing {len(aml_files)} AML images...")
    distribute_files(aml_files, "AML")

    # 2. Distribute CML
    cml_root = Path("dataset/raw_download/cml_download/CML")
    cml_files = [
        f for f in cml_root.rglob("*")
        if f.is_file() and f.suffix.lower() in valid_exts
    ]
    print(f"\nDistributing {len(cml_files)} CML images...")
    distribute_files(cml_files, "CML")

    print("\n" + "=" * 65)
    from preprocessing.dataset_split import verify_dataset_structure
    verify_dataset_structure()
    print("=" * 65)

if __name__ == "__main__":
    main()
