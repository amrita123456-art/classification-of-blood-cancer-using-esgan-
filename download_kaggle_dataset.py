import os
import sys
import json
import shutil
import random
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CLASSES, DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR

# List of verified accessible Kaggle datasets containing multi-class leukemia blood smear images
CANDIDATE_DATASETS = [
    "dona8584/leukemia",
    "nuraishahafzainizam/chronic-myeloid-leukemia",
    "eugeneshenderov/acute-promyelocytic-leukemia-apl",
    "shimantokumar/multi-class-leukemia-dataset-all-subtypesamlnormal"
]

def load_and_export_credentials():
    """Reads kaggle.json from home or common download locations and sets environment variables."""
    possible_paths = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path("F:/Downloads/kaggle.json"),
        Path.home() / "Downloads" / "kaggle.json"
    ]

    for p in possible_paths:
        if p.exists():
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                if "username" in data and "key" in data:
                    os.environ["KAGGLE_USERNAME"] = data["username"]
                    os.environ["KAGGLE_KEY"] = data["key"]
                    
                    # Ensure ~/.kaggle/kaggle.json exists as well
                    target = Path.home() / ".kaggle" / "kaggle.json"
                    if not target.exists() or p != target:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(p), str(target))
                        
                    print(f"Loaded credentials for user: {data['username']}")
                    return True
            except Exception:
                pass
                
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False

def try_download_dataset(slug, temp_dir):
    """Tries downloading a dataset using kagglehub first, then kaggle API."""
    print(f"\n" + "-" * 60)
    print(f"Attempting download for: {slug}")
    print("-" * 60)

    # 1. Try with kagglehub
    try:
        import kagglehub
        print(f"[KaggleHub] Requesting {slug}...")
        download_path = kagglehub.dataset_download(slug)
        if download_path and Path(download_path).exists():
            print(f"[KaggleHub Success] Downloaded to: {download_path}")
            return Path(download_path)
    except Exception as e:
        print(f"[KaggleHub Notice] {e}")

    # 2. Try with standard kaggle API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        target_path = temp_dir / slug.replace("/", "_")
        target_path.mkdir(parents=True, exist_ok=True)
        print(f"[Kaggle API] Downloading files into: {target_path}...")
        api.dataset_download_files(slug, path=str(target_path), unzip=True)
        print(f"[Kaggle API Success] Download and extraction complete.")
        return target_path
    except Exception as e:
        print(f"[Kaggle API Notice] {e}")

    return None

def sort_and_split_dataset(source_path, split_ratios=(0.70, 0.15, 0.15)):
    """
    Scans the downloaded folder for blood smear images, maps them to the 5 target classes,
    and sorts them into train (70%), validation (15%), and test (15%).
    """
    print("\n" + "=" * 65)
    print("📂 ORGANIZING DOWNLOADED BLOOD SMEAR IMAGES INTO DATASET FOLDERS")
    print("=" * 65)

    source_path = Path(source_path)
    train_ratio, val_ratio, test_ratio = split_ratios

    # Keyword mappings to 5 target clinical categories
    class_map = {
        "all": "ALL",
        "acute lymphoblastic": "ALL",
        "lymphoblast": "ALL",
        "aml": "AML",
        "acute myeloid": "AML",
        "myeloblast": "AML",
        "apl": "AML",
        "promyelocyte": "AML",
        "blast": "AML",
        "npm1": "AML",
        "cbfb": "AML",
        "runx1": "AML",
        "cll": "CLL",
        "chronic lymphocytic": "CLL",
        "cml": "CML",
        "chronic myeloid": "CML",
        "normal": "NORMAL",
        "healthy": "NORMAL",
        "benign": "NORMAL",
        "control": "NORMAL",
        "non-cancer": "NORMAL",
        "hem": "NORMAL"
    }

    discovered = {c: [] for c in CLASSES}
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # Recursively traverse directory
    for file_path in source_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            # Check directory ancestors and filename
            path_parts = [p.lower() for p in file_path.parts]
            matched_cls = None

            # First priority: direct ancestor folder name exactly matches class
            for p in file_path.parents:
                p_upper = p.name.upper()
                if p_upper in CLASSES:
                    matched_cls = p_upper
                    break

            # Second priority: folder name & keyword matching
            if not matched_cls:
                for part in path_parts:
                    part_clean = part.replace("-", " ").replace("_", " ")
                    for keyword, target_cls in class_map.items():
                        if keyword == part or keyword in part_clean:
                            matched_cls = target_cls
                            break
                    if matched_cls:
                        break

            if matched_cls:
                discovered[matched_cls].append(file_path)

    print("\nImages identified by category:")
    total_found = 0
    for cls in CLASSES:
        count = len(discovered[cls])
        total_found += count
        print(f"  - {cls:<8}: {count:>5} images")

    if total_found == 0:
        print("\n[Notice] Could not automatically map subfolder names.")
        print(f"Please check the downloaded folder contents at: {source_path}")
        return False

    # Distribute images
    print("\nDistributing files into train (70%), validation (15%), and test (15%) splits...")
    random.seed(42)

    for cls in CLASSES:
        files = discovered[cls]
        if not files:
            continue
        random.shuffle(files)

        n_total = len(files)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))

        splits = {
            TRAIN_DIR / cls: files[:n_train],
            VAL_DIR / cls: files[n_train:n_train + n_val],
            TEST_DIR / cls: files[n_train + n_val:]
        }

        for split_dir, split_files in splits.items():
            split_dir.mkdir(parents=True, exist_ok=True)
            for src_file in split_files:
                dest_file = split_dir / src_file.name
                if dest_file.exists():
                    dest_file = split_dir / f"{src_file.stem}_{random.randint(1000, 9999)}{src_file.suffix}"
                shutil.copy2(str(src_file), str(dest_file))

    print("\n✅ Dataset sorting and partitioning complete!")
    from preprocessing.dataset_split import verify_dataset_structure
    verify_dataset_structure()
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download and organize Leukemia dataset from Kaggle")
    parser.add_argument("--dataset", type=str, default=None, help="Specific Kaggle dataset slug")
    parser.add_argument("--local_dir", type=str, default=None, help="Path to an existing extracted folder")
    args = parser.parse_args()

    # If user provided a local directory directly
    if args.local_dir:
        local_p = Path(args.local_dir)
        if local_p.exists():
            print(f"\n📂 Processing local directory: {local_p}")
            if sort_and_split_dataset(local_p):
                print("\n🎉 Successfully sorted classes from local directory!")
                return
        else:
            print(f"Error: Provided local directory does not exist: {local_p}")
            return

    # Check for downloaded ZIP files in common directories
    import zipfile
    search_dirs = [
        Path("F:/Downloads"),
        Path.home() / "Downloads",
        DATASET_DIR / "raw_download"
    ]
    for s_dir in search_dirs:
        if s_dir.exists():
            for z in s_dir.glob("*.zip"):
                z_name = z.name.lower()
                if any(kw in z_name for kw in ["archive", "leukemia", "aml", "cll", "cml", "blood", "cancer"]):
                    print(f"\n📦 Found downloaded ZIP archive: {z}")
                    extract_target = DATASET_DIR / "raw_download" / z.stem
                    extract_target.mkdir(parents=True, exist_ok=True)
                    print(f"Extracting {z.name} to {extract_target}...")
                    try:
                        with zipfile.ZipFile(str(z), 'r') as zip_ref:
                            zip_ref.extractall(str(extract_target))
                        print("Extraction complete. Running automated sorting...")
                        if sort_and_split_dataset(extract_target):
                            print("\n🎉 Successfully sorted classes from downloaded ZIP archive!")
                            return
                    except Exception as e:
                        print(f"Error extracting {z}: {e}")

    # Check already downloaded raw subfolders
    temp_download_dir = DATASET_DIR / "raw_download"
    temp_download_dir.mkdir(parents=True, exist_ok=True)
    for raw_folder in temp_download_dir.iterdir():
        if raw_folder.is_dir() and any(raw_folder.glob("**/*.*")):
            # Check if this raw folder has files not yet in dataset
            print(f"Checking existing raw folder: {raw_folder.name}")
            sort_and_split_dataset(raw_folder)

    # Determine which datasets to try downloading from Kaggle
    datasets_to_try = [args.dataset] if args.dataset else CANDIDATE_DATASETS

    for slug in datasets_to_try:
        downloaded_dir = try_download_dataset(slug, temp_download_dir)
        if downloaded_dir:
            success = sort_and_split_dataset(downloaded_dir)
            if success:
                print(f"\n🎉 Successfully downloaded and installed dataset from {slug}!")

    print("\n" + "=" * 65)
    from preprocessing.dataset_split import verify_dataset_structure
    verify_dataset_structure()
    print("=" * 65)

if __name__ == "__main__":
    main()
