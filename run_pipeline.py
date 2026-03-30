import pandas as pd
from pathlib import Path

from src.split.stratified_split import create_splits, get_folders
from src.dataloader.coca_dataset import COCADataset, create_dataloader
from src.utils.stats import compute_stats

def main():
    project_root = Path(__file__).resolve().parent

    csv_path = project_root / "data_canonical" / "tables" / "scan_index.csv"
    
    if not csv_path.exists():
        print(f"[ERROR] Cannot find index CSV at {csv_path}.")
        return

    print("=" * 50)
    print("      COMMON TASK PIPELINE RUNTIME")
    print("=" * 50)

    print("\n[INFO] Computing dataset-level statistics...")
    # Read the full dataset index before splitting
    df = pd.read_csv(csv_path)
    all_folders = [Path(p) for p in df["folder_path"]]
    full_dataset = COCADataset(all_folders, apply_window=False)
    compute_stats(full_dataset)

    print("\n[INFO] Creating stratified splits...")
    train_df, val_df, test_df = create_splits(csv_path)

    train_folders = get_folders(train_df)
    val_folders = get_folders(val_df)
    test_folders = get_folders(test_df)

    print(f"Train samples: {len(train_folders)} "
          f"| Val samples: {len(val_folders)} "
          f"| Test samples: {len(test_folders)}")

    print("\n[INFO] Initializing Dataloaders...")
    train_loader = create_dataloader(train_folders, batch_size=2, shuffle=True)
    val_loader = create_dataloader(val_folders, batch_size=2, shuffle=False)
    test_loader = create_dataloader(test_folders, batch_size=2, shuffle=False)
    
    print(f"DataLoaders mapped successfully. (Train batches: {len(train_loader)})")

    print("\n[DONE] Common Task Pipeline Ready!\n")

if __name__ == "__main__":
    main()