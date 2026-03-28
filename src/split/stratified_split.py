import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def create_splits(csv_path):
    df = pd.read_csv(csv_path)

    # Create binary label
    df["label"] = (df["voxels"] > 0).astype(int)

    # First split
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=42
    )

    # Second split
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )

    return train_df, val_df, test_df


def get_folders(df):
    return [Path(p) for p in df["folder_path"]]