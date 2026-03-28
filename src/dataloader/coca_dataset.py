import SimpleITK as sitk
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from src.preprocessing.hu_window import hu_window

class COCADataset(Dataset):
    def __init__(self, folders, apply_window=True):
        self.folders = folders
        self.apply_window = apply_window

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        scan_id = folder.name

        img_path = folder / f"{scan_id}_img.nii.gz"
        seg_path = folder / f"{scan_id}_seg.nii.gz"

        # Load image
        img = sitk.ReadImage(str(img_path))
        img = sitk.GetArrayFromImage(img)

        # Load mask
        seg = sitk.ReadImage(str(seg_path))
        seg = sitk.GetArrayFromImage(seg)

        if self.apply_window:
            img = hu_window(img)

        return {
            "image": img,
            "mask": seg,
            "id": scan_id
        }