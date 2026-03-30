import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import label

def get_agatston_weight(max_hu):
    if 130 <= max_hu < 200:
        return 1
    elif 200 <= max_hu < 300:
        return 2
    elif 300 <= max_hu < 400:
        return 3
    elif max_hu >= 400:
        return 4
    return 0

def calculate_scan_agatston(img_path, mask_path):
    # Read the canonical (unresampled) image to get accurate physical spacing
    img_sitk = sitk.ReadImage(str(img_path))
    mask_sitk = sitk.ReadImage(str(mask_path))

    # X, Y define the area of a slice
    spacing = img_sitk.GetSpacing()
    pixel_area_mm2 = spacing[0] * spacing[1]

    img = sitk.GetArrayFromImage(img_sitk)
    mask = sitk.GetArrayFromImage(mask_sitk)

    total_agatston = 0.0

    # Agatston is computed slice by slice
    for z in range(img.shape[0]):
        slice_img = img[z]
        slice_mask = mask[z]

        # Region of interest where mask indicates plaque AND density > 130
        roi = (slice_mask > 0) & (slice_img > 130)

        if not np.any(roi):
            continue

        # Find isolated lesions in the 2D slice
        labeled_lesions, num_lesions = label(roi)

        for i in range(1, num_lesions + 1):
            lesion_pixels = (labeled_lesions == i)
            lesion_area = np.sum(lesion_pixels) * pixel_area_mm2

            # Only consider lesions > 1 mm^2
            if lesion_area >= 1.0:
                max_density = np.max(slice_img[lesion_pixels])
                weight = get_agatston_weight(max_density)
                total_agatston += lesion_area * weight

    return total_agatston

def get_risk_category(score):
    if score == 0:
        return '0'
    elif 1 <= score <= 99:
        return '1-99'
    elif 100 <= score <= 399:
        return '100-399'
    return '≥400'

def compute_dataset_agatston(canonical_dir, output_csv="outputs/agatston_scores.csv"):
    csv_path = Path(canonical_dir) / "tables" / "scan_index.csv"
    if not csv_path.exists():
        print(f"[ERROR] Cannot find index for Agatston {csv_path}")
        return

    df = pd.read_csv(csv_path)
    results = []

    for folder_path in tqdm(df["folder_path"], desc="Computing Agatston Scores"):
        folder = Path(folder_path)
        scan_id = folder.name
        img_path = folder / f"{scan_id}_img.nii.gz"
        seg_path = folder / f"{scan_id}_seg.nii.gz"

        if img_path.exists() and seg_path.exists():
            score = calculate_scan_agatston(img_path, seg_path)
            category = get_risk_category(score)
            results.append({"scan_id": scan_id, "agatston_score": score, "risk_category": category})

    results_df = pd.DataFrame(results)
    
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved Agatston classifications to {output_csv}")
