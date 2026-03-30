import numpy as np
import json
from tqdm import tqdm
from pathlib import Path

def compute_stats(dataset, output_path="outputs/stats.json"):
    all_pixels = []
    labels = []

    for item in tqdm(dataset, desc="Computing stats"):
        img = item["image"]
        mask = item["mask"]

        roi_pixels = img[mask > 0]
        # Remove invalid values (ensure strictly calcium based on CT physical properties, calcium > 130 HU)
        roi_pixels = roi_pixels[roi_pixels > 130]
        
        if len(roi_pixels) > 0:
            all_pixels.extend(roi_pixels.flatten())

        labels.append(int(mask.sum() > 0))

    # Dataset Ratio (Normal:Disease)
    num_disease = int(sum(labels))
    num_normal = int(len(labels) - sum(labels))
    ratio = f"{num_normal}:{num_disease}"
    
    if len(all_pixels) > 0:
        all_pixels = np.array(all_pixels)
        mean_hu = float(all_pixels.mean())
        std_hu = float(all_pixels.std())
        min_hu = float(all_pixels.min())
        max_hu = float(all_pixels.max())
    else:
        mean_hu = std_hu = min_hu = max_hu = 0.0

    stats = {
        "total_samples": len(dataset),
        "positive_cases": num_disease,
        "negative_cases": num_normal,
        "ratio_normal_to_disease": ratio,
        "mean_hu": mean_hu,
        "std_hu": std_hu,
        "min_hu": min_hu,
        "max_hu": max_hu
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Stats saved to {output_path}")
    return stats