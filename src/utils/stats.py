import numpy as np
import json
from tqdm import tqdm

def compute_stats(dataset, output_path="outputs/stats.json"):
    intensities = []
    labels = []

    for item in tqdm(dataset, desc="Computing stats"):
        img = item["image"]
        mask = item["mask"]

        intensities.append(img.mean())
        labels.append(int(mask.sum() > 0))

    # Dataset Ratio (Normal:Disease)
    num_disease = int(sum(labels))
    num_normal = int(len(labels) - sum(labels))
    ratio = f"{num_normal}:{num_disease}"
    
    stats = {
        "total_samples": len(dataset),
        "positive_cases": num_disease,
        "negative_cases": num_normal,
        "ratio_normal_to_disease": ratio,
        "mean_intensity": float(np.mean(intensities)),
        "std_intensity": float(np.std(intensities))
    }

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Stats saved to {output_path}")
    return stats