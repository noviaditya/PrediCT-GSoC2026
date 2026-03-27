import numpy as np

def hu_window(image, level=300, width=800):
    """
    Apply HU windowing for cardiac CT
    """
    min_val = level - width // 2
    max_val = level + width // 2

    image = np.clip(image, min_val, max_val)
    image = (image - min_val) / (max_val - min_val)

    return image.astype(np.float32)