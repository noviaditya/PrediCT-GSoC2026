import numpy as np
import random


def add_gaussian_noise(image, mean=0.0, std=0.01):
    """
    Add small Gaussian noise (safe for radiomics if small)
    """
    noise = np.random.normal(mean, std, image.shape)
    image = image + noise
    return image


def random_flip(image, mask, prob=0.5):
    """
    Flip along x-axis (left-right)
    Keep anatomical consistency
    """
    if random.random() < prob:
        image = np.flip(image, axis=2)
        mask = np.flip(mask, axis=2)
    return image, mask


def augment(image, mask):
    """
    Radiomics-safe augmentation
    """
    # Optional flip
    image, mask = random_flip(image, mask)

    # Small noise
    image = add_gaussian_noise(image)

    return image, mask