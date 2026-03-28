import numpy as np
import random

def random_flip(image, mask):
    if random.random() > 0.5:
        image = np.flip(image, axis=2)  # horizontal flip
        mask = np.flip(mask, axis=2)
    return image, mask


def random_rotate(image, mask):
    k = random.randint(0, 3)  # 0, 90, 180, 270 degrees
    image = np.rot90(image, k, axes=(1, 2))
    mask = np.rot90(mask, k, axes=(1, 2))
    return image, mask


def add_noise(image):
    noise = np.random.normal(0, 0.01, image.shape)
    return image + noise


def augment(image, mask):
    image, mask = random_flip(image, mask)
    image, mask = random_rotate(image, mask)
    image = add_noise(image)

    return image, mask