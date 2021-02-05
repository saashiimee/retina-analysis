import numpy as np
import cv2


# CLAHE (Contrast Limited Adaptive Histogram Equalization) adaptive histogram equalization is used. In this,
# image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks
# are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is
# noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin
# is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly
# to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders,
# bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape) == 4)
    assert (imgs.shape[1] == 1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = clahe.apply(np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized


# normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape) == 4)
    assert (imgs.shape[1] == 1)
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
                np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))) * 255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape) == 4)
    assert (imgs.shape[1] == 1)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i, 0] = cv2.LUT(np.array(imgs[i, 0], dtype=np.uint8), table)
    return new_imgs
