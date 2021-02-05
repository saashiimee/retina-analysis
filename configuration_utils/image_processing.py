import numpy as np

from configuration_utils.preprocessing_functions import dataset_normalized
from configuration_utils.preprocessing_functions import clahe_equalized
from configuration_utils.preprocessing_functions import adjust_gamma

from configuration_utils.extract_patches import extract_patches
from configuration_utils.extract_patches import paint_border


def img_process(data, rl=False):
    assert (len(data.shape) == 4)
    data = data.transpose(0, 3, 1, 2)
    if rl:
        train_imgs = np.zeros(data.shape)
        for index in range(data.shape[1]):
            train_img = np.zeros([data.shape[0], 1, data.shape[2], data.shape[3]])
            train_img[:, 0, :, :] = data[:, index, :, :]
            # print("original",np.max(train_img),np.min(train_img))
            train_img = dataset_normalized(train_img)
            # print("normal",np.max(train_img), np.min(train_img))
            train_img = clahe_equalized(train_img)
            # print("clahe",np.max(train_img), np.min(train_img))
            train_img = adjust_gamma(train_img, 1.2)
            # print("gamma",np.max(train_img), np.min(train_img))
            train_img = train_img / 255.
            # print("reduce",np.max(train_img), np.min(train_img))
            train_imgs[:, index, :, :] = train_img[:, 0, :, :]

    else:
        train_imgs = np.zeros(data.shape)
        for index in range(data.shape[1]):
            train_img = np.zeros([data.shape[0], 1, data.shape[2], data.shape[3]])
            train_img[:, 0, :, :] = data[:, index, :, :]
            train_img = dataset_normalized(train_img)
            train_imgs[:, index, :, :] = train_img[:, 0, :, :] / 255.

    train_imgs = train_imgs.transpose(0, 2, 3, 1)
    return train_imgs


def get_test_patches(img, config, rl=False):
    test_img = [img]

    test_img = np.asarray(test_img)

    test_img_adjust = img_process(test_img, rl=rl)
    test_imgs = paint_border(test_img_adjust, config)

    test_img_patch = extract_patches(test_imgs, config)

    return test_img_patch, test_imgs.shape[1], test_imgs.shape[2], test_img_adjust


