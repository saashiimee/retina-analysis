import h5py
import numpy as np


def load_hdf5(file):
    with h5py.File(file, "r") as f:
        return f["image"][()]


def write_hdf5(arr, file):
    with h5py.File(file, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def generate_masks(masks, channels):
    assert (len(masks.shape) == 4)
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks, (masks.shape[0], channels, im_h * im_w))
    new_masks = np.empty((masks.shape[0], im_h * im_w, channels + 1))

    new_masks[:, :, 0:channels] = masks[:, 0:channels, :].transpose(0, 2, 1)
    mask_total = np.ma.array(new_masks[:, :, 0], mask=new_masks[:, :, 0]).mask
    for index in range(channels):
        mask = new_masks[:, :, index]
        m = np.ma.array(new_masks[:, :, index], mask=mask)
        mask_total = mask_total | m.mask

    new_masks[:, :, channels] = 1 - mask_total
    return new_masks


def gray2binary(image, threshold=0.5):
    image = (image >= threshold) * 1
    return image


def colorize(img, gt, prob):
    image = np.copy(img)
    if np.max(gt) > 1:
        gt = gt / 255.
    gt_list = [np.where(gt >= 0.5)]
    prob_list = [np.where(prob == 1)]
    gt_x = gt_list[0][0]
    gt_y = gt_list[0][1]
    for index in range(gt_x.shape[0]):
        image[gt_x[index], gt_y[index], 0] = 0
        image[gt_x[index], gt_y[index], 1] = 1
        image[gt_x[index], gt_y[index], 2] = 0

    prob_x = prob_list[0][0]
    prob_y = prob_list[0][1]
    for index in range(prob_x.shape[0]):
        if image[prob_x[index], prob_y[index], 1] != 1:
            image[prob_x[index], prob_y[index], 0] = 1
            image[prob_x[index], prob_y[index], 1] = 0
            image[prob_x[index], prob_y[index], 2] = 0
        else:
            image[prob_x[index], prob_y[index], 0] = 0
            image[prob_x[index], prob_y[index], 1] = 0
            image[prob_x[index], prob_y[index], 2] = 1
    return image


def visualize(image, subplot):
    row = int(subplot[0])
    col = int(subplot[1])
    height, width = image[0].shape[:2]
    result = np.zeros((height * row, width * col, 3))

    total_image = len(image)
    index = 0
    for i in range(row):
        for j in range(col):
            row_index = i * height
            col_index = j * width
            if index < total_image:
                try:
                    result[row_index:row_index + height, col_index:col_index + width, :] = image[index] * 255
                except:
                    result[row_index:row_index + height, col_index:col_index + width, 0] = image[index] * 255
                    result[row_index:row_index + height, col_index:col_index + width, 1] = image[index] * 255
                    result[row_index:row_index + height, col_index:col_index + width, 2] = image[index] * 255
            index = index + 1
    result = result.astype(np.uint8)
    return result
