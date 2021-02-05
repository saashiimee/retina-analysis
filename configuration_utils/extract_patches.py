import numpy as np


def extract_patches(full_imgs, config):
    assert (len(full_imgs.shape) == 4)
    img_h = full_imgs.shape[1]
    img_w = full_imgs.shape[2]

    assert ((img_h - config.patch_height) % config.stride_height == 0 and (
            img_w - config.patch_width) % config.stride_width == 0)
    n_patches_img = ((img_h - config.patch_height) // config.stride_height + 1) * (
            (img_w - config.patch_width) // config.stride_width + 1)
    n_patches_total = n_patches_img * full_imgs.shape[0]

    patches = np.empty((n_patches_total, config.patch_height, config.patch_width, full_imgs.shape[3]))
    iter_tot = 0
    for i in range(full_imgs.shape[0]):
        for h in range((img_h - config.patch_height) // config.stride_height + 1):
            for w in range((img_w - config.patch_width) // config.stride_width + 1):
                patch = full_imgs[i, h * config.stride_height:(h * config.stride_height) + config.patch_height,
                        w * config.stride_width:(w * config.stride_width) + config.patch_width, :]
                patches[iter_tot] = patch
                iter_tot += 1
    assert (iter_tot == n_patches_total)
    return patches


def predict_to_patches(predict, config):
    assert (len(predict.shape) == 3)

    predict_images = np.empty((predict.shape[0], predict.shape[1], config.seg_num + 1))
    predict_images[:, :, 0:config.seg_num + 1] = predict[:, :, 0:config.seg_num + 1]
    predict_images = np.reshape(predict_images,
                                (predict_images.shape[0], config.patch_height,
                                 config.patch_width, config.seg_num + 1))
    return predict_images


def paint_border(imgs, config):
    assert (len(imgs.shape) == 4)
    img_h = imgs.shape[1]
    img_w = imgs.shape[2]
    leftover_h = (img_h - config.patch_height) % config.stride_height
    leftover_w = (img_w - config.patch_width) % config.stride_width
    full_imgs = None
    if leftover_h != 0:
        tmp_imgs = np.zeros((imgs.shape[0], img_h + (config.stride_height - leftover_h), img_w, imgs.shape[3]))
        tmp_imgs[0:imgs.shape[0], 0:img_h, 0:img_w, 0:imgs.shape[3]] = imgs
        full_imgs = tmp_imgs
    if leftover_w != 0:
        tmp_imgs = np.zeros(
            (full_imgs.shape[0], full_imgs.shape[1], img_w + (config.stride_width - leftover_w), full_imgs.shape[3]))
        tmp_imgs[0:imgs.shape[0], 0:imgs.shape[1], 0:img_w, 0:full_imgs.shape[3]] = imgs
        full_imgs = tmp_imgs
    print("new full images shape: \n" + str(full_imgs.shape))
    return full_imgs


def recompose_overlap(preds, config, img_h, img_w):
    assert (len(preds.shape) == 4)

    patch_h = config.patch_height
    patch_w = config.patch_width
    n_patches_h = (img_h - patch_h) // config.stride_height + 1
    n_patches_w = (img_w - patch_w) // config.stride_width + 1
    n_patches_img = n_patches_h * n_patches_w
    print("n_patches_h: " + str(n_patches_h))
    print("n_patches_w: " + str(n_patches_w))
    print("n_patches_img: " + str(n_patches_img))
    n_full_imgs = preds.shape[0] // n_patches_img
    print("According to the dimension inserted, there are " + str(n_full_imgs) + " full images (of " + str(
        img_h) + "x" + str(img_w) + " each)")
    full_prob = np.zeros(
        (n_full_imgs, img_h, img_w, preds.shape[3]))
    full_sum = np.zeros((n_full_imgs, img_h, img_w, preds.shape[3]))

    k = 0
    for i in range(n_full_imgs):
        for h in range((img_h - patch_h) // config.stride_height + 1):
            for w in range((img_w - patch_w) // config.stride_width + 1):
                full_prob[i, h * config.stride_height:(h * config.stride_height) + patch_h,
                w * config.stride_width:(w * config.stride_width) + patch_w, :] += preds[k]
                full_sum[i, h * config.stride_height:(h * config.stride_height) + patch_h,
                w * config.stride_width:(w * config.stride_width) + patch_w, :] += 1
                k += 1
    print(k, preds.shape[0])
    assert (k == preds.shape[0])
    assert (np.min(full_sum) >= 1.0)
    final_avg = full_prob / full_sum
    print('using avg')
    return final_avg
