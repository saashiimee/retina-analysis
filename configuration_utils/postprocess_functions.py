import cv2
import math
import numpy as np

from skimage import measure, morphology


def connect_table(image, min_size, connect):
    label_image = measure.label(image)
    dst = morphology.remove_small_objects(label_image, min_size=min_size, connectivity=connect)
    return dst, measure.regionprops(dst)


def count_white(image):
    return np.count_nonzero(image)


def img_resize(image, scale):
    dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    return resized


def post_process(prob_result, prob_image):
    dst, region_props = connect_table(prob_result, 3000, 1)
    result = np.zeros_like(prob_result)
    prob = np.zeros_like(prob_image)
    candidates = []
    prob_result = prob_result.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    prob_result = cv2.morphologyEx(prob_result, cv2.MORPH_CLOSE, kernel)
    prob_result = cv2.morphologyEx(prob_result, cv2.MORPH_CLOSE, kernel)

    for region in region_props:
        min_r, min_c, max_r, max_c = region.bbox
        area = (max_r - min_r) * (max_c - min_c)

        if math.fabs((max_r - min_r) / (max_c - min_c)) > 1.3 \
                or math.fabs((max_r - min_r) / (max_c - min_c)) < 0.8 \
                or area * 4 / 3.1415926 < count_white(prob_result[min_r:max_r, min_c:max_c]):
            continue
        candidates.append(region.bbox)
    select_min_r = 0
    select_max_r = 0
    select_min_c = 0
    select_max_c = 0
    w_h_ratio = 0

    for candi in range(len(candidates)):
        min_r, min_c, max_r, max_c = candidates[candi]
        if math.fabs(w_h_ratio - 1.0) > math.fabs((max_r - min_r) / (max_c - min_c) - 1.0):
            select_min_r = min_r
            select_max_r = max_r
            select_min_c = min_c
            select_max_c = max_c
    result[select_min_r:select_max_r, select_min_c:select_max_c] = prob_result[
                                                               select_min_r:select_max_r,
                                                               select_min_c:select_max_c]
    prob[select_min_r:select_max_r, select_min_c:select_max_c] = prob_image[
                                                             select_min_r:select_max_r,
                                                             select_min_c:select_max_c]

    if np.max(prob) == 0:
        prob = prob_image
    return result.astype(np.uint8), prob
