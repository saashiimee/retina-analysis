import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from abstract_classes.retina_predict_abstract import AbstractPredict
from configuration_utils.image_processing import get_test_patches
from configuration_utils.postprocess_functions import img_resize, post_process
from configuration_utils.extract_patches import predict_to_patches, recompose_overlap
from configuration_utils.utility_functions import visualize, gray2binary
from tensorflow.keras.models import model_from_json


class SegmentationPredict(AbstractPredict):
    def __init__(self, config):
        super(SegmentationPredict, self).__init__(config)
        self.model = model_from_json(
            open(self.config.hdf5_path + self.config.dataset_name + '_architecture.json').read())
        self.load_model()

    def load_model(self):
        self.model.load_weights(self.config.hdf5_path + self.config.dataset_name + '_best_weights.h5')

    def analyze_name(self, path):
        return (path.split('\\')[-1]).split(".")[0]

    def predict(self):
        groundtruth_list = glob.glob(self.config.test_groundtruth_path + "*." + self.config.test_datatype)
        predict_list = glob.glob(self.config.test_img_path + "*." + self.config.test_datatype)
        for (path, gt_path) in zip(predict_list, groundtruth_list):
            org_img_temp = plt.imread(path)
            org_img = org_img_temp[:, :, 1] * 0.75 + org_img_temp[:, :, 0] * 0.25
            height, width = org_img.shape[:2]
            org_img = np.reshape(org_img, (height, width, 1))

            gt_img_temp = plt.imread(gt_path)
            gt_img = gt_img_temp[:, :] * 0.75 + gt_img_temp[:, :] * 0.25
            height_gt, width_gt = gt_img.shape[:2]
            gt_img = np.reshape(gt_img, (height_gt, width_gt, 3))

            patches_predict, new_height, new_width, adjust_img, adjust_gt_img = get_test_patches(org_img, gt_img, self.config)

            predictions = self.model.predict(patches_predict, batch_size=32, verbose=1)
            pred_patches = predict_to_patches(predictions, self.config)

            predict_images = recompose_overlap(pred_patches, self.config, new_height, new_width)
            predict_images = predict_images[:, 0:height, 0:width, :]

            adjust_img = adjust_img[0, 0:height, 0:width, :]
            print(adjust_img.shape)
            adjust_gt_img = adjust_gt_img[0, 0:height, 0:width, :]
            print(adjust_gt_img.shape)

            prob_result = predict_images[0, :, :, 0]
            binary_result = gray2binary(prob_result)

            result_merge = visualize([adjust_img, adjust_gt_img, binary_result, prob_result], [1, 4])
            result_merge = cv2.cvtColor(result_merge, cv2.COLOR_RGB2BGR)

            cv2.imwrite(self.config.test_result_path + self.analyze_name(path) + "_merge.jpg", result_merge)
            cv2.imwrite(self.config.test_result_path + self.analyze_name(path) + "_prob.bmp",
                        (prob_result * 255).astype(np.uint8))

    # def predict(self):
    #     groundtruth_list = glob.glob(self.config.test_groundtruth_path + "*." + self.config.test_datatype)
    #     predict_list = glob.glob(self.config.test_img_path + "*." + self.config.test_datatype)
    #
    #     for (path, gt_path) in zip(predict_list, groundtruth_list):
    #         org_img_temp = plt.imread(path)
    #         if org_img_temp.shape[0] > 1000 or org_img_temp.shape[1] > 1000:
    #             org_img_temp = img_resize(org_img_temp, 0.5)
    #         org_img = org_img_temp[:, :, 1] * 0.75 + org_img_temp[:, :, 0] * 0.25
    #         height, width = org_img.shape[:2]
    #         org_img = np.reshape(org_img, (height, width, 1))
    #
    #         gt_img_temp = plt.imread(gt_path)
    #         if gt_img_temp.shape[0] > 1000 or gt_img_temp.shape[1] > 1000:
    #             org_img_temp = img_resize(gt_img_temp, 0.5)
    #         gt_img = gt_img_temp[:, :] * 0.75 + gt_img_temp[:, :] * 0.25
    #         height, width = gt_img.shape[:2]
    #         gt_img = np.reshape(gt_img, (height, width, 1))
    #
    #         patches_predict, new_height, new_width, adjust_img, adjust_gt_img = get_test_patches(org_img, gt_img, self.config)
    #
    #         predictions = self.model.predict(patches_predict, batch_size=32, verbose=1)
    #         predict_patches = predict_to_patches(predictions, self.config)
    #
    #         predict_images = recompose_overlap(predict_patches, self.config, new_height, new_width)
    #         predict_images = predict_images[:, 0:height, 0:width, :]
    #
    #         prob_result = predict_images[0, :, :, 0]
    #         binary_result = gray2binary(prob_result, threshold=0.5)
    #
    #         binary_result, prob_result = post_process(binary_result, prob_result)  # post process to get the optic disc
    #
    #         contours, _ = cv2.findContours(binary_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #         adjust_img = cv2.drawContours(org_img_temp, contours, -1, (0, 255, 0), 2)
    #
    #         result_merge = visualize([adjust_img / 255., adjust_gt_img, binary_result, prob_result], [1, 4])
    #         result_merge = cv2.cvtColor(result_merge, cv2.COLOR_RGB2BGR)
    #
    #         cv2.imwrite(self.config.test_result_path + self.analyze_name(path) + "_merge.jpg", result_merge)
    #         cv2.imwrite(self.config.test_result_path + self.analyze_name(path) + "_prob.bmp",
    #                     (prob_result * 255).astype(np.uint8))