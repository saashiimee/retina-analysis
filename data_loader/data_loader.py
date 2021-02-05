import glob
import numpy as np
import matplotlib.pyplot as plt

from abstract_classes.data_loader_abstract import AbstractDataLoader
from configuration_utils.utility_functions import load_hdf5, write_hdf5


class DataLoader(AbstractDataLoader):
    def __init__(self, config=None):
        super(DataLoader, self).__init__(config)
        self.dataset_name = config.dataset_name
        self.hdf5_path = config.hdf5_path
        self.height = config.height
        self.width = config.width
        self.num_seg_class = config.seg_num

        self.train_img_path = config.train_img_path
        self.train_groundtruth_path = config.train_groundtruth_path
        self.train_type = config.train_datatype
        self.validate_img_path = config.validate_img_path
        self.validate_groundtruth_path = config.validate_groundtruth_path
        self.validate_type = config.validate_datatype

    def access_dataset(self, image_path, groundtruth_path, datatype):
        img_list = glob.glob(image_path + "*." + datatype)
        gt_list = glob.glob(groundtruth_path + "*." + datatype)

        assert (len(img_list) == len(gt_list))

        imgs = np.empty((len(img_list), self.height, self.width, 1))
        groundtruth = np.empty((len(gt_list), self.num_seg_class, self.height, self.width))

        for index in range(len(img_list)):
            input_image_path = img_list[index]
            input_image = plt.imread(input_image_path)
            imgs[index, :, :, 0] = np.asarray(input_image[:, :, 1] * 0.75 + input_image[:, :, 0] * 255)

            for no_seg in range(self.num_seg_class):
                input_groundtruth_path = gt_list[index]
                input_groundtruth_image = plt.imread(input_groundtruth_path, 0)
                groundtruth[index, no_seg] = np.asarray(input_groundtruth_image)

        print("[INFO] Reading Data...")
        assert (np.max(groundtruth) == 255)
        assert (np.min(groundtruth) == 0)
        return imgs, groundtruth

    def prepare_dataset(self):
        train_imgs, groundtruth = self.access_dataset(self.train_img_path, self.train_groundtruth_path, self.train_type)
        write_hdf5(train_imgs, self.hdf5_path + "/train_img.hdf5")
        write_hdf5(groundtruth, self.hdf5_path + "/train_groundtruth.hdf5")
        print("[INFO] Saving Training Data...")

        validate_imgs, groundtruth = self.access_dataset(self.validate_img_path, self.validate_groundtruth_path,
                                                         self.validate_type)
        write_hdf5(validate_imgs, self.hdf5_path + "/validate_img.hdf5")
        write_hdf5(groundtruth, self.hdf5_path + "/validate_groundtruth.hdf5")
        print("[INFO] Saving Validation Data...")

    def get_train_data(self):
        train_imgs = load_hdf5(self.hdf5_path + "/train_img.hdf5")
        train_groundtruth = load_hdf5(self.hdf5_path + "/train_groundtruth.hdf5")
        return train_imgs, train_groundtruth

    def get_validate_data(self):
        validate_imgs = load_hdf5(self.hdf5_path + "/validate_img.hdf5")
        validate_groundtruth = load_hdf5(self.hdf5_path + "/validate_groundtruth.hdf5")
        return validate_imgs, validate_groundtruth
