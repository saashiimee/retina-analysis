import json
import os
import shutil

from bunch import Bunch


def mkdir_if_not_exist(dir_name, is_delete=False):
    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print(u'[INFO] Directory "%s" exists, deleting.' % dir_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(u'[INFO] Directory "%s" not exists, creating.' % dir_name)
        return True
    except Exception as e:
        print('[Error] %s' % e)
        return False


def get_config_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = Bunch(config_dict)

    return config, config_dict


def prepare_config(json_file):
    config, _ = get_config_json(json_file)

    config.checkpoint = os.path.join("./data", config.dataset_name, "checkpoint/")
    config.hdf5_path = os.path.join("./data", config.dataset_name, "hdf5/")
    config.train_img_path = os.path.join("./data", config.dataset_name, "dataset/training/images/")
    config.train_groundtruth_path = os.path.join("./data", config.dataset_name, "dataset/training/groundtruth/")
    config.validate_img_path = os.path.join("./data", config.dataset_name, "dataset/validate/images/")
    config.validate_groundtruth_path = os.path.join("./data", config.dataset_name, "dataset/validate/groundtruth/")
    config.test_img_path = os.path.join("./data", config.dataset_name, "test/images/")
    config.test_groundtruth_path = os.path.join("./data", config.dataset_name, "test/groundtruth/")
    config.test_result_path = os.path.join("./data", config.dataset_name, "test/result/")

    mkdir_if_not_exist(config.checkpoint)
    mkdir_if_not_exist(config.hdf5_path)
    mkdir_if_not_exist(config.train_img_path)
    mkdir_if_not_exist(config.train_groundtruth_path)
    mkdir_if_not_exist(config.validate_img_path)
    mkdir_if_not_exist(config.validate_groundtruth_path)
    mkdir_if_not_exist(config.test_img_path)
    mkdir_if_not_exist(config.test_groundtruth_path)
    mkdir_if_not_exist(config.test_result_path)

    return config
