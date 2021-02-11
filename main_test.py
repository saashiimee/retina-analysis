from segmentation.retina_predict import SegmentationPredict
from statistics import *
from configuration.configuration import prepare_config

repeat_prediction = True


def main_test():
    print('[INFO] Reading Configs...')
    config = None

    try:
        config = prepare_config('configuration/segmentation_config.json')
    except Exception as e:
        print('[Error] Config Error, %s' % e)
        exit(0)

    if repeat_prediction:
        print('[INFO] Predicting...')
        prediction = SegmentationPredict(config)
        prediction.predict()

    print('[INFO] Metric results...')
    gt_list = fileList(config.test_groundtruth_path, '*' + config.test_gt_datatype)
    prob_list = fileList(config.test_result_path, '*.bmp')
    model_name = ['attention u-net']
    drawCurve(gt_list, [prob_list], model_name, 'DRION-DB', config.checkpoint)

    print('[INFO] Finished...')


if __name__ == '__main__':
    main_test()
