from configuration.configuration import prepare_config
from datasets.data_loader.data_loader import DataLoader
from segmentation.dense_unet import SegmentationModel as ModelDenseUNet
from segmentation.attention_unet import SegmentationModel as ModelAttentionBasedUNet
from segmentation.retina_train import SegmentationTrain


def main_train(model_name):
    print('[INFO] Reading Configs...')

    config = None

    try:
        config = prepare_config('configuration/segmentation_config')
    except Exception as e:
        print('[Error] Config Error, %s' % e)
        exit(0)

    data_loader = DataLoader(config=config)
    data_loader.prepare_dataset()
    print('[INFO] Preparing Data...')

    train_imgs, train_groundtruth = data_loader.get_train_data()
    validate_imgs, validate_groundtruth = data_loader.get_validate_data()

    print('[INFO] Building Model...')
    if model_name == "ModelDenseUNet":
        model = ModelDenseUNet(config=config)
    elif model_name == "ModelAttentionBasedUNet":
        model = ModelAttentionBasedUNet(config=config)

    print('[INFO] Training...')
    train_segment = SegmentationTrain(
        model=model.model,
        data=[train_imgs, train_groundtruth, validate_imgs, validate_groundtruth],
        config=config)
    train_segment.train()
    print('[INFO] Finishing...')


if __name__ == '__main__':
    # main_train("ModelDenseUNet")
    main_train("ModelAttentionBasedUNet")
