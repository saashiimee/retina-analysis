from configuration.configuration import prepare_config
from data_loader.data_loader import DataLoader
from dense_unet import SegmentationModel
from segmentation.retina_train import SegmentationTrain


def main_train():
    print('[INFO] Reading Configs...')

    config = None

    try:
        config = prepare_config('config/segmentation_config.json')
    except Exception as e:
        print('[Exception] Config Error, %s' % e)
        exit(0)

    print('[INFO] Preparing Data...')
    data_loader = DataLoader(config=config)
    data_loader.prepare_dataset()

    train_imgs, train_gt = data_loader.get_train_data()
    val_imgs, val_gt = data_loader.get_val_data()

    print('[INFO] Building Model...')
    model = SegmentationModel(config=config)

    print('[INFO] Training...')
    train_segment = SegmentationTrain(
        model=model.model,
        data=[train_imgs, train_gt, val_imgs, val_gt],
        config=config)
    train_segment.train()
    print('[INFO] Finishing...')


if __name__ == '__main__':
    main_train()
