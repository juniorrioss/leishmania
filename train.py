from data import LeishmaniaDataModule
from model import SemanticModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import albumentations as A
import cv2
import time
from config import Config
from dotenv import dotenv_values
import os

if __name__ == '__main__':
    cfg = Config()
    dotenv_config = dotenv_values(".env")

    os.environ['WANDB_API_KEY'] = dotenv_config['WANDB_API_KEY']

    N_EPOCHS = cfg.epochs
    BATCH_SIZE = cfg.batch_size
    IMAGE_PATH = 'croped_dataset_v2'
    MASK_PATH = 'mask_v2'

    # t_train = A.Compose([A.Resize(cfg.img_size, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(), A.VerticalFlip(),
    #                     A.GridDistortion(p=0.2), A.RandomBrightnessContrast(
    #                         (0, 0.5), (0, 0.5)),
    #                     A.GaussNoise()])

    t_train = A.Compose([A.Resize(cfg.image_height, cfg.image_width, interpolation=cv2.INTER_NEAREST),
                         A.ShiftScaleRotate(
                             shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                         A.RGBShift(r_shift_limit=25, g_shift_limit=25,
                                    b_shift_limit=25, p=0.5),
                         A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5), ])

    t_val = A.Compose(
        [A.Resize(cfg.image_height, cfg.image_width, interpolation=cv2.INTER_NEAREST)])

    print('LOADING THE DATA')
    datamodule = LeishmaniaDataModule(
        image_path=IMAGE_PATH, mask_path=MASK_PATH,
        train_transforms=t_train, test_transforms=t_val,
        batch_size=BATCH_SIZE
    )

    print('SETUP THE TRAINER')
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    logger = [
        TensorBoardLogger('lightning_logs', name=cfg.arch +
                          '_'+cfg.backbone+'_'+str(cfg.lr)),
        WandbLogger(name=cfg.arch+'_'+cfg.backbone+'_' + str(cfg.lr),
                    save_dir='wandblogs', project='leishmania')
    ]

    pl.seed_everything(0)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=N_EPOCHS,
        gpus=1,
        progress_bar_refresh_rate=30,
        checkpoint_callback=checkpoint_callback,
        precision=16
    )

    print('DOWNLOAD THE MODEL')
    model = SemanticModel()

    start = time.time()
    print('INICIO DO TREINAMENTO')
    trainer.fit(model, datamodule)
    end = time.time()
    print('\n\nFIM DO TREINAMENTO')
    print(end - start)
