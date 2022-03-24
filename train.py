from data import LeishmaniaDataModule
from model import SemanticModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import albumentations as A
import cv2


if __name__ == '__main__':

    N_EPOCHS = 1
    BATCH_SIZE = 1
    IMAGE_PATH = 'croped_dataset'
    MASK_PATH = 'mask'

    t_train = A.Compose([A.Resize(768, 768, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(), A.VerticalFlip(),
                        A.GridDistortion(p=0.2), A.RandomBrightnessContrast(
                            (0, 0.5), (0, 0.5)),
                        A.GaussNoise()])

    t_val = A.Compose([A.Resize(768, 768, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(),
                       A.GridDistortion(p=0.2)])

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

    logger = TensorBoardLogger('lightning_logs', name='segformer-b0')

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=N_EPOCHS,
        gpus=1,
        progress_bar_refresh_rate=30,
        checkpoint_callback=checkpoint_callback
    )

    print('DOWNLOAD THE MODEL')
    model = SemanticModel()

    trainer.fit(model, datamodule)
