import cv2
import os
from torchvision import transforms as T
import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from config import Config

cfg = Config()

class Leishdata:
    def __init__(self, img_dir, mask_dir, image_list, mean, std, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_list = image_list
        self.length = len(self.img_list)
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_dir, self.img_list[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(
            self.mask_dir, self.img_list[idx]), cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)

        mask = torch.from_numpy(mask).long()

        return {'pixel_values': img, 'labels': mask}


class LeishmaniaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_path: str,
        mask_path: str,
        batch_size: int = 2,
        test_size: int = 0.2,
        train_transforms=None,
        test_transforms=None,
    ):
        super().__init__()

        self.image_path = image_path
        self.mask_path = mask_path
        self.batch_size = batch_size
        self.image_list = [i for i in os.listdir(
            image_path) if i.endswith('.png')]  # ALL IMAGE DATA
        self.test_size = test_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def setup(self, stage=None):

        train_list, test_list = train_test_split(
            self.image_list, test_size=self.test_size)

        self.train_dataset = Leishdata(
            self.image_path, self.mask_path, train_list, self.mean, self.std, self.train_transforms)
        self.test_dataset = Leishdata(
            self.image_path, self.mask_path, test_list, self.mean, self.std, self.test_transforms)

        print("Number of training examples:", len(self.train_dataset))
        print("Number of validation examples:", len(self.test_dataset))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=cfg.data_workers)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=cfg.data_workers)
