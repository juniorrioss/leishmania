from transformers import SegformerForSemanticSegmentation
import segmentation_models_pytorch as smp


class Config:
    def __init__(self) -> None:

        self.epochs = 20
        self.interval_metrics = 100
        self.data_workers = 16
        self.batch_size = 2
        self.test_size = 0.2
        self.image_height = 512
        self.image_width = 512

        # 'segformer', 'unet', 'unetplusplus', 'manet', 'linknet', 'fpn', 'pspnet', 'deeplabv3', 'deeplabv3plus', 'pan'
        self.arch = 'deeplabv3plus'
        # b0-b5 or encoder type e.g resnet101# https://smp.readthedocs.io/en/latest/encoders_timm.html
        self.backbone = 'resnext50_32x4d'

        self.id2label = {0: 'fundo', 1: 'leishmania',
                         2: 'macrofago contavel', 3: 'macrofago nao contavel'}
        self.label2id = {v: k for k, v in self.id2label.items()}

    def prepare_model(self):
        assert self.arch in ['segformer', 'unet', 'unetplusplus', 'manet', 'linknet',
                             'fpn', 'pspnet', 'deeplabv3', 'deeplabv3plus', 'pan'], 'Architeture not accepted'
        if self.arch == "segformer":
            return SegformerForSemanticSegmentation.from_pretrained(f"nvidia/mit-{self.backbone}", ignore_mismatched_sizes=True,
                                                                    num_labels=4, id2label=self.id2label, label2id=self.label2id,
                                                                    reshape_last_stage=True)

        else:
            return smp.create_model(
                self.arch, encoder_name=self.backbone, in_channels=3, classes=4, activation=None
            )
