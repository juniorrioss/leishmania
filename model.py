import pytorch_lightning as pl
import torchmetrics
from torch import nn
from transformers import SegformerForSemanticSegmentation
import torch
from datasets import load_metric
from config import Config
import segmentation_models_pytorch as smp
import numpy as np

cfg = Config()


class SemanticModel(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.id2label = cfg.id2label
        self.label2id = cfg.label2id
        self.model = cfg.prepare_model()

        self.metric = load_metric("mean_iou")

        # smp.losses.FocalLoss(mode='multiclass')
        # smp.losses.SoftCrossEntropyLoss()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, pixel_values):
        logits = self.model(pixel_values)

        if cfg.arch == 'segformer':
            logits = logits.logits

        return logits

    def shared_step(self, batch, stage, batch_idx):
        metric_computed = {}
        iou_score = np.nan

        image = batch["pixel_values"].to(self.device)  # image to device
        labels = batch['labels'].to(self.device)  # labels to device

        outputs = self(image)  # model predict

        if cfg.arch == 'segformer':
            # segformer does the predict in shape/8, so it's necessary to upsample to label shape
            outputs = nn.functional.interpolate(
                outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False)

        # loss = outputs.loss  # CROSS ENTROPY LOSS
        loss = self.loss_fn(outputs, labels)  # custom loss

        metrics = {'loss': loss}

        if batch_idx % cfg.interval_metrics == 0:
            with torch.no_grad():

                # argmax in N_class dimension
                pred_mask = torch.argmax(outputs, dim=1)
                iou_score = torchmetrics.functional.iou(
                    pred_mask, labels.long())  # iou metric in all predictions
                self.metric.add_batch(predictions=pred_mask.detach().cpu(
                ).numpy(), references=labels.long().detach().cpu().numpy())

                metric_computed = self.metric.compute(num_labels=len(self.id2label),
                                                      ignore_index=0,
                                                      reduce_labels=False,  # we've already reduced the labels before)
                                                      )

                # MERGE DICTIONARY (NEEDS PYTHON 3.9)
                #metrics = metrics | metric_computed
                metrics = {**metric_computed, **metrics}
                metrics['mIoU'] = iou_score

        return metrics

    def shared_epoch_end(self, output_list, stage):
        metrics = {}
        for metrics_dict in output_list:
            for k, v in metrics_dict.items():

                metrics.setdefault(stage+'_'+k, []).append(v)

        results = {}
        for k, v in metrics.items():
            v = torch.tensor(v)
            # torch.nanmean(a) == torch.mean(a[~a.isnan()])) # NEEDS TORCH 1.11
            results[k] = torch.mean(v[~v.isnan()])

        self.log_dict(results, prog_bar=False)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train", batch_idx)

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid", batch_idx)

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test", batch_idx)

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
