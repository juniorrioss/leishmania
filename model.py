import pytorch_lightning as pl
import torchmetrics
from torch import nn
from transformers import SegformerForSemanticSegmentation
import torch
from datasets import load_metric
from config import Config

cfg = Config()


class SemanticModel(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.id2label = cfg.id2label
        self.label2id = cfg.label2id
        self.model = cfg.prepare_model()

        self.metric = load_metric("mean_iou")

    def forward(self, pixel_values, labels):
        logits = self.model(pixel_values, labels)

        return logits

    def shared_step(self, batch, stage, batch_idx):
        metric_computed = {}
        iou_score = torch.nan

        image = batch["pixel_values"].to(self.device)  # image to device
        labels = batch['labels'].to(self.device)  # labels to device

        outputs = self(pixel_values=image, labels=labels)  # model predict

        loss = outputs.loss  # CROSS ENTROPY LOSS
        # loss = self.loss_fn(logits_mask, mask) # custom loss

        metrics = {'loss': loss}

        # segformer does the predict in shape/8, so it's necessary to upsample to label shape
        upsampled_logits = nn.functional.interpolate(
            outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)

        if batch_idx % cfg.interval_metrics == 0:
            with torch.no_grad():
                # argmax in N_class dimension
                pred_mask = torch.argmax(upsampled_logits, dim=1)
                iou_score = torchmetrics.functional.iou(
                    pred_mask, labels.long())  # iou metric in all predictions
                self.metric.add_batch(predictions=pred_mask.detach().cpu(
                ).numpy(), references=labels.long().detach().cpu().numpy())

                metric_computed = self.metric.compute(num_labels=len(self.id2label),
                                                      ignore_index=0,
                                                      reduce_labels=False,  # we've already reduced the labels before)
                                                      )

                # MERGE DICTIONARY (NEEDS PYTHON 3.9)
                metrics = metrics | metric_computed
                metrics['mIoU'] = iou_score

        return metrics

    def shared_epoch_end(self, output_list, stage):
        # gathering all metrics for batches and get the Average
        metrics = {}
        for metrics_dict in output_list:
            for k, v in metrics_dict.items():

                metrics.setdefault(k, []).append(v)

        results = {}
        for k, v in metrics.items():
            results[stage+'_'+k] = torch.tensor(v).nanmean().item()

        print(results)

        # loss = torch.tensor([x['loss'] for x in output_list]).mean().item()
        # miou_score = torch.tensor([x['mIoU']
        #                           for x in output_list]).mean().item()
        # mean_ioU_score = torch.tensor(
        #     [x['mean_IoU'] for x in output_list]).mean().item()

        # metrics = {
        #     stage+'_loss': loss,
        #     stage+'_mIoU': miou_score,
        #     stage+'_mean_ioU_score': mean_ioU_score,
        # }

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
        return torch.optim.AdamW(self.model.parameters(), lr=0.00006)
