import pytorch_lightning as pl
import torchmetrics
from torch import nn
from transformers import SegformerForSemanticSegmentation
import numpy as np
import torch


def iou(pred, target):
    scores = np.full(3, float('nan'))
    # ignore index 0 (background)
    for idx, cls in enumerate(target.unique()[1:]):
        y_pred = pred[:, cls, :, :]

        y_true = target == cls

        scores[idx] = torchmetrics.functional.iou(y_pred, y_true)

    metrics = {'leish': scores[0], 'macrofago contavel': scores[1],
               'macrofago nao contavel': scores[2], 'mIoU': scores.mean()}
    return metrics


class SemanticModel(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        id2label = {0: 'fundo', 1: 'leishmania',
                    2: 'macrofago contavel', 3: 'macrofago nao contavel'}
        label2id = {v: k for k, v in id2label.items()}
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b3", ignore_mismatched_sizes=True,
                                                                      num_labels=4, id2label=id2label, label2id=label2id,
                                                                      reshape_last_stage=True)

        self.jaccard_index = iou
        #self.device = 'cuda'  if torch.cuda.is_available() else 'cpu'

    def forward(self, pixel_values, labels):
        logits = self.model(pixel_values, labels)

        return logits

    def shared_step(self, batch, stage):
        image = batch["pixel_values"].to(self.device)
        labels = batch['labels'].to(self.device)

        outputs = self(pixel_values=image, labels=labels)

        #loss = self.loss_fn(logits_mask, mask)
        loss = outputs.loss  # CROSS ENTROPY LOSS

        #pred_mask = torch.nn.functional.softmax(logits_mask, dim=1)
        upsampled_logits = nn.functional.interpolate(
            outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        pred_mask = torch.argmax(upsampled_logits, dim=1)

        #iou_score = self.jaccard_index(upsampled_logits, labels.long())

        #iou_score = torchmetrics.functional.iou(pred_mask, labels.long())
        metrics = self.jaccard_index(upsampled_logits, labels.long())
        metrics.update({'loss': loss})
        return metrics

    def shared_epoch_end(self, output_list, stage):
        loss = torch.tensor([x['loss'] for x in output_list]).mean().item()
        leishmania_iou = torch.tensor([x['leish']
                                      for x in output_list]).mean().item()
        contavel_iou = torch.tensor(
            [x['macrofago contavel'] for x in output_list]).mean().item()
        nao_contavel_iou = torch.tensor(
            [x['macrofago nao contavel'] for x in output_list]).mean().item()

        metrics = {
            stage+'_loss': loss,
            stage+'_leishmania_iou': leishmania_iou,
            stage+'_contavel_iou': contavel_iou,
            stage+'_nao_contavel_iou': nao_contavel_iou,
        }

        self.log_dict(metrics, prog_bar=False)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=0.00006)
