import pytorch_lightning as pl
import torchmetrics
from torch import nn
from transformers import SegformerForSemanticSegmentation
import torch



class SemanticModel(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        id2label = {0: 'fundo', 1: 'leishmania',
                    2: 'macrofago contavel', 3: 'macrofago nao contavel'}
        label2id = {v: k for k, v in id2label.items()}
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b3", ignore_mismatched_sizes=True,
                                                                      num_labels=4, id2label=id2label, label2id=label2id,
                                                                      reshape_last_stage=True)

    def forward(self, pixel_values, labels):
        logits = self.model(pixel_values, labels)

        return logits

    def shared_step(self, batch, stage):
        image = batch["pixel_values"].to(self.device) # image to device
        labels = batch['labels'].to(self.device) # labels to device

        outputs = self(pixel_values=image, labels=labels) #model predict

        #loss = self.loss_fn(logits_mask, mask) # custom loss
        loss = outputs.loss  # CROSS ENTROPY LOSS

        # segformer does the predict in shape/8, so it's necessary to upsample to label shape
        upsampled_logits = nn.functional.interpolate(
            outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)


        pred_mask = torch.argmax(upsampled_logits, dim=1) # argmax in N_class dimension 

        iou_score = torchmetrics.functional.iou(pred_mask, labels.long()) # iou metric in all predictions

        metrics = {'mIoU': iou_score}
        metrics.update({'loss': loss})
        return metrics

    def shared_epoch_end(self, output_list, stage):
        # gathering all metrics for batches and get the Average
        loss = torch.tensor([x['loss'] for x in output_list]).mean().item()
        miou_score = torch.tensor([x['mIoU'] for x in output_list]).mean().item()

        metrics = {
            stage+'_loss': loss,
            stage+'_mIoU': miou_score,
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



# test with iou for each class

# def iou(pred, target):
#     metrics = {}
#     # LEISHMANIA
#     y_pred = pred[:, 1, :, :]
#     y_true = target == 1

#     metrics['leishmania'] = torchmetrics.functional.iou(y_pred, y_true)

#     # macrofago_contavel
#     y_pred = pred[:, 2, :, :]
#     y_true = target == 2

#     metrics['macrofago_contavel'] = torchmetrics.functional.iou(y_pred, y_true)

#     # macrofago_nao_contavel
#     y_pred = pred[:, 3, :, :]
#     y_true = target == 3

#     metrics['macrofago_nao_contavel'] = torchmetrics.functional.iou(
#         y_pred, y_true)

#     # total_preds
#     y_pred = pred.argmax(dim=1)
#     y_true = target
#     metrics['segmentation'] = torchmetrics.functional.iou(y_pred, y_true)

#     return metrics