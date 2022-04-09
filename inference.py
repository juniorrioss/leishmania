from torchvision import transforms as T
import cv2
from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage import (
    color, feature, measure, segmentation
)


def inference(model, image: np.ndarray):

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    t = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    img = t(image)

    img_input = img.unsqueeze(0)

    model.freeze()
    model.eval()
    output = model.model(img_input)
    logits = output.logits

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=img.shape[-2:],  # (height, width)
        mode='bilinear',
        align_corners=False
    )

    leishmania_preds = torch.sigmoid(upsampled_logits[0][1])
    contavel_preds = torch.sigmoid(upsampled_logits[0][2])
    nao_contavel_preds = torch.sigmoid(upsampled_logits[0][3])
    seg = upsampled_logits.argmax(dim=1)[0]

    return {
        'leishmania': leishmania_preds,
        'macrofago nao contavel': nao_contavel_preds,
        'macrofago contavel': contavel_preds,
        'segmentation': seg
    }


def leish_count(cells, pred_contavel=None, verbose=False):
    if pred_contavel is not None:
        contours, hierarchy = cv2.findContours(image=pred_contavel.astype(
            np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        mask_c = np.zeros(cells.shape)
        cv2.drawContours(image=mask_c, contours=contours, contourIdx=-1,
                         color=(1), thickness=-1, lineType=cv2.LINE_AA)

        cells = np.logical_and(cells, mask_c)
    distance = ndi.distance_transform_edt(cells)

    local_max_coords = feature.peak_local_max(distance, min_distance=5)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = measure.label(local_max_mask)

    segmented_cells = segmentation.watershed(-distance, markers, mask=cells)
    plt.figure(figsize=(15, 15))
    plt.imshow(color.label2rgb(segmented_cells, bg_label=0))
    n_leish = segmented_cells.max()
    if pred_contavel is not None:
        plt.imshow(pred_contavel, alpha=.2, cmap='inferno')
    plt.title(f'Segmented leishmanias countable {n_leish}')
    plt.axis('off')

    if verbose:
        plt.show()

    return n_leish
