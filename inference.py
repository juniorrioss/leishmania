import cv2
from model import SemanticModel
import os
import numpy as np
import pandas as pd

from scipy import ndimage as ndi
from skimage import (
    feature, measure, segmentation
)
from skimage.measure import find_contours
import torch
from torchvision import transforms as T
from torch import nn
from skimage.color import rgb2gray


def inference(model, image: np.ndarray, device='cpu'):
    """Prediction function

    Args:
        model (torch.Module): Model trained
        image (np.ndarray): image to predict
        device (str, optional): Use or not CUDA (cuda or cpu). Defaults to 'cpu'.

    Returns:
        Dict[torch.Tensor]: Outputs prediction with segmentation of: Contable, Non Contables, Leishmania and Argmax Segmentation
    """

    img_input = preprocessing_image(image).to(device)

    model = model.to(device)  # make sure the model is at the correct device
    output = model.model(img_input)
    logits = output.logits

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=img_input.shape[-2:],  # (height, width)
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


def prepare_model(checkpoint_path, device='cpu'):
    """Function to prepare the model in prediction way

    Args:
        checkpoint_path (str): Path to the weights of model
        device (str, optional): Use or not CUDA (cuda or cpu). Defaults to 'cpu'.

    Returns:
        torch.Module: Pytorch model trained
    """
    # LOADING THE MODEL CHECKPOINT
    print('LOADING THE MODEL CHECKPOINT')
    model = SemanticModel().load_from_checkpoint(checkpoint_path).to(device)
    print(f'Model loaded to {device}!')

    model.freeze()
    model.eval()
    return model


def generate_contable_roi(contable_segmentation):
    """Create a bounding box around each macrophage contable 

    Args:
        contable_segmentation (np.array): contable_segmentation already thresholded

    Returns:
        List[Dict[Coordinates]]: A list of a Dict contain the coordinates of bounding boxes
    """

    if isinstance(contable_segmentation, torch.Tensor):
        contable_segmentation = contable_segmentation.numpy()

    contours = find_contours(contable_segmentation)

    bounding_boxes = []
    for contour in contours:
        contour = contour.astype(int)
        Xmin = np.min(contour[:, 1])
        Xmax = np.max(contour[:, 1])
        Ymin = np.min(contour[:, 0])
        Ymax = np.max(contour[:, 0])

        # FILTRO PASSA ALTA DE BBOX (RETIRAR RUÍDOS COM COMPRIMENTO MENOR DO QUE 20 PX)
        if Xmax - Xmin > 20 and Ymax - Ymin > 20:
            bounding_boxes.append(
                {'Xmin': Xmin, 'Xmax': Xmax, 'Ymin': Ymin, 'Ymax': Ymax})

    return bounding_boxes


def leish_count_by_roi(leish_roi):
    """Implements the watershed algoritm to count and separate overlaps in leishmania

    Args:
        leish_roi (np.array): Region of Interest of Leishmania Segmentation

    Returns:
        int: Number of leishmania
    """

    if leish_roi.sum() == 0:  # IF ALL PIXELS OF ROI IS BLANK RETURN N = 0
        return 0

    distance = ndi.distance_transform_edt(leish_roi)

    local_max_coords = feature.peak_local_max(distance, min_distance=2)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = measure.label(local_max_mask)

    segmented_cells = segmentation.watershed(
        -distance, markers, mask=leish_roi)
    n_leish = segmented_cells.max()
    return n_leish


def leish_dict_per_image(contable_segmentation, leish_segmentation):
    """Wrapper function to run the counting process throught a entire image

    Args:
        contable_segmentation (np.array): Segmentation of macrophages contable already thresholded
        leish_segmentation (np.array): Segmentation of leishmania already thresholded

    Returns:
        Dict[Coordinates: N_leish]: A dict with coordinates of each macrophage and how many leishmania is in it
    """

    if isinstance(leish_segmentation, torch.Tensor):
        leish_segmentation = leish_segmentation.cpu().numpy()

    if isinstance(contable_segmentation, torch.Tensor):
        contable_segmentation = contable_segmentation.cpu().numpy()

    bboxes = generate_contable_roi(contable_segmentation)

    contable_dict = {}
    for bbox in bboxes:
        leish_roi = leish_segmentation[
            bbox['Ymin']:bbox['Ymax'], bbox['Xmin']: bbox['Xmax']]
        n_leish = leish_count_by_roi(leish_roi)
        contable_dict[(bbox['Xmin'], bbox['Ymin'])] = n_leish

    return contable_dict


def remove_bg(image, mask=None):
    """Recebe a imagem colorida e a mascara gerada da imagem e retira o fundo preto nas duas imagens
       E retorna o ROI da imagem e da mascara

    Args:
        imagem_colorida ([img]): [raw image]
        mask ([img]): [mask from raw image]

    Returns:
        np.array OR [list[np.array]]: roi_img OR [roi_img, roi_mask]
    """
    image_gray = rgb2gray(image)
    contours = find_contours(image_gray)

    for contour in contours:
        contour = contour.astype(int)
        Xmin = np.min(contour[:, 1])
        Xmax = np.max(contour[:, 1])
        Ymin = np.min(contour[:, 0])
        Ymax = np.max(contour[:, 0])

        if Xmax - Xmin > 500 and Ymax - Ymin > 500:
            roi_img = image[Ymin:Ymax, Xmin: Xmax]
            if mask:
                roi_mask = mask[Ymin:Ymax, Xmin:Xmax]

    if mask:
        return roi_img, roi_mask
    else:
        return roi_img


def preprocessing_image(image, remove_background=True):
    """Function to do all preprocessing steps to do inference

    Args:
        image (np.array): Image Matrix
        remove_background (bool, optional): Remove the background throught remove_bg function. Defaults to True.

    Returns:
        np.array image: Roi image with the minimal background
    """

    if remove_background:
        image = remove_bg(image)

    image = cv2.resize(image, (768, 768))

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    t = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    img = t(image)

    img_input = img.unsqueeze(0)

    return img_input


# def leish_count(cells, pred_contavel=None, verbose=False):
#     if pred_contavel is not None:
#         contours, hierarchy = cv2.findContours(image=pred_contavel.astype(
#             np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
#         mask_c = np.zeros(cells.shape)
#         cv2.drawContours(image=mask_c, contours=contours, contourIdx=-1,
#                          color=(1), thickness=-1, lineType=cv2.LINE_AA)

#         cells = np.logical_and(cells, mask_c)
#     distance = ndi.distance_transform_edt(cells)

#     local_max_coords = feature.peak_local_max(distance, min_distance=5)
#     local_max_mask = np.zeros(distance.shape, dtype=bool)
#     local_max_mask[tuple(local_max_coords.T)] = True
#     markers = measure.label(local_max_mask)

#     segmented_cells = segmentation.watershed(-distance, markers, mask=cells)
#     plt.figure(figsize=(15, 15))
#     plt.imshow(color.label2rgb(segmented_cells, bg_label=0))
#     n_leish = segmented_cells.max()
#     if pred_contavel is not None:
#         plt.imshow(pred_contavel, alpha=.2, cmap='inferno')
#     plt.title(f'Segmented leishmanias countable {n_leish}')
#     plt.axis('off')

#     if verbose:
#         plt.show()

#     return n_leish
