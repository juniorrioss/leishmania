"""
  JSON TO MASK 3 class
  Multiclass -> background/leishmania/macrofago contavel/macrofago n contavel
                  0           1         2                      3                
"""

import zipfile
import os
import cv2
import json
import numpy as np
import gdown
import os

# função para pegar os pontos dos poligonos da label do arquivo json


def get_json(ann_path):

    with open(ann_path, 'rb') as handle:
        data = json.load(handle)

    shape_dicts = data['shapes']
    img_height = data['imageHeight']
    img_width = data['imageWidth']

    return shape_dicts, (img_height, img_width)


def label2poly_multiclass(shape, label2poly):

    blank_channel = np.zeros(shape, dtype=np.uint8)

    # Preenchendo os poligonos com suas respectivas classes
    for label in (label2poly):
        if label[0] in ['leishmania']:
            cv2.fillPoly(blank_channel, [label[1]], 1)  # LEISHMANIA
            # cv2.polylines(blank_channel, [label[1]], True, 2, thickness=2) # EDGE

        if label[0] in ['macrofago contavel']:
            cv2.fillPoly(blank_channel, [label[1]], 2)  # MACROFAGO CONTAVEL

        if label[0] in ['macrofago nao contavel']:
            # MACROFAGO NAO CONTAVEL
            cv2.fillPoly(blank_channel, [label[1]], 3)

    return blank_channel


def create_mask(shape, shape_dicts, mode):

    # coletando todas as labels
    labels = [x['label'] for x in shape_dicts]

    # coletando todos os poligonos
    poly = [np.array(x['points'], dtype=np.int32) for x in shape_dicts]

    label2poly = []

    # juntando as labels e seus poligonos
    # EX: ('macrofago nao contavel', array([[2919, 2320],
    #    [2826, 2306],
    #    [2789, 2223],
    #    [2817, 2127],
    #    [2893, 2081],
    #    [2956, 2109]]))
    for i in range(len(labels)):
        label2poly.append((labels[i], poly[i]))

    mask = label2poly_multiclass(shape, label2poly)

    return mask


# DOWNLOAD THE LEISHMANIA DATA
# url = 'https://drive.google.com/uc?id=10Z08dl7irHLGv004fP_Yv4IQEUHqp0Hz' v1
url = 'https://drive.google.com/file/d/1_XayeJBsPOJKzNM-wYnmQuhKe3Tfu2m4/view?usp=sharing'  # v2
gdown.download(url)
with zipfile.ZipFile('croped_dataset_v2.zip', 'r') as zip_ref:
    zip_ref.extractall('./')


mode = 'multiclass'
save_dir = 'mask_v2'
json_path = 'croped_dataset_v2'

print('-'*15)
print('Converting json to mask')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

json_list = [i for i in os.listdir(json_path) if i.endswith('.json')]
for json_name in json_list:
    shape_dicts, shape_img = get_json(os.path.join(json_path, json_name))
    mask = create_mask(shape_img, shape_dicts, mode)

    img_name = json_name.split('.')[0] + '.png'
    cv2.imwrite(os.path.join(save_dir, img_name), mask)


print('-'*15)
print('Conversion Ended')

os.remove("croped_dataset_v2.zip")
