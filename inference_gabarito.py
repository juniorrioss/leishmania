import cv2
from model import SemanticModel
from inference import inference, leish_count
import os
import numpy as np
import pandas as pd

if __name__ == '__main__':

    PATH = r'GABARITO\013 16 1'
    THRESHOLD_LEISHMANIA = 0.3
    THRESHOLD_MACROFAGO_CONTAVEL = 0.4
    CHECKPOINT_PATH = 'lightning_logs/version_0/checkpoints/epoch=18-step=3096.ckpt'

    FOLDER_LIST = os.listdir(PATH)

    # LOADING THE MODEL CHECKPOINT
    print('LOADING THE MODEL CHECKPOINT')
    model = SemanticModel().load_from_checkpoint(CHECKPOINT_PATH)
    print('Model loaded!')

    results_contagem = {}
    for folder in FOLDER_LIST:
        image_list = os.listdir(os.path.join(PATH, folder))
        for image in image_list:
            
            img = cv2.imread(os.path.join(PATH, folder, image))
            assert isinstance(img, np.ndarray), "Error in loading Image"
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (768, 768))

            print('Realizando inferencia na imagem: ', image)
            outputs = inference(model=model, image=img)

            preds_leish = (outputs['leishmania'] >
                           THRESHOLD_LEISHMANIA).numpy()
            preds_contavel = (outputs['macrofago contavel'] >
                              THRESHOLD_MACROFAGO_CONTAVEL).numpy()

            contagem = leish_count(preds_leish, preds_contavel, verbose=False)
            results_contagem[image] = contagem

            print('Contagem de leishmanias PREDITAS->', contagem)

        results_dataframe = pd.DataFrame.from_dict(
            results_contagem, orient='index', columns=['Quantidade de leishmania'])
        results_dataframe.to_excel('013_16_1_'+folder+'.xlsx')
