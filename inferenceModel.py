import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    # Loads logs for CHoiCE Model
    #configs = BaseModelConfigs.load("Capstone/202404230507/configs.yaml")

    #Loads Logs for IAM Model 
    configs = BaseModelConfigs.load("C:Capstone/20240423_Model1/configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    #Tests model against different data sets
    df = pd.read_csv("C:Capstone/IAM_Val.csv").values.tolist()
    #df = pd.read_csv("C:Capstone/betts_et_al.csv").values.tolist()
    #df = pd.read_csv("C:Capstone/Betts_et_al_letter.csv").values.tolist()

    #command to tensorboard for models
    #tensorboard --logdir Capstone\202404230507\logs
    #tensorboard --logdir Capstone\20240423_Model1\logs
    

    #Displays image being read from the file documents
    #Press 0 to move to the next image
    accum_cer = []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path)
        print(image)

        prediction_text = model.predict(image)

        cer = get_cer(prediction_text, label)
        print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

        cv2.imshow(label, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        accum_cer.append(cer)

    print(f"Average CER: {np.average(accum_cer)}")