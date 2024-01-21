from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import tensorflow as tf
import uvicorn
import os
import cv2
import numpy as np
from io import BytesIO
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# labels = {
#     '2017051001': 0,
#     '2017051017': 1,
#     '2117051009': 2,
#     '2117051019': 3,
#     '2117051027': 4,
#     '2117051043': 5,
#     '2117051048': 6,
#     '2117051050': 7,
#     '2117051095': 8,
#     '2157051001': 9,
#     '2157051006': 10
# }


@app.post('/api/v1/predict')
async def root(file: UploadFile = File(...)):
    if 'image' not in file.content_type:
        raise HTTPException(415, 'File harus berupa gambar!')
    try:
        image = await file.read()
        image = Image.open(BytesIO(image))
        image = image.resize((224, 224))

        x = img_to_array(image)
        x = x[:, :, :3]
        x = tf.keras.applications.resnet.preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        img = np.vstack([x])

        pred_resnet50 = resnet50.predict(img)
        pred_resnet101 = resnet101.predict(img)

        labels = os.listdir(dataset_dir)

        idx_class_resnet50 = np.argmax(pred_resnet50, axis=1)[0]
        idx_class_resnet101 = np.argmax(pred_resnet101, axis=1)[0]

        prob_class_resnet50 = pred_resnet50.tolist()[0]
        prob_class_resnet101 = pred_resnet101.tolist()[0]
        
        max_prob_class_resnet_50 = max(prob_class_resnet50)
        max_prob_class_resnet_101 = max(prob_class_resnet101)

        if max_prob_class_resnet_50 < 0.4:
            resnet50_class = 'Unknown'
        else:
            resnet50_class = labels[idx_class_resnet50]

        if max_prob_class_resnet_101 < 0.4:
            resnet101_class = 'Unknown'
        else:
            resnet101_class = labels[idx_class_resnet101]

        if max_prob_class_resnet_101 > max_prob_class_resnet_50:
            prediction = resnet101_class
            prob = max_prob_class_resnet_101
        else:
            prediction = resnet50_class
            prob = max_prob_class_resnet_50

        return {
            'prediction': prediction,
            'probability': prob * 100,
        }
    except Exception as exception:
        raise HTTPException(500, exception)


if __name__ == "__main__":

    dataset_dir = '../../datasets/to_train/training'

    resnet50 = tf.keras.models.load_model(
        '../../models/resnet50_best_model.h5')
    resnet101 = tf.keras.models.load_model(
        '../../models/resnet101_best_model.h5')

    uvicorn.run(app, host="127.0.0.1", port=8001)
