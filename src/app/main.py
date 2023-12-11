from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import tensorflow as tf
import uvicorn
import os
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


@app.post('/predict')
async def root(file: UploadFile = File(...)):
    if 'image' not in file.content_type:
        raise HTTPException(415, 'File harus berupa gambar!')
    try:
        image = await file.read()
        image = Image.open(BytesIO(image))
        image = image.resize((224, 224))

        x = img_to_array(image)
        x = np.expand_dims(x, axis=0)
        img = np.vstack([x])

        pred_resnet50 = resnet50.predict(img)
        pred_resnet101 = resnet101.predict(img)

        labels = os.listdir(dataset_dir)

        idx_class_resnet50 = np.argmax(pred_resnet50, axis=1)[0]
        idx_class_resnet101 = np.argmax(pred_resnet101, axis=1)[0]

        prob_class_resnet50 = pred_resnet50.tolist()[0]
        prob_class_resnet101 = pred_resnet101.tolist()[0]

        if max(prob_class_resnet50) < 0.8:
            resnet50_class = 'Unknown'
        else:
            resnet50_class = labels[idx_class_resnet50]

        if max(prob_class_resnet101) < 0.8:
            resnet101_class = 'Unknown'
        else:
            resnet101_class = labels[idx_class_resnet101]

        return {
            'ResNet50': {
                'prediction': resnet50_class,
                'probability': max(prob_class_resnet50) * 100,
            },

            'ResNet101': {
                'prediction': resnet101_class,
                'probability': max(prob_class_resnet101) * 100,
            }
        }
    except Exception as e:
        raise HTTPException(e)


@app.get('/test')
def test_function():
    model = tf.keras.models.load_model('../../models/resnet101.h5')
    image_path = os.path.join(dataset_dir, npm, filename)
    image = load_img(image_path, target_size=(128, 128))

    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)
    img = np.vstack([x])

    pred = model.predict(img)

    predicted_class = np.argmax(pred, axis=1)
    labels = {'2017051001': 0, '2017051017': 1,
              '2117051009': 2, '2117051048': 3}
    labels = dict((v, k) for k, v in labels.items())

    predictions = [labels[k] for k in predicted_class]

    return {
        'tes': predictions,
        'predict': pred.tolist()[0],
        'Class': '',
    }


if __name__ == "__main__":

    dataset_dir = '../../datasets/to_train/training'

    resnet50 = tf.keras.models.load_model('../../models/resnet50_best.h5')
    resnet101 = tf.keras.models.load_model('../../models/resnet101_best.h5')

    uvicorn.run(app, host="127.0.0.1", port=8001)
