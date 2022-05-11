from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI(
    title="Animal Biometric API",
    description="### Developed by Anand Kumar 18BCE2225 and Vanni Tripathi 18BCE0693"
)

MODEL_3_CLASS = tf.keras.models.load_model('./saved_models/5_3_class')

MODEL_ALL_CLASS = tf.keras.models.load_model('./saved_models/4_all_class_final')

CLASS_NAMES = ["Critically endangered", "Endangered", "Least Concern"]

CLASS_NAME_6_CLASS = ["Amur Leopard", "Rhino", "asian_elephant", "cats", "dogs", "tiger"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive, V!!"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image




@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_without_alpha = image[:256, :256, :3]
    img_batch = np.expand_dims(image_without_alpha, 0)

    predictions_3_class = MODEL_3_CLASS.predict(img_batch)
    predicttions_all_class = MODEL_ALL_CLASS.predict(img_batch)

    predicted_3_class = CLASS_NAMES[np.argmax(predictions_3_class[0])]
    predicted_all_class = CLASS_NAME_6_CLASS[np.argmax(predicttions_all_class[0])]
    confidence = np.max(predicttions_all_class[0])
    return {
        'Specie': predicted_all_class,
        'Conservation Status': predicted_3_class,
        'Confidence (%)': float(confidence)*100
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
