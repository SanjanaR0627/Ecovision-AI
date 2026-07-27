from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(
    title="EcoVision AI API",
    description="AI Waste Classification API",
    version="1.0"
)

# Load model
model = tf.keras.models.load_model("waste_classifier.keras")

class_names = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass"
]

@app.get("/")
def home():
    return {
        "message": "Welcome to EcoVision AI API",
        "status": "Running"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((224, 224))

    img = np.array(image, dtype=np.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0]

    index = np.argmax(prediction)

    return JSONResponse({
        "prediction": class_names[index],
        "confidence": round(float(prediction[index] * 100), 2)
    })