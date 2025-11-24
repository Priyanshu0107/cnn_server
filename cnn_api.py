from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

model = tf.keras.models.load_model("cnn_model.h5")
CLASS_NAMES = ["electrical", "road", "water"]

@app.post("/cnn")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224))

        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, 0)

        pred = model.predict(arr)
        idx = int(np.argmax(pred))
        confidence = float(np.max(pred))

        return {
            "class": CLASS_NAMES[idx],
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9001)
