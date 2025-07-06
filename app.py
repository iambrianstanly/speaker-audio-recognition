from fastapi import FastAPI, UploadFile
import tensorflow as tf
import numpy as np
import joblib
import os
from src.feature_engineering.feature_extraction import extract_mfcc

model = tf.keras.models.load_model("models/best_model")

app = FastAPI()


def pipeline(path):
    class_names = ["Benjamin", "Jens","Julia","Margaret","Nelson"]

    mfccs = (extract_mfcc(path).T).astype(np.float32)
    f = open("models/standard_scaler.joblib", "rb")
    scaler = joblib.load(f)
    mfccs_norm = scaler.transform(mfccs)
    mfccs_norm = np.expand_dims(mfccs_norm, axis=0)
    y_pred = model.predict(mfccs_norm)
    idx = np.argmax(y_pred)
    class_name = class_names[idx]
    f.close()
    # return {"predicted Name": class_name}
    return class_name


@app.get("/")
def welcome():
    return {"message": "Welcome!"}


@app.post("/predict")
async def predict(file: UploadFile):
    path = os.path.join("test", file.filename)
    class_name = pipeline(path)

    return {"message": class_name}


