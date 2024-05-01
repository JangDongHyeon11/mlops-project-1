import os
import cv2
import base64
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import Optional
from fastapi import FastAPI, Request, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict
from utils import (GradCAM, tf_load_model, array_to_encoded_str, process_heatmap, 
                   prepare_db, load_drift_detectors, commit_results_to_db, 
                   commit_only_api_log_to_db, check_db_healthy)


# define Pydantic models for type validation
class Message(BaseModel):
    message: str

class PredictionResult(BaseModel):
    model_name: str
    prediction: Dict[str,float]
    overlaid_img: str
    raw_hm_img: str
    message: str

FORMAT = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)-3d | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
print(f'Created logger with name {__name__}')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(FORMAT)
logger.addHandler(ch)

app = FastAPI()

# init model to None
model: tf.keras.models.Model = None
model_meta = None

# init drift detector models to None too
uae: tf.keras.models.Model = None
bbsd: tf.keras.models.Model = None

# prepare database
prepare_db()

@app.get("/health_check", response_model=Message, responses={404: {"model": Message}})
def health_check(request: Request):
    resp_code = 200
    resp_message = "Service is ready and healthy."
    try:
        check_db_healthy()
    except:
        resp_code = 404
        resp_message = "DB is not functional. Service is unhealthy."
    return JSONResponse(status_code=resp_code, content={"message": resp_message})


@app.put("/update_model/{model_metadata_file_path}", response_model=Message, responses={404: {"model": Message}})
def update_model(request: Request, model_metadata_file_path: str, background_tasks: BackgroundTasks):
    global model
    global model_meta
    global uae
    global bbsd
    start_time = time.time()
    logger.info('Updating model')
    try:
        # prepare drift detectors along with the model here
        model, model_meta = tf_load_model(model_metadata_file_path)
        uae, bbsd = load_drift_detectors(model_metadata_file_path)
    except Exception as e:
        logger.error(f'Loading model failed with exception:\n {e}')
        time_spent = round(time.time() - start_time, 4)
        resp_code = 404
        resp_message = f"Updating model failed due to failure in model loading method with path parameter: {model_metadata_file_path}"
        background_tasks.add_task(commit_only_api_log_to_db, request, resp_code, resp_message, time_spent)
        return JSONResponse(status_code=resp_code, content={"message": resp_message})
    
    time_spent = round(time.time() - start_time, 4)
    resp_code = 200
    resp_message = "Update the model successfully"
    background_tasks.add_task(commit_only_api_log_to_db, request, resp_code, resp_message, time_spent)
    return {"message": resp_message}




    
