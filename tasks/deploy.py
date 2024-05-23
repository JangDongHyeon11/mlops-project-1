import os
import shutil
import mlflow
import requests
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Dict, Union, Tuple, Any
from prefect import task, get_run_logger, variables
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer, GlobalAveragePooling2D
from .utils.tf_data_utils import build_data_pipeline

PREFECT_PORT = os.getenv('PREFECT_PORT', '4200')
PREFECT_API_URL = os.getenv('PREFECT_API_URL',f'http://prefect:{PREFECT_PORT}/api')

@task(name='deploy_prefect_flow', log_prints=True)
def deploy_prefect_flow(git_repo_root: str, deploy_name: str):
    subprocess.run([f"cd {git_repo_root} && prefect --no-prompt deploy --name {deploy_name}"],
                    shell=True)

@task(name='create_or_update_prefect_vars')
def create_or_update_prefect_vars(kv_vars: Dict[str, Any]):
    logger = get_run_logger()
    for var_name, var_value in kv_vars.items():
        headers = {'Content-type': 'application/json'}
        body = {
                "name": var_name,
                "value": var_value
                }
        current_value = variables.get(var_name)
        if current_value is None:
            # create if not exist
            logger.info(f"Creating a new variable: {var_name}={var_value}")
            url = f'{PREFECT_API_URL}/variables'
            res = requests.post(url, json=body, headers=headers)
            if not str(res.status_code).startswith('2'):
                logger.error(f'Failed to create a Prefect variable, POST return {res.status_code}')
            logger.info(f'status code: {res.status_code}')
            
        else:
            # update if already existed
            logger.info(f"The variable '{var_name}' has already existed, updating the value with '{var_value}'")
            url = f'{PREFECT_API_URL}/variables/name/{var_name}'
            res = requests.patch(url, json=body, headers=headers)
            if not str(res.status_code).startswith('2'):
                logger.error(f'Failed to create a Prefect variable, PATCH return {res.status_code}')
            logger.info(f'status code: {res.status_code}')

@task(name='put_model_to_service')
def put_model_to_service(model_metadata_file_name: str, service_host: str='nginx',
                        service_port: str='80'):
    logger = get_run_logger()
    endpoint = f'http://{service_host}:{service_port}/update_model/{model_metadata_file_name}'
    res = requests.put(endpoint)
    if res.status_code == 200:
        logger.info("PUT model to the service successfully")
    else:
        logger.error("PUT model failed")
        raise Exception(f"Failed to put model to {endpoint}")

@task(name='save_and_upload_ref_data')
def save_and_upload_ref_data(ref_data_df: pd.DataFrame, remote_dir: str, 
                             model_cfg: Dict[str, Union[str, List[str], List[int]]]):
    logger = get_run_logger()
    save_file_name = model_cfg['model_name'] + model_cfg['drift_detection']['reference_data_suffix'] + '.parquet'
    save_file_path = os.path.join(model_cfg['save_dir'], save_file_name)
    if not os.path.exists(model_cfg['save_dir']):
        logger.info(f"save_dir {model_cfg['save_dir']} does not exist. Created.")
        os.makedirs(model_cfg['save_dir'])
    ref_data_df.to_parquet(save_file_path)
    logger.info(f'Saved ref_data in {save_file_path}')
    
    mlflow.log_artifact(save_file_path)


    # 파일을 클라우드 스토리지 서비스에 업로드
    shutil.copy2(save_file_path, remote_dir)
    logger.info(f'Uploaded {save_file_name} file to {remote_dir}')


@task(name='build_ref_data')
def build_ref_data(uae_model: tf.keras.models.Model, bbsd_model: tf.keras.models.Model, 
                   annotation_df: pd.DataFrame, n_sample: int, classes: List[str], 
                   img_size: List[int], batch_size: int):
    logger = get_run_logger()
    train_ds = build_data_pipeline(annotation_df, classes, 'train', img_size, batch_size, 
                                   do_augment=False, augmenter=None)

    sampled_train_ds = train_ds.take(n_sample)
    logger.info('Getting ground truths and extracting features')
    y_true_bin = np.concatenate([y for _, y in sampled_train_ds], axis=0)
    uae_feats = uae_model.predict(sampled_train_ds)
    bbsd_feats = bbsd_model.predict(sampled_train_ds)
    data = {
        'uae_feats': list(uae_feats),
        'bbsd_feats': list(bbsd_feats),
        'label': list(y_true_bin)
    }
    ref_data_df = pd.DataFrame(data)
    return ref_data_df

@task(name='save_and_upload_drift_detectors')
def save_and_upload_drift_detectors(uae_model: tf.keras.models.Model, bbsd_model: tf.keras.models.Model, remote_dir: str,
                                   model_cfg: Dict[str, Union[str, List[str], List[int]]]):
    logger = get_run_logger()
    uae_model_dir = os.path.join(model_cfg['save_dir'], model_cfg['model_name'] + model_cfg['drift_detection']['uae_model_suffix'])
    if not os.path.exists(model_cfg['save_dir']):
        logger.info(f"save_dir {model_cfg['save_dir']} does not exist. Created.")
        os.makedirs(model_cfg['save_dir'])
    uae_model.save(uae_model_dir)
    logger.info(f"Untrained AutoEncoder (UAE) model for {model_cfg['model_name']} is saved to {uae_model_dir}")

    bbsd_model_dir = os.path.join(model_cfg['save_dir'], model_cfg['model_name'] + model_cfg['drift_detection']['bbsd_model_suffix'])
    if not os.path.exists(model_cfg['save_dir']):
        logger.info(f"save_dir {model_cfg['save_dir']} does not exist. Created.")
        os.path.makedirs(model_cfg['save_dir'])
    bbsd_model.save(bbsd_model_dir)
    logger.info(f"Black-Box Shift Detector (BBSD) model for {model_cfg['model_name']} is saved to {bbsd_model_dir}")
    
    # upload to mlflow
    mlflow.log_artifact(uae_model_dir)
    mlflow.log_artifact(bbsd_model_dir)

    # 모델 파일을 클라우드 스토리지 서비스에 업로드
    # uae
    uae_model_upload_dir = os.path.join(remote_dir, model_cfg['model_name'] + model_cfg['drift_detection']['uae_model_suffix'])
    shutil.copytree(uae_model_dir, uae_model_upload_dir, dirs_exist_ok=True)
    logger.info(f'Uploaded UAE model to {uae_model_upload_dir}')
    # bbsd
    bbsd_model_upload_dir = os.path.join(remote_dir, model_cfg['model_name'] + model_cfg['drift_detection']['bbsd_model_suffix'])
    shutil.copytree(bbsd_model_dir, bbsd_model_upload_dir, dirs_exist_ok=True)
    logger.info(f'Uploaded BBSD model to {bbsd_model_upload_dir}')

@task(name='build_drift_detectors')
def build_drift_detectors(main_model: tf.keras.models.Model, model_input_size: Tuple[int, int], 
                          softmax_layer_idx: int = -1, encoding_dims: int = 32):
    uae = tf.keras.Sequential(
        [
            InputLayer(input_shape = model_input_size+(3,)),
            Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
            Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
            Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
            Flatten(),
            Dense(encoding_dims,)
        ]
    )
    bbsd = Model(inputs=main_model.inputs, outputs=[main_model.layers[softmax_layer_idx].output])

    return uae, bbsd