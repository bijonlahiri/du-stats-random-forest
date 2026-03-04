from databricks.sql import connect
from du_stats.exception.exception import DUStatsException
from du_stats.logging.logger import logging
import pandas as pd
import numpy as np
import torch
import os, sys
from tqdm import tqdm
from dotenv import load_dotenv
import time, yaml, pickle
from du_stats.constants.dustats_pipeline import SLEEP_TIME, SCHEMA_FILEPATH

def load_object_from_file(filepath:str)->object:
    try:
        if os.path.exists(filepath):
            with open(filepath, 'rb') as file:
                return pickle.load(file)
        else:
            logging.info(f'Model does not exist in filepath: {filepath}')
            return None
    except Exception as e:
        raise DUStatsException(e, sys)

def save_object_to_file(filepath:str, obj:object)->bool:
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as file:
            pickle.dump(obj, file)
        return True
    except Exception as e:
        raise DUStatsException(e, sys)

def load_numpy_array_from_file(filepath:str)->np.array:
    try:
        if os.path.exists(filepath):
            with open(filepath, 'rb') as file:
                return np.load(file)
        else:
            logging.info(f'Numpy array file does not exist in path: {filepath}')
            return None
    except Exception as e:
        raise DUStatsException(e, sys)

def save_numpy_array_to_file(filepath:str, data_arr:np.array)->bool:
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as file:
            np.save(file, data_arr)
        return True
    except Exception as e:
        raise DUStatsException(e, sys)

def save_yaml(filepath:str, obj:object)->bool:
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as yaml_file:
            yaml.dump(obj, yaml_file)
        return True
    except Exception as e:
        raise DUStatsException(e, sys)

def read_yaml(filepath:str)->dict:
    try:
        if os.path.exists(filepath):
            logging.info('Parsing yaml file')
            with open(filepath, 'r') as yaml_file:
                return yaml.safe_load(yaml_file)
        else:
            logging.info('Yaml file does not exist.')
            return None
    except Exception as e:
        raise DUStatsException(e, sys)

def read_dataframe_from_file(filepath:str)->pd.DataFrame:
    try:
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            logging.info('Dataframe filepath does not exist.')
            return pd.DataFrame()
    except Exception as e:
        raise DUStatsException(e, sys)

def save_dataframe_to_file(filepath:str, dataframe:pd.DataFrame)->bool:
    try:
        file_dir = os.path.dirname(filepath)
        try:
            os.makedirs(file_dir, exist_ok=True)
            dataframe.to_csv(filepath, index=False, header=True)
            return True
        
        except Exception as e:
            logging.info(f'Could not save dataframe to filepath: {filepath}\nError:{e}')
            return False
        
    except Exception as e:
        raise DUStatsException(e, sys)

def check_env()->bool:
    try:
        if os.getenv('DATABRICKS_SERVER_HOSTNAME'):
            return True
        
        return False
    
    except Exception as e:
        raise DUStatsException(e, sys)

def fetch_data(fetch_query:str)->pd.DataFrame:
    try:
        schema=read_yaml(SCHEMA_FILEPATH)
        columns = [list(col.keys())[0] for col in schema['columns']]
        load_dotenv()
        if check_env():
            logging.info('Environment successfully loaded')
            for i in range(5):
                try:
                    rows = []
                    batch_size = 1000
                    with connect(
                        server_hostname=os.getenv('DATABRICKS_SERVER_HOSTNAME'),
                        http_path=os.getenv('DATABRICKS_HTTP_PATH'),
                        access_token=os.getenv('DATABRICKS_ACCESS_TOKEN')
                    ) as connection:
                        with connection.cursor() as cursor:
                            cursor.execute('SELECT COUNT(*) FROM `du_stats`.`silver`.`synth_histo_table`')
                            total_rows = cursor.fetchone()[0]
                            cursor.execute(fetch_query)
                            with tqdm(total=total_rows, desc='Downloading data') as pbar:
                                while True:
                                    batch = cursor.fetchmany(batch_size)
                                    if not batch:
                                        break
                                    rows.extend(batch)
                                    pbar.update(len(batch))
                    df = pd.DataFrame(data=rows, columns=columns)
                    return df

                except Exception as e:
                    logging.info(f'Error while fetching data: {e}')
                    df = pd.DataFrame()
                    return df
                time.sleep(SLEEP_TIME*np.exp(i/2.2))
        
        else:
            logging.info('Environment does not exist')
            df = pd.DataFrame()
            return df
        
    except Exception as e:
        raise DUStatsException(e, sys)

def save_tensor_artifact(filepath:str, X:torch.Tensor, y:torch.Tensor)->None:
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save([X, y], filepath)
    except Exception as e:
        raise DUStatsException(e, sys)

def load_tensor_artifact(filepath:str)->tuple:
    try:
        X, y = torch.load(filepath)
        return X, y
    except Exception as e:
        raise DUStatsException(e, sys)

def save_model_state_dict(filepath:str, model_state_dict:dict):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(model_state_dict, filepath)
    except Exception as e:
        raise DUStatsException(e, sys)

def load_model_state_dict(filepath:str)->dict:
    try:
        if os.path.exists(filepath):
            state_dict = torch.load(filepath, weights_only=True)
            return state_dict
        else:
            logging.info(f'No model found in path: {filepath}')
            return None
    except Exception as e:
        raise DUStatsException(e, sys)