import os

"""
Define common constants
"""
TARGET_COLUMN:str='rca_label'
TABLE_FETCH_QUERY:str="""
SELECT *
FROM `du_stats`.`silver`.`synth_histo_table`
--WHERE log_date=DATE('2026-01-01') AND ueid=17017
"""

ARTIFACT_DIR:str='artifacts'
RAW_DATA_FILENAME:str='dustats_histo.csv'
TRAIN_FILENAME:str='train.csv'
TEST_FILENAME:str='test.csv'
SCHEMA_FILEPATH:str=os.path.join('data_schema', 'schema.yaml')
SLEEP_TIME:float=10

"""
Define DUStats Ingestion Constants
"""
DUSTATS_INGESTION_DIR:str='ingestion'
DUSTATS_INGESTION_RAW_DATA_DIR:str='raw_data'
DUSTATS_INGESTION_SPLIT_DATA_DIR:str='split_data'
DUSTATS_INGESTION_TRAIN_TEST_SPLIT_RATIO:float=0.2
DUSTATS_INGESTION_TRAIN_TEST_SPLIT_RANDOM_STATE:int=10

"""
Define DUStats Validation Constants
"""
DUSTATS_VALIDATION_DIR:str='validation'
DUSTSTS_VALIDATION_REPORT_DIR:str='validation_report'
DUSTATS_VALIDATION_REPORT_FILENAME:str='report.yaml'
DUSTATS_VALIDATION_VALID_DIR:str='validated'
DUSTATS_VALIDATION_INVALID_DIR:str='invalid'
DUSTATS_VALIDATION_DATA_DRIFT_THRESHOLD:float=0.05

"""
Define DUStats Transformation Constants
"""
DUSTATS_TRANSFORMATION_DIR:str='transformation'
DUSTATS_TRANSFORMATION_DATA_DIR:str='transformed_data'
DUSTATS_TRANSFORMATION_MODEL_DIR:str='transformer_model'
DUSTATS_TRANSFORMATION_MODEL_FILENAME:str='preprocessor.pkl'

"""
Define DUStats Model Trainer Constants
"""
DUSTATS_MODEL_TRAINER_DIR:str='model_trainer'
DUSTATS_MODEL_TRAINER_MODEL_REPORT_DIR:str='model_report'
DUSTATS_MODEL_TRAINER_MODEL_REPORT_FILENAME:str='model_report.yaml'
DUSTATS_MODEL_TRAINER_MODEL_DIR:str='model'
DUSTATS_MODEL_TRAINER_MODEL_FILENAME:str='model.pkl'