import os

"""
Define common constants
"""

TABLE_FETCH_QUERY:str="""
SELECT *
FROM `du_stats`.`silver`.`synth_histo_table`
"""

ARTIFACT_DIR:str='artifacts'
RAW_DATA_FILENAME:str='dustats_histo.csv'
TRAIN_FILENAME:str='train.csv'
TEST_FILENAME:str='test.csv'
SCHEMA_FILEPATH:str=os.path.join('data_schema', 'schema.yaml')

"""
Define DUStats Ingestion Constants
"""
DUSTATS_INGESTION_RAW_DATA_DIR:str='raw_data'
DUSTATS_INGESTION_SPLIT_DATA_DIR:str='split_data'
DUSTATS_INGESTION_TRAIN_TEST_SPLIT_RATIO:float=0.2