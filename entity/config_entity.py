import os
from datetime import datetime
from constants import dustats_pipeline


class DUStatsPipelineConfig:

    def __init__(self, timestamp=datetime.now()):
        self.timestamp = timestamp
        self.artifact_dir=os.path.join(dustats_pipeline.ARTIFACT_DIR, self.timestamp)

class DUStatsIngestionConfig:

    def __init__(self, dustats_pipeline_config:DUStatsPipelineConfig):
        self.artifact_dir = dustats_pipeline_config.artifact_dir
        self.raw_data_filepath = os.path.join(
            self.artifact_dir,
            dustats_pipeline.DUSTATS_INGESTION_RAW_DATA_DIR,
            dustats_pipeline.RAW_DATA_FILENAME
        )
        self.train_data_filepath = os.path.join(
            dustats_pipeline_config.artifact_dir,
            dustats_pipeline.DUSTATS_INGESTION_SPLIT_DATA_DIR,
            dustats_pipeline.TRAIN_FILENAME
        )
        self.test_data_filepath = os.path.join(
            dustats_pipeline_config.artifact_dir,
            dustats_pipeline.DUSTATS_INGESTION_SPLIT_DATA_DIR,
            dustats_pipeline.TEST_FILENAME
        )
        self.fetch_query = dustats_pipeline.TABLE_FETCH_QUERY
        self.train_test_split_ratio = dustats_pipeline.DUSTATS_INGESTION_TRAIN_TEST_SPLIT_RATIO