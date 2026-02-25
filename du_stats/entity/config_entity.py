import os, sys
from datetime import datetime
from du_stats.constants import dustats_pipeline
from du_stats.exception.exception import DUStatsException


class DUStatsPipelineConfig:

    def __init__(self, timestamp=datetime.now()):
        try:
            self.timestamp = timestamp.strftime('%d_%m_%y_%H_%M_%S')
            self.artifact_dir=os.path.join(dustats_pipeline.ARTIFACT_DIR, self.timestamp)
        except Exception as e:
            raise DUStatsException(e, sys)

class DUStatsIngestionConfig:

    def __init__(self, dustats_pipeline_config:DUStatsPipelineConfig):
        try:
            self.artifact_dir = dustats_pipeline_config.artifact_dir
            self.raw_data_filepath = os.path.join(
                self.artifact_dir,
                dustats_pipeline.DUSTATS_INGESTION_DIR,
                dustats_pipeline.DUSTATS_INGESTION_RAW_DATA_DIR,
                dustats_pipeline.RAW_DATA_FILENAME
            )
            self.train_data_filepath = os.path.join(
                dustats_pipeline_config.artifact_dir,
                dustats_pipeline.DUSTATS_INGESTION_DIR,
                dustats_pipeline.DUSTATS_INGESTION_SPLIT_DATA_DIR,
                dustats_pipeline.TRAIN_FILENAME
            )
            self.test_data_filepath = os.path.join(
                dustats_pipeline_config.artifact_dir,
                dustats_pipeline.DUSTATS_INGESTION_DIR,
                dustats_pipeline.DUSTATS_INGESTION_SPLIT_DATA_DIR,
                dustats_pipeline.TEST_FILENAME
            )
            self.fetch_query = dustats_pipeline.TABLE_FETCH_QUERY
            self.train_test_split_ratio = dustats_pipeline.DUSTATS_INGESTION_TRAIN_TEST_SPLIT_RATIO
            self.train_test_split_random_state = dustats_pipeline.DUSTATS_INGESTION_TRAIN_TEST_SPLIT_RANDOM_STATE
        except Exception as e:
            raise DUStatsException(e, sys)

class DUStatsValidationConfig:

    def __init__(self, du_stats_pipeline_config:DUStatsPipelineConfig):
        try:
            self.artifact_dir=du_stats_pipeline_config.artifact_dir
            self.validation_report_filepath:str=os.path.join(
                self.artifact_dir,
                dustats_pipeline.DUSTATS_VALIDATION_DIR,
                dustats_pipeline.DUSTSTS_VALIDATION_REPORT_DIR,
                dustats_pipeline.DUSTATS_VALIDATION_REPORT_FILENAME
            )
            self.valid_train_data_filepath:str=os.path.join(
                self.artifact_dir,
                dustats_pipeline.DUSTATS_VALIDATION_DIR,
                dustats_pipeline.DUSTATS_VALIDATION_VALID_DIR,
                dustats_pipeline.TRAIN_FILENAME
            )
            self.valid_test_data_filepath:str=os.path.join(
                self.artifact_dir,
                dustats_pipeline.DUSTATS_VALIDATION_DIR,
                dustats_pipeline.DUSTATS_VALIDATION_VALID_DIR,
                dustats_pipeline.TEST_FILENAME
            )
            self.invalid_train_data_filepath:str=os.path.join(
                self.artifact_dir,
                dustats_pipeline.DUSTATS_VALIDATION_DIR,
                dustats_pipeline.DUSTATS_VALIDATION_INVALID_DIR,
                dustats_pipeline.TRAIN_FILENAME
            )
            self.invalid_test_data_filepath:str=os.path.join(
                self.artifact_dir,
                dustats_pipeline.DUSTATS_VALIDATION_DIR,
                dustats_pipeline.DUSTATS_VALIDATION_INVALID_DIR,
                dustats_pipeline.TEST_FILENAME
            )
            self.validation_schema_filepath:str=dustats_pipeline.SCHEMA_FILEPATH
            self.validation_data_drift_threshold:float=dustats_pipeline.DUSTATS_VALIDATION_DATA_DRIFT_THRESHOLD

        except Exception as e:
            raise DUStatsException(e, sys)

class DUStatsTransformationConfig:

    def __init__(self, dustats_pipeline_config:DUStatsPipelineConfig):
        try:
            self.artifact_dir = dustats_pipeline_config.artifact_dir
            self.train_data_filepath = os.path.join(
                self.artifact_dir,
                dustats_pipeline.DUSTATS_TRANSFORMATION_DIR,
                dustats_pipeline.DUSTATS_TRANSFORMATION_DATA_DIR,
                dustats_pipeline.TRAIN_FILENAME.replace('.csv', '.npy')
            )
            self.test_data_filepath = os.path.join(
                self.artifact_dir,
                dustats_pipeline.DUSTATS_TRANSFORMATION_DIR,
                dustats_pipeline.DUSTATS_TRANSFORMATION_DATA_DIR,
                dustats_pipeline.TEST_FILENAME.replace('.csv', '.npy')
            )
            self.preprocessor_filepath = os.path.join(
                self.artifact_dir,
                dustats_pipeline.DUSTATS_TRANSFORMATION_DIR,
                dustats_pipeline.DUSTATS_TRANSFORMATION_MODEL_DIR,
                dustats_pipeline.DUSTATS_TRANSFORMATION_MODEL_FILENAME
            )
            self.target_column = dustats_pipeline.TARGET_COLUMN
        except Exception as e:
            raise DUStatsException(e, sys)

class DUStatsModelTrainerConfig:

    def __init__(self, dustats_pipeline_config:DUStatsPipelineConfig):
        try:
            self.artifact_dir = dustats_pipeline_config.artifact_dir
            self.model_filepath = os.path.join(
                self.artifact_dir,
                dustats_pipeline.DUSTATS_MODEL_TRAINER_DIR,
                dustats_pipeline.DUSTATS_MODEL_TRAINER_MODEL_DIR,
                dustats_pipeline.DUSTATS_MODEL_TRAINER_MODEL_FILENAME
            )
            self.model_report_filepath = os.path.join(
                self.artifact_dir,
                dustats_pipeline.DUSTATS_MODEL_TRAINER_DIR,
                dustats_pipeline.DUSTATS_MODEL_TRAINER_MODEL_REPORT_DIR,
                dustats_pipeline.DUSTATS_MODEL_TRAINER_MODEL_REPORT_FILENAME
            )
        except Exception as e:
            raise DUStatsException(e, sys)

class DUStatsNeuralNetworkConfig:
    def __init__(self, dustats_pipeline_config:DUStatsPipelineConfig):
        try:
            self.artifact_dir = dustats_pipeline_config.artifact_dir
            self.training_dir = os.path.join(
                self.artifact_dir,
                dustats_pipeline.DUSTATS_NN_DIR,
                dustats_pipeline.DUSTATS_NN_TRAINING_DIR
            )
            self.training_report_filepath = os.path.join(
                self.training_dir,
                dustats_pipeline.DUSTATS_NN_TRAINING_REPORT_FILENAME
            )
            self.hidden_layers = dustats_pipeline.DUSTATS_NN_NUM_HIDDEN_LAYERS
            self.num_neurons = dustats_pipeline.DUSTATS_NN_NUM_NEURONS
            self.num_epochs = dustats_pipeline.DUSTATS_NN_NUM_TRAINING_EPOCHS
            self.random_seed = dustats_pipeline.DUSTATS_NN_RANDOM_SEED
        except Exception as e:
            raise DUStatsException(e, sys)