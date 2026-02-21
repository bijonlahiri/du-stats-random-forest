import os, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from du_stats.exception.exception import DUStatsException
from du_stats.logging.logger import logging
from du_stats.entity.config_entity import DUStatsTransformationConfig
from du_stats.entity.artifact_entity import DUStatsValidationArtifact, DUStatsTransformationArtifact
from du_stats.utils.main_utils.utils import(
    read_dataframe_from_file, read_yaml, save_numpy_array_to_file, save_object_to_file
)
from du_stats.constants.dustats_pipeline import SCHEMA_FILEPATH

class DUStatsTransformation:

    def __init__(self, dustats_validation_artifact:DUStatsValidationArtifact, dustats_transformation_config:DUStatsTransformationConfig):
        try:
            self.validation_done:bool=dustats_validation_artifact.validation_completed
            self.valid_train_data_filepath:str = dustats_validation_artifact.valid_train_data_filepath
            self.valid_test_data_filepath:str = dustats_validation_artifact.valid_test_data_filepath
            self.preprocessor_object_filepath:str=dustats_transformation_config.preprocessor_filepath
            self.train_array_filepath:str=dustats_transformation_config.train_data_filepath
            self.test_array_filepath:str=dustats_transformation_config.test_data_filepath
            self.target_column:str=dustats_transformation_config.target_column
            self._schema = read_yaml(SCHEMA_FILEPATH)

        except Exception as e:
            raise DUStatsException(e, sys)
    
    def get_X_y_numpy_array(self, dataframe:pd.DataFrame)->tuple:
        try:
            numerical_schema = self._schema['numerical_columns']
            numerical_columns = [list(col.keys())[0] for col in numerical_schema]
            X = dataframe[numerical_columns]
            y = dataframe[self.target_column]
            return (X, y)
        except Exception as e:
            raise DUStatsException(e, sys)
        
    def initiate_data_transformation(self)->DUStatsTransformationArtifact:
        logging.info('Data transformation initiated.')
        dustats_transformation_artifact=DUStatsTransformationArtifact()
        try:
            if self.validation_done:
                train_df = read_dataframe_from_file(self.valid_train_data_filepath)
                X_train, y_train = self.get_X_y_numpy_array(train_df)
                test_df = read_dataframe_from_file(self.valid_test_data_filepath)
                X_test, y_test = self.get_X_y_numpy_array(train_df)
                scaler = StandardScaler()
                scaler.fit(X_train)
                save_object_to_file(self.preprocessor_object_filepath, scaler)
                X_train_transformed = scaler.transform(X_train)
                X_test_transformed = scaler.transform(X_test)
                y_train_transformed = y_train.map({
                    'GOOD': 0,
                    'BAD CHANNEL': 1,
                    'GOOD CHANNEL HIGH BLER': 2,
                    'SCHEDULER LIMITED': 3
                })
                y_test_transformed = y_test.map({
                    'GOOD': 0,
                    'BAD CHANNEL': 1,
                    'GOOD CHANNEL HIGH BLER': 2,
                    'SCHEDULER LIMITED': 3
                })
                train_arr = np.c_[X_train_transformed, np.array(y_train_transformed)]
                test_arr = np.c_[X_test_transformed, np.array(y_test_transformed)]
                save_numpy_array_to_file(self.train_array_filepath, train_arr)
                save_numpy_array_to_file(self.test_array_filepath, test_arr)
                dustats_transformation_artifact.transformation_done=True
                dustats_transformation_artifact.transformed_train_data_filepath=self.train_array_filepath
                dustats_transformation_artifact.transformed_test_data_filepath=self.test_array_filepath
                dustats_transformation_artifact.preprocessor_filepath=self.preprocessor_object_filepath
                logging.info('Data transformation completed.')
                return dustats_transformation_artifact
            else:
                logging.info('Validation not completed. Skipping transformation.')
                return dustats_transformation_artifact
        except Exception as e:
            raise DUStatsException(e, sys)