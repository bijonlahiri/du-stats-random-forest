import os, sys
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from du_stats.exception.exception import DUStatsException
from du_stats.logging.logger import logging
from du_stats.entity.config_entity import DUStatsTransformationConfig
from du_stats.entity.artifact_entity import DUStatsValidationArtifact, DUStatsTransformationArtifact
from du_stats.utils.main_utils.utils import(
    read_dataframe_from_file, read_yaml, save_numpy_array_to_file, save_object_to_file, save_tensor_artifact
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
            self.convnet_seq_len = getattr(dustats_transformation_config, 'convnet_seq_len', None)
            self.convnet_train_test_ratio = getattr(dustats_transformation_config, 'convnet_train_test_ratio', None)

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
    
    def get_X_y_tensor(self, train_df:pd.DataFrame, test_df:pd.DataFrame)->tuple:
        try:
            dataframe = pd.concat([train_df, test_df], ignore_index=True)
            df = dataframe.sort_values(
                by=['site_name', 'log_date', 'cellid', 'ueid', 'uptime']
            )
            numerical_schema = self._schema['numerical_columns']
            numerical_columns = [list(col.keys())[0] for col in numerical_schema]
            scaler = StandardScaler()
            X = df[numerical_columns]
            scaler.fit(X)
            X_tensor = torch.tensor(X.values).reshape(-1, self.convnet_seq_len, len(numerical_columns))
            y = df.groupby(['site_name', 'log_date', 'cellid', 'ueid']).head(1)[self.target_column]
            y_transformed = y.map({
                'GOOD': 0,
                'BAD CHANNEL': 1,
                'GOOD CHANNEL HIGH BLER': 2,
                'SCHEDULER LIMITED': 3
            })
            y_tensor = torch.tensor(y_transformed.values)
            
            idx = np.arange(X_tensor.shape[0])
            train_idx_list, test_idx_list = train_test_split(idx, test_size=self.convnet_train_test_ratio)
            X_train_tensor_list = [X_tensor[idx] for idx in train_idx_list]
            X_test_tensor_list = [X_tensor[idx] for idx in test_idx_list]
            y_train_tensor_list = [y_tensor[idx] for idx in train_idx_list]
            y_test_tensor_list = [y_tensor[idx] for idx in test_idx_list]
            X_train = torch.stack(X_train_tensor_list, dim=0)
            X_test = torch.stack(X_test_tensor_list, dim=0)
            y_train = torch.stack(y_train_tensor_list, dim=0)
            y_test = torch.stack(y_test_tensor_list, dim=0)
            return X_train, y_train, X_test, y_test, scaler

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
                X_test, y_test = self.get_X_y_numpy_array(test_df)
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
    
    def initiate_data_transformation_convnet(self)->DUStatsTransformationArtifact:
        logging.info('ConvNet Data transformation initiated.')
        dustats_transformation_artifact=DUStatsTransformationArtifact()
        try:
            if self.validation_done:
                train_df = read_dataframe_from_file(self.valid_train_data_filepath)
                test_df = read_dataframe_from_file(self.valid_test_data_filepath)
                X_train, y_train, X_test, y_test, scaler = self.get_X_y_tensor(train_df, test_df)
                save_object_to_file(self.preprocessor_object_filepath, scaler)
                save_tensor_artifact(self.train_array_filepath, X_train, y_train)
                save_tensor_artifact(self.test_array_filepath, X_test, y_test)
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

    def initiate_data_transformation_ml(self)->DUStatsTransformationArtifact:
        """Traditional ML transformation: StandardScaler normalization for scikit-learn models."""
        logging.info('ML Data transformation initiated.')
        dustats_transformation_artifact=DUStatsTransformationArtifact()
        try:
            if self.validation_done:
                train_df = read_dataframe_from_file(self.valid_train_data_filepath)
                X_train, y_train = self.get_X_y_numpy_array(train_df)
                test_df = read_dataframe_from_file(self.valid_test_data_filepath)
                X_test, y_test = self.get_X_y_numpy_array(test_df)
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
                logging.info('ML Data transformation completed.')
                return dustats_transformation_artifact
            else:
                logging.info('Validation not completed. Skipping transformation.')
                return dustats_transformation_artifact
        except Exception as e:
            raise DUStatsException(e, sys)

    def initiate_data_transformation_ann(self)->DUStatsTransformationArtifact:
        """ANN transformation: StandardScaler normalization with flattened features for feedforward networks."""
        logging.info('ANN Data transformation initiated.')
        dustats_transformation_artifact=DUStatsTransformationArtifact()
        try:
            if self.validation_done:
                train_df = read_dataframe_from_file(self.valid_train_data_filepath)
                X_train, y_train = self.get_X_y_numpy_array(train_df)
                test_df = read_dataframe_from_file(self.valid_test_data_filepath)
                X_test, y_test = self.get_X_y_numpy_array(test_df)
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
                logging.info('ANN Data transformation completed.')
                return dustats_transformation_artifact
            else:
                logging.info('Validation not completed. Skipping transformation.')
                return dustats_transformation_artifact
        except Exception as e:
            raise DUStatsException(e, sys)