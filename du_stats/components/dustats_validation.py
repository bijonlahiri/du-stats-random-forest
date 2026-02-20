import os, sys
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from du_stats.entity.artifact_entity import DUStatsIngestionArtifact, DUStatsValidationArtifact
from du_stats.entity.config_entity import DUStatsValidationConfig
from du_stats.exception.exception import DUStatsException
from du_stats.logging.logger import logging
from du_stats.utils.main_utils.utils import read_yaml, read_dataframe_from_file, save_yaml, save_dataframe_to_file

class DUStatsValidation:

    def __init__(self, dustats_ingestion_artifact:DUStatsIngestionArtifact, dustats_validation_config:DUStatsValidationConfig):
        try:
            self.ingestion_completed:bool=dustats_ingestion_artifact.ingestion_done
            self.ingested_train_data_filepath:str=dustats_ingestion_artifact.train_data_filepath
            self.ingested_test_data_filepath:str=dustats_ingestion_artifact.test_data_filepath
            self.validation_report_filepath:str=dustats_validation_config.validation_report_filepath
            self.valid_train_data_filepath:str=dustats_validation_config.valid_train_data_filepath
            self.valid_test_data_filepath:str=dustats_validation_config.valid_test_data_filepath
            self.invalid_train_data_filepath:str=dustats_validation_config.invalid_train_data_filepath
            self.invalid_test_data_filepath:str=dustats_validation_config.invalid_test_data_filepath
            self._schema:dict=read_yaml(dustats_validation_config.validation_schema_filepath)
            self.data_drift_threshold:float=dustats_validation_config.validation_data_drift_threshold
        except Exception as e:
            raise DUStatsException(e,  sys)
        
    def validate_columns(self, dataframe:pd.DataFrame)->bool:
        try:
            if self._schema:
                logging.info('Comparing schema')
                validated_num_of_columns = (len(self._schema['columns'])==len(dataframe.columns.tolist()))
                validated_all_columns=True
                schema_columns = [list(col.keys())[0] for col in self._schema['columns']]
                dataframe_columns = dataframe.columns.tolist()
                for column in schema_columns:
                    if column not in dataframe_columns:
                        validated_all_columns=False
                logging.info(f'Num of columns: {validated_num_of_columns}\nAll columns: {validated_all_columns}')
                return validated_num_of_columns and validated_all_columns
        except Exception as e:
            raise DUStatsException(e, sys)
    
    def detect_data_drift(self, train_df:pd.DataFrame, test_df:pd.DataFrame)->bool:
        validated_data_drift=True
        try:
            test_columns = [list(col.keys())[0] for col in self._schema['numerical_columns']]
            report={}
            for column in test_columns:
                drift_result = float(np.round(ks_2samp(train_df[column], test_df[column]).pvalue, 2))
                report[column] = {
                    'p_value': drift_result,
                    'drift_status': (drift_result < self.data_drift_threshold)
                }
            save_yaml(self.validation_report_filepath, report)
            for i in report.values():
                if i['drift_status']:
                    validated_data_drift=False
            return validated_data_drift
        except Exception as e:
            raise DUStatsException(e, sys)
        
    def initiate_data_validation(self)->DUStatsValidationArtifact:
        dustats_validation_artifact=DUStatsValidationArtifact()
        try:
            ## Check if data was properly ingested
            if self.ingestion_completed:
                train_df = read_dataframe_from_file(self.ingested_train_data_filepath)
                ## Validate train data columns
                if train_df.columns.tolist():
                    validate_train_data_columns=self.validate_columns(train_df)
                else:
                    logging.info('Train dataframe read unsuccessful')
                    return dustats_validation_artifact
                test_df = read_dataframe_from_file(self.ingested_test_data_filepath)
                ## validate test data columns
                if test_df.columns.tolist():
                    validate_test_data_columns=self.validate_columns(test_df)
                else:
                    logging.info('Test dataframe read unsuccessful.')
                    return dustats_validation_artifact
                ## If both columns are validated, then validate data frift
                if validate_train_data_columns and validate_test_data_columns:
                    validate_data_drift = self.detect_data_drift(train_df, test_df)
                    if validate_data_drift:
                        save_dataframe_to_file(self.valid_train_data_filepath, train_df)
                        save_dataframe_to_file(self.valid_test_data_filepath, test_df)
                        dustats_validation_artifact.validation_completed=True
                        dustats_validation_artifact.validation_report_filepath=self.validation_report_filepath
                        dustats_validation_artifact.valid_train_data_filepath=self.valid_train_data_filepath
                        dustats_validation_artifact.valid_test_data_filepath=self.valid_test_data_filepath
                    else:
                        save_dataframe_to_file(self.invalid_train_data_filepath, train_df)
                        save_dataframe_to_file(self.invalid_test_data_filepath, test_df)
                        dustats_validation_artifact.validation_completed=False
                        dustats_validation_artifact.invalid_train_data_filepath=self.invalid_train_data_filepath
                        dustats_validation_artifact.invalid_test_data_filepath=self.invalid_test_data_filepath

                    logging.info('Data validation completed.')
                    return dustats_validation_artifact
                else:
                    logging.info('Mismatch in expected columns.')
                    return dustats_validation_artifact
            else:
                logging.info('Data Ingestion failed. Skipping data validation.')
                return dustats_validation_artifact
        except Exception as e:
            raise DUStatsException(e, sys)