import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from du_stats.entity.config_entity import DUStatsIngestionConfig
from du_stats.exception.exception import DUStatsException
from du_stats.entity.artifact_entity import DUStatsIngestionArtifact
from du_stats.logging.logger import logging
from du_stats.utils.main_utils.utils import fetch_data, save_dataframe_to_file

class DUStatsIngestion:

    def __init__(self, dustats_ingestion_config:DUStatsIngestionConfig)->None:
        self.fetch_query=dustats_ingestion_config.fetch_query
        self.raw_data_filepath=dustats_ingestion_config.raw_data_filepath
        self.train_test_split_ratio=dustats_ingestion_config.train_test_split_ratio
        self.train_data_filepath=dustats_ingestion_config.train_data_filepath
        self.test_data_filepath=dustats_ingestion_config.test_data_filepath
    
    def ingest_data(self, query:str, raw_data_filepath:str)->pd.DataFrame:
        try:
            df=fetch_data(fetch_query=query)

            if df.columns.tolist():
                logging.info('Successfully fetched data from Databricks server')
                return df
            else:
                logging.info('Could not fetch data from Databricks server. Please check environments are set properly.')
                return pd.DataFrame()
            
        except Exception as e:
            raise DUStatsException(e, sys)
        
    def initiate_data_ingestion(self)->DUStatsIngestionArtifact:
        dustats_ingestion_artifact=DUStatsIngestionArtifact()
        try:
            df:pd.DataFrame=self.ingest_data(self.fetch_query, self.raw_data_filepath)
            if df.columns.tolist():
                logging.info('Splitting into train test data')
                train_set, test_set = train_test_split(df, test_size=self.train_test_split_ratio)
                if save_dataframe_to_file(self.raw_data_filepath, df):
                    dustats_ingestion_artifact.raw_data_filepath=self.raw_data_filepath
                if save_dataframe_to_file(self.train_data_filepath, train_set):
                    dustats_ingestion_artifact.train_data_filepath=self.train_data_filepath
                if save_dataframe_to_file(self.test_data_filepath, test_set):
                    dustats_ingestion_artifact.test_data_filepath=self.test_data_filepath
                dustats_ingestion_artifact.ingestion_done=True
            else:
                logging.info('No dataframe to save')
            logging.info('Ingestion Completed.')
            return dustats_ingestion_artifact
        
        except Exception as e:
            raise DUStatsException(e, sys)