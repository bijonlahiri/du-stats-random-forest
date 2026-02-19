import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from entity.config_entity import DUStatsIngestionConfig
from exception.exception import DUStatsException
from logging.logger import logging

class DUStatsIngestion:

    def __init__(self, dustats_ingestion_config:DUStatsIngestionConfig)->None:
        self.fetch_query=dustats_ingestion_config.fetch_query
        self.raw_data_filepath=dustats_ingestion_config.raw_data_filepath
        self.train_test_split_ratio=dustats_ingestion_config.train_test_split_ratio
        self.train_data_filepath=dustats_ingestion_config.train_data_filepath
        self.test_data_filepath=dustats_ingestion_config.test_data_filepath
    
    def initiate_data_ingestion(self):
        try:
            pass
        except Exception as e:
            raise DUStatsException(e, sys)