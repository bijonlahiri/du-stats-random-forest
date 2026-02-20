import os, sys
from du_stats.logging.logger import logging
from du_stats.components.dustats_ingestion import DUStatsIngestion
from du_stats.components.dustats_validation import DUStatsValidation
from du_stats.entity.config_entity import DUStatsPipelineConfig, DUStatsIngestionConfig, DUStatsValidationConfig
from du_stats.exception.exception import DUStatsException

if __name__=='__main__':
    try:
        dustats_pipeline_config=DUStatsPipelineConfig()
        dustats_ingestion_config=DUStatsIngestionConfig(dustats_pipeline_config=dustats_pipeline_config)
        dustats_ingestion=DUStatsIngestion(dustats_ingestion_config=dustats_ingestion_config)
        dustats_ingestion_artifact=dustats_ingestion.initiate_data_ingestion()
        print(f"""
            Ingestion Raw Dump: {dustats_ingestion_artifact.raw_data_filepath}\n
            Ingestion Train Dump: {dustats_ingestion_artifact.train_data_filepath}\n
            Ingestion Test Dump: {dustats_ingestion_artifact.test_data_filepath}\n
        """)
        dustats_validation_config = DUStatsValidationConfig(dustats_pipeline_config)
        dustats_validation = DUStatsValidation(dustats_ingestion_artifact, dustats_validation_config)
        dustats_validation_artifact = dustats_validation.initiate_data_validation()
        print(f"""
            Validation Completed: {dustats_validation_artifact.validation_completed}\n
            Validation Report Filepath: {dustats_validation_artifact.validation_report_filepath}\n
            Valid Train Filepath: {dustats_validation_artifact.valid_train_data_filepath}\n
            Valid Test Filepath: {dustats_validation_artifact.valid_test_data_filepath}\n
            Invalid Train Filepath: {dustats_validation_artifact.invalid_train_data_filepath}\n
            Invalid Test Filepath: {dustats_validation_artifact.invalid_test_data_filepath}\n
        """)
    except Exception as e:
        raise DUStatsException(e, sys)