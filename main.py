import os, sys
from du_stats.logging.logger import logging
from du_stats.components.dustats_ingestion import DUStatsIngestion
from du_stats.components.dustats_validation import DUStatsValidation
from du_stats.components.dustats_transformation import DUStatsTransformation
from du_stats.components.dustats_model_trainer import DUStatsModelTrainer
from du_stats.components.dustats_nn_model_trainer import DUStatsNeuralNetworkTrainer
from du_stats.entity.config_entity import(
    DUStatsPipelineConfig, DUStatsIngestionConfig, DUStatsValidationConfig, DUStatsTransformationConfig,
    DUStatsModelTrainerConfig, DUStatsNeuralNetworkConfig
)
from du_stats.exception.exception import DUStatsException

if __name__=='__main__':
    try:
        dustats_pipeline_config=DUStatsPipelineConfig()
        dustats_ingestion_config=DUStatsIngestionConfig(dustats_pipeline_config=dustats_pipeline_config)
        dustats_ingestion=DUStatsIngestion(dustats_ingestion_config=dustats_ingestion_config)
        dustats_ingestion_artifact=dustats_ingestion.initiate_data_ingestion()
        print(f"""
            Ingestion Completed: {dustats_ingestion_artifact.ingestion_done}\n
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
        dustats_transformation_config = DUStatsTransformationConfig(dustats_pipeline_config)
        dustats_transformation = DUStatsTransformation(dustats_validation_artifact, dustats_transformation_config)
        dustats_transformation_artifact = dustats_transformation.initiate_data_transformation_convnet()
        print(f"""
            Transformation Completed: {dustats_transformation_artifact.transformation_done}\n
            Preprocessor Filepath: {dustats_transformation_artifact.preprocessor_filepath}\n
            Train Array Filepath: {dustats_transformation_artifact.transformed_train_data_filepath}\n
            Test Array Filepath: {dustats_transformation_artifact.transformed_train_data_filepath}\n
        """)
        # dustats_model_trainer_config = DUStatsModelTrainerConfig(dustats_pipeline_config)
        # dustats_model_trainer = DUStatsModelTrainer(dustats_transformation_artifact, dustats_model_trainer_config)
        # dustats_model_trainer_artifact = dustats_model_trainer.initiate_model_trainer()
        # print(f"""
        #     Model Training Completed: {dustats_model_trainer_artifact.model_training_done}\n
        #     Model Filepath: {dustats_model_trainer_artifact.model_filepath}\n
        #     Model Report Filepath: {dustats_model_trainer_artifact.model_report_filepath}\n
        #     Best Model Name: {dustats_model_trainer_artifact.best_model_name}\n
        #     Best Model Params: {dustats_model_trainer_artifact.best_model_params}\n
        # """)
        dustats_nn_config = DUStatsNeuralNetworkConfig(dustats_pipeline_config)
        dustats_nn_trainer = DUStatsNeuralNetworkTrainer(dustats_transformation_artifact, dustats_nn_config)
        dustats_nn_artifact = dustats_nn_trainer.initiate_convnet_training()
        print(f"""
            Neural Network Training Completed: {dustats_nn_artifact.nn_training_done}\n
            Training report filepath: {dustats_nn_artifact.nn_training_report_filepath}\n
        """)

    except Exception as e:
        raise DUStatsException(e, sys)