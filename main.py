import os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from du_stats.logging.logger import logging
from du_stats.components.dustats_ingestion import DUStatsIngestion
from du_stats.components.dustats_validation import DUStatsValidation
from du_stats.components.dustats_transformation import DUStatsTransformation
from du_stats.components.dustats_model_trainer import DUStatsModelTrainer
from du_stats.components.dustats_nn_model_trainer import DUStatsNeuralNetworkTrainer
from du_stats.entity.config_entity import(
    DUStatsPipelineConfig, DUStatsIngestionConfig, DUStatsValidationConfig, DUStatsTransformationConfig,
    DUStatsModelTrainerConfig, DUStatsNeuralNetworkConfig, DUStatsTransformationMLConfig,
    DUStatsTransformationANNConfig, DUStatsTransformationConvNetConfig
)
from du_stats.exception.exception import DUStatsException

def transform_and_train_ml(pipeline_config, validation_artifact):
    """Transform data for ML and train traditional ML models."""
    try:
        logging.info("ML: Transformation and training started.")
        ml_config = DUStatsTransformationMLConfig(pipeline_config)
        transformation = DUStatsTransformation(validation_artifact, ml_config)
        ml_transform_artifact = transformation.initiate_data_transformation_ml()

        model_trainer_config = DUStatsModelTrainerConfig(pipeline_config)
        model_trainer = DUStatsModelTrainer(ml_transform_artifact, model_trainer_config)
        ml_model_artifact = model_trainer.initiate_model_trainer()

        logging.info("ML: Training completed.")
        return ('ML', ml_model_artifact)
    except Exception as e:
        logging.error(f"ML: Error occurred - {str(e)}")
        return ('ML', None)

def transform_and_train_ann(pipeline_config, validation_artifact):
    """Transform data for ANN and train artificial neural network."""
    try:
        logging.info("ANN: Transformation and training started.")
        ann_config = DUStatsTransformationANNConfig(pipeline_config)
        transformation = DUStatsTransformation(validation_artifact, ann_config)
        ann_transform_artifact = transformation.initiate_data_transformation_ann()

        # Note: ANN trainer may need to be implemented or use existing NN trainer
        # For now, using the same neural network trainer
        nn_config = DUStatsNeuralNetworkConfig(pipeline_config)
        nn_trainer = DUStatsNeuralNetworkTrainer(ann_transform_artifact, nn_config)
        ann_artifact = nn_trainer.initiate_convnet_training()

        logging.info("ANN: Training completed.")
        return ('ANN', ann_artifact)
    except Exception as e:
        logging.error(f"ANN: Error occurred - {str(e)}")
        return ('ANN', None)

def transform_and_train_convnet(pipeline_config, validation_artifact):
    """Transform data for ConvNet and train 1D convolutional neural network."""
    try:
        logging.info("ConvNet: Transformation and training started.")
        convnet_config = DUStatsTransformationConvNetConfig(pipeline_config)
        transformation = DUStatsTransformation(validation_artifact, convnet_config)
        convnet_transform_artifact = transformation.initiate_data_transformation_convnet()

        nn_config = DUStatsNeuralNetworkConfig(pipeline_config)
        nn_trainer = DUStatsNeuralNetworkTrainer(convnet_transform_artifact, nn_config)
        convnet_artifact = nn_trainer.initiate_convnet_training()

        logging.info("ConvNet: Training completed.")
        return ('ConvNet', convnet_artifact)
    except Exception as e:
        logging.error(f"ConvNet: Error occurred - {str(e)}")
        return ('ConvNet', None)

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

        print("\n" + "="*80)
        print("Running ML, ANN, and ConvNet in parallel...")
        print("="*80 + "\n")

        # Run all three model pipelines in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(transform_and_train_ml, dustats_pipeline_config, dustats_validation_artifact): 'ML',
                executor.submit(transform_and_train_ann, dustats_pipeline_config, dustats_validation_artifact): 'ANN',
                executor.submit(transform_and_train_convnet, dustats_pipeline_config, dustats_validation_artifact): 'ConvNet'
            }

            results = {}
            for future in as_completed(futures):
                model_type, artifact = future.result()
                results[model_type] = artifact
                if artifact:
                    print(f"\n{model_type} Training Results:")
                    print(f"  Training Done: {artifact.model_training_done if hasattr(artifact, 'model_training_done') else artifact.nn_training_done}")
                    if hasattr(artifact, 'model_filepath'):
                        print(f"  Model Filepath: {artifact.model_filepath}")
                        print(f"  Report Filepath: {artifact.model_report_filepath}")
                    else:
                        print(f"  Model Filepath: {artifact.nn_model_filepath}")
                        print(f"  Training Report: {artifact.nn_training_report_filepath}")
                        print(f"  Evaluation Report: {artifact.nn_evaluation_report_filepath}")

        print("\n" + "="*80)
        print("All models training completed!")
        print("="*80)

    except Exception as e:
        raise DUStatsException(e, sys)