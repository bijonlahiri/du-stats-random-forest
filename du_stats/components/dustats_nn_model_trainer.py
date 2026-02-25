import os, sys
from du_stats.exception.exception import DUStatsException
from du_stats.logging.logger import logging
from du_stats.entity.artifact_entity import DUStatsTransformationArtifact, DUStatsNeuralNetworkArtifact
from du_stats.entity.config_entity import DUStatsNeuralNetworkConfig
from du_stats.utils.nn_utils.nn_training_util import nn_training
from du_stats.utils.main_utils.utils import load_numpy_array_from_file, save_yaml

class DUStatsNeuralNetworkTrainer:
    def __init__(self, dustats_transformation_artifact: DUStatsTransformationArtifact, dustats_nn_config: DUStatsNeuralNetworkConfig):
        self.hidden_layers = dustats_nn_config.hidden_layers
        self.num_neurons = dustats_nn_config.num_neurons
        self.num_epochs = dustats_nn_config.num_epochs
        self.training_report_filepath = dustats_nn_config.training_report_filepath
        self.train_filepath = dustats_transformation_artifact.transformed_train_data_filepath
        self.test_filepath = dustats_transformation_artifact.transformed_test_data_filepath
        self.transformation_done = dustats_transformation_artifact.transformation_done
        self.random_seed = dustats_nn_config.random_seed
    
    def initiate_nn_training(self):
        dustats_nn_artifact = DUStatsNeuralNetworkArtifact()
        try:
            if self.transformation_done:
                train_data = load_numpy_array_from_file(self.train_filepath)
                X_train, y_train = train_data[:, :-1], train_data[:,-1]
                logging.info(f"Train data size: {train_data.shape}")
                training_report = nn_training(
                    X_train=X_train,
                    y_train=y_train,
                    num_hidden_layers=self.hidden_layers,
                    num_neurons=self.num_neurons,
                    num_epochs=self.num_epochs,
                    random_seed = self.random_seed if self.random_seed is not None else 0
                )
                save_yaml(self.training_report_filepath, training_report)
                dustats_nn_artifact.nn_training_done = True
                dustats_nn_artifact.nn_training_report_filepath = self.training_report_filepath
                return dustats_nn_artifact
            else:
                logging.info('Transformation not done. Skipping NN training.')
                return dustats_nn_artifact
        except Exception as e:
            raise DUStatsException(e, sys)