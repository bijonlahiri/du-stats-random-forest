import os, sys
from du_stats.exception.exception import DUStatsException
from du_stats.logging.logger import logging
from du_stats.entity.artifact_entity import DUStatsTransformationArtifact, DUStatsNeuralNetworkArtifact
from du_stats.entity.config_entity import DUStatsNeuralNetworkConfig
from du_stats.utils.nn_utils.nn_training_util import nn_training, nn_evaluation
from du_stats.utils.main_utils.utils import (
    load_numpy_array_from_file, save_yaml, load_tensor_artifact, save_model_state_dict
)
from du_stats.utils.conv_utils.data_loader_utils import create_data_loader
from du_stats.utils.conv_utils.conv_net_training_util import convnet_training
from du_stats.utils.conv_utils.conv_net_evaluation_util import convnet_evaluation

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
        self.model_filepath = dustats_nn_config.model_filepath
        self.evaluation_report_filepath = dustats_nn_config.evaluation_report_filepath
    
    def initiate_nn_training(self):
        dustats_nn_artifact = DUStatsNeuralNetworkArtifact()
        try:
            if self.transformation_done:
                train_data = load_numpy_array_from_file(self.train_filepath)
                X_train, y_train = train_data[:, :-1], train_data[:,-1]
                logging.info(f"Train data size: {train_data.shape}")
                test_data = load_numpy_array_from_file(self.test_filepath)
                X_test, y_test = test_data[:, :-1], test_data[:, -1]
                training_report, nn_model = nn_training(
                    X_train=X_train,
                    y_train=y_train,
                    num_hidden_layers=self.hidden_layers,
                    num_neurons=self.num_neurons,
                    num_epochs=self.num_epochs,
                    random_seed = self.random_seed if self.random_seed is not None else 0
                )
                evaluation_report = nn_evaluation(nn_model, X_test, y_test)
                save_yaml(f"ann_{self.training_report_filepath}", training_report)
                save_yaml(f"ann_{self.evaluation_report_filepath}", evaluation_report)
                save_model_state_dict(f"ann_{self.model_filepath}", nn_model)
                dustats_nn_artifact.nn_training_done = True
                dustats_nn_artifact.nn_training_report_filepath = self.training_report_filepath
                dustats_nn_artifact.nn_model_filepath = self.model_filepath
                dustats_nn_artifact.nn_evaluation_report_filepath = self.evaluation_report_filepath
                return dustats_nn_artifact
            else:
                logging.info('Transformation not done. Skipping NN training.')
                return dustats_nn_artifact
        except Exception as e:
            raise DUStatsException(e, sys)
    
    def initiate_convnet_training(self):
        dustats_convnet_artifact = DUStatsNeuralNetworkArtifact()
        try:
            if self.transformation_done:
                X_train, y_train = load_tensor_artifact(filepath=self.train_filepath)
                train_loader = create_data_loader(X_train, y_train, batch_size=32, shuffle=True)
                X_test, y_test = load_tensor_artifact(filepath=self.test_filepath)
                test_loader = create_data_loader(X_test, y_test, batch_size=32, shuffle=False)
                training_report, model = convnet_training(train_loader, 22, 4, 100, 42)
                evaluation_report = convnet_evaluation(model, test_loader)
                save_yaml(f"convnet_{self.training_report_filepath}", training_report)
                save_yaml(f"convnet_{self.evaluation_report_filepath}", evaluation_report)
                save_model_state_dict(f"convnet_{self.model_filepath}", model.state_dict())
                dustats_convnet_artifact.nn_training_done = True
                dustats_convnet_artifact.nn_training_report_filepath = self.training_report_filepath
                dustats_convnet_artifact.nn_model_filepath = self.model_filepath
                dustats_convnet_artifact.nn_evaluation_report_filepath = self.evaluation_report_filepath

                return dustats_convnet_artifact

            else:
                logging.info('Transformation not done. Skipping CONVNET training.')
                return dustats_convnet_artifact
        except Exception as e:
            raise DUStatsException(e, sys)