import os, sys
from du_stats.exception.exception import DUStatsException
from du_stats.logging.logger import logging
from du_stats.entity.artifact_entity import DUStatsTransformationArtifact, DUStatsModelTrainerArtifact
from du_stats.entity.config_entity import DUStatsModelTrainerConfig
from du_stats.utils.main_utils.utils import(
    load_numpy_array_from_file, load_object_from_file, save_object_to_file, save_yaml, read_yaml
)
from du_stats.utils.ml_utils.model_evaluation import evaluate_model
from du_stats.utils.ml_utils.model import DUStatsModel

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import(
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)

class DUStatsModelTrainer:

    def __init__(self, dustats_transformation_artifact:DUStatsTransformationArtifact, dustats_model_trainer_config:DUStatsModelTrainerConfig):
        try:
            self.transformation_done:bool=dustats_transformation_artifact.transformation_done
            self.train_data_filepath:str=dustats_transformation_artifact.transformed_train_data_filepath
            self.test_data_filepath:str=dustats_transformation_artifact.transformed_test_data_filepath
            self.preprocessor_filepath:str=dustats_transformation_artifact.preprocessor_filepath
            self.model_filepath:str=dustats_model_trainer_config.model_filepath
            self.model_report_filepath:str=dustats_model_trainer_config.model_report_filepath
        except Exception as e:
            raise DUStatsException(e, sys)
    
    def train_model(self, X_train, X_test, y_train, y_test)->tuple:
        try:
            model_dict = {
                # 'Logistic Regression': LogisticRegression(),
                # 'KNN Classifier': KNeighborsClassifier(),
                # 'Naive Bayes': GaussianNB(),
                'Decision Tree': DecisionTreeClassifier(),
                'Adaboost': AdaBoostClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'Random Forest': RandomForestClassifier()
            }
            params_dict = {
                # 'Logistic Regression': {
                #     'C': [0.01, 0.1, 1, 10, 100],
                #     'l1_ratio': [0, 0.25, 0.5, 0.75, 1]
                # },
                # 'KNN Classifier': {
                #     'n_neighbors': [1, 5, 10, 15],
                #     'algorithm': ['ball_tree', 'kd_tree', 'brute']
                # },
                # 'Naive Bayes': {},
                'Decision Tree': {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_depth': [5, 10, 15],
                    # 'max_features': [1.0, 0.75, 0.5, 0.25]
                },
                'Adaboost': {
                    'n_estimators': [5, 10, 15],
                    'learning_rate': [0.001, 0.01, 0.1]
                },
                'Gradient Boosting': {
                    'n_estimators': [5, 10, 15],
                    'learning_rate': [0.001, 0.01, 0.1]
                },
                'Random Forest': {
                    # 'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_depth': [5, 10, 15],
                    # 'max_features': [1.0, 0.75, 0.5],
                    'n_estimators': [5, 10, 15]
                }
            }
            model_report = evaluate_model(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                models=model_dict,
                params=params_dict
            )
            model_details = [
                (name, stats['Best Params'], stats['Test Score']['F1 Score'])
                for name, stats in model_report.items()
            ]
            f1_score_list = [i[2] for i in model_details]
            max_f1_score = max(f1_score_list)
            max_f1_score_index = f1_score_list.index(max_f1_score)
            best_model_name = model_details[max_f1_score_index][0]
            best_model_params = model_details[max_f1_score_index][1]
            best_model = model_report[best_model_name]['Estimator']
            best_model.set_params(**best_model_params)
            best_model.fit(X_train, y_train)
            preprocessor = load_object_from_file(self.preprocessor_filepath)
            dustats_model = DUStatsModel(preprocessor, best_model)
            save_object_to_file(self.model_filepath, best_model)
            model_report_to_save = {
                k: {ik: iv for ik, iv in v.items() if ik != 'Estimator'}
                for k, v in model_report.items()
            }
            save_yaml(self.model_report_filepath, model_report_to_save)
            return (best_model_name, best_model_params)
        except Exception as e:
            raise DUStatsException(e, sys)

    def initiate_model_trainer(self)->DUStatsModelTrainerArtifact:
        dustats_model_trainer_artifact = DUStatsModelTrainerArtifact()
        try:
            if self.transformation_done:
                train_arr = load_numpy_array_from_file(self.train_data_filepath)
                test_arr = load_numpy_array_from_file(self.test_data_filepath)
                X_train, X_test, y_train, y_test = (
                    train_arr[:, :-1],
                    test_arr[:, :-1],
                    train_arr[:, -1],
                    test_arr[:, -1]
                )
                best_model_name, best_model_params = self.train_model(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test
                )
                dustats_model_trainer_artifact.model_training_done=True
                dustats_model_trainer_artifact.model_filepath=self.model_filepath
                dustats_model_trainer_artifact.model_report_filepath=self.model_report_filepath
                dustats_model_trainer_artifact.best_model_name=best_model_name
                dustats_model_trainer_artifact.best_model_params=best_model_params
                return dustats_model_trainer_artifact
            else:
                logging.info('Data transformation failed. Skipping Model training.')
                return dustats_model_trainer_artifact
        except Exception as e:
            raise DUStatsException(e, sys)