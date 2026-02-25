import os, sys
import numpy as np
from du_stats.exception.exception import DUStatsException
from du_stats.logging.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
import mlflow

def evaluate_model(X_train:np.array, X_test:np.array, y_train:np.array, y_test:np.array, models:dict, params:dict)->dict:
    try:
        report={}
        # mlflow.sklearn.autolog()
        for name, model in models.items():
            gs = GridSearchCV(
                estimator=model,
                param_grid=params[name],
                cv=3,
                n_jobs=-1,
                verbose=3
            )
            with mlflow.start_run() as run:
                gs.fit(X_train, y_train)
                y_train_pred = gs.predict(X_train)
                y_test_pred = gs.predict(X_test)
                train_f1_score = f1_score(y_train, y_train_pred, average='micro')
                train_precision_score = precision_score(y_train, y_train_pred, average='micro')
                train_recall_score = recall_score(y_train, y_train_pred, average='micro')
                mlflow.log_metric('train_f1_score', train_f1_score)
                mlflow.log_metric('train_precision_score', train_precision_score)
                mlflow.log_metric('train_recall_score', train_recall_score)
                test_f1_score = f1_score(y_test, y_test_pred, average='micro')
                test_precision_score = precision_score(y_test, y_test_pred, average='micro')
                test_recall_score = recall_score(y_test, y_test_pred, average='micro')
                mlflow.log_metric('test_f1_score', test_f1_score)
                mlflow.log_metric('test_precision_score', test_precision_score)
                mlflow.log_metric('test_recall_score', test_recall_score)
                logging.info(f"GS Best Params for {name}: {gs.best_params_}")
                mlflow.log_params(gs.best_params_)
                estimator = gs.best_estimator_
                estimator.set_params(**gs.best_params_)
                estimator.fit(X_train, y_train)
                mlflow.sklearn.log_model(
                    model=estimator,
                    name=name
                )
            report[name] = {
                'Estimator': model,
                'Best Params': gs.best_params_,
                'Train Score': {
                    'F1 Score': train_f1_score,
                    'Precision Score': train_precision_score,
                    'Recall Score': train_recall_score
                },
                'Test Score': {
                    'F1 Score': test_f1_score,
                    'Precision Score': test_precision_score,
                    'Recall Score': test_recall_score
                }
            }
            
        return report

    except Exception as e:
        raise DUStatsException(e, sys)