import os, sys
import numpy as np
from du_stats.exception.exception import DUStatsException
from du_stats.logging.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate_model(X_train:np.array, X_test:np.array, y_train:np.array, y_test:np.array, models:dict, params:dict)->dict:
    try:
        report={}
        for name, model in models.items():
            gs = GridSearchCV(
                estimator=model,
                param_grid=params[name],
                cv=3,
                n_jobs=-1,
                verbose=3
            )
            gs.fit(X_train, y_train)
            y_train_pred = gs.predict(X_train)
            y_test_pred = gs.predict(X_test)
            train_f1_score = f1_score(y_train, y_train_pred, average='micro')
            train_precision_score = precision_score(y_train, y_train_pred, average='micro')
            train_recall_score = recall_score(y_train, y_train_pred, average='micro')
            test_f1_score = f1_score(y_test, y_test_pred, average='micro')
            test_precision_score = precision_score(y_test, y_test_pred, average='micro')
            test_recall_score = recall_score(y_test, y_test_pred, average='micro')
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