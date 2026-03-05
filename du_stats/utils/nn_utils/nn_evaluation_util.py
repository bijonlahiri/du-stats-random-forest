import os, sys
import numpy as np
import torch
from torch import nn
from du_stats.exception.exception import DUStatsException
from du_stats.logging.logger import logging
from du_stats.utils.nn_utils.nn_training_util import accuracy_fn

def nn_evaluation(model:nn.Module, X_test:np.array, y_test:np.array)->dict:
    try:
        X_test, y_test = torch.from_numpy(X_test).type(torch.float), torch.from_numpy(y_test).type(torch.long)
        model.eval()
        with torch.inference_mode():
            y_logits = model(X_test)
            y_pred = torch.argmax(y_logits, dim=1)
            acc = accuracy_fn(y_test, y_pred)
        
        return {'Test accuracy': f'{acc:.2f}%'}
    except Exception as e:
        raise DUStatsException(e, sys)