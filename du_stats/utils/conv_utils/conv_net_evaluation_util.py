import torch
from torch import nn
from du_stats.exception.exception import DUStatsException
from du_stats.logging.logger import logging
from du_stats.utils.nn_utils.nn_training_util import accuracy_fn

def convnet_evaluation(model:nn.Module, dataloader:torch.utils.data.DataLoader)->dict:
    try:
        acc = 0
        model.eval()
        with torch.inference_mode():
            for (X, y) in dataloader:
                y_logits = model(X)
                y_pred = torch.argmax(y_logits, dim=1)
                acc += accuracy_fn(y, y_pred)
            acc /= len(dataloader)
        
        return {'Test accuracy': f'{acc:.2f}%'}
    except Exception as e:
        raise DUStatsException(e, sys)