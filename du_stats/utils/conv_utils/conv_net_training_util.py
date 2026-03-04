import torch
from torch import nn
import os, sys
from tqdm import tqdm
from du_stats.exception.exception import DUStatsException
from du_stats.logging.logger import logging
from du_stats.utils.nn_utils.nn_training_util import accuracy_fn

class DUStatsConvNet(nn.Module):
    def __init__(self, in_channels:int, num_classes:int):
        super().__init__()
        self.conv1=nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.relu=nn.ReLU()
        self.pool=nn.MaxPool1d(kernel_size=2)
        
        self.conv2=nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        self.global_pool=nn.AdaptiveAvgPool1d(1)
        
        self.fc=nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

def convnet_training(dataloader:torch.utils.data.DataLoader,
                     in_channels:int,
                     num_classes:int,
                     num_epochs:int=100,
                     random_seed:int=0)->dict:
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(random_seed)
        
        convnet_model = DUStatsConvNet(in_channels=in_channels, num_classes=num_classes).to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=convnet_model.parameters(), lr=0.01)
        convnet_model.train()
        report = {}
        for epoch in tqdm(range(num_epochs), desc='Training Conv Net'):
            train_loss = 0
            acc = 0
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                y_logits = convnet_model(X)
                y_pred = torch.argmax(y_logits, dim=1)
                loss = loss_fn(y_logits, y)
                train_loss += loss.item()
                acc += accuracy_fn(y, y_pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss /= len(dataloader)
            acc /= len(dataloader)
            report[epoch] = {
                'Train Loss': f'{train_loss:.4f}',
                'Train Accuracy': f'{acc:.2f}%'
            }

        return report, convnet_model.cpu()

    except Exception as e:
        raise DUStatsException(e, sys)