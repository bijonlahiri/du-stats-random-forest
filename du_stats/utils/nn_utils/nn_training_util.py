import torch
from torch import nn
import os, sys
from du_stats.exception.exception import DUStatsException
from du_stats.logging.logger import logging
from tqdm import tqdm

class DUStatsNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layers=1, num_neurons=4):
        super().__init__()
        layers = []

        for layer in range(num_hidden_layers):
            if layer == 0:
                layers.append(nn.Linear(input_size, num_neurons))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(num_neurons, num_neurons))
                layers.append(nn.ReLU())
        
        layers.append(nn.Linear(num_neurons, output_size))

        self.neural_network = nn.Sequential(*layers)

    def forward(self, x):
        return self.neural_network(x)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct/len(y_true))*100

def nn_training(X_train, y_train, num_hidden_layers, num_neurons, num_epochs=100, random_seed=0)->dict:
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(random_seed)
        X_train = torch.from_numpy(X_train).type(torch.float).to(device)
        y_train = torch.from_numpy(y_train).type(torch.long).to(device)
        # X_test = torch.from_numpy(X_test).type(torch.float).to(device)
        # y_test = torch.from_numpy(y_test).type(torch.float).to(device)

        nn_model = DUStatsNeuralNetwork(
            input_size=X_train.shape[1],
            output_size=len(torch.unique(y_train)),
            num_hidden_layers=num_hidden_layers,
            num_neurons=num_neurons
        ).to(device)
        logging.info(f"Model: {nn_model}")
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(nn_model.parameters(), lr=0.01)
        report = {}
        for epoch in tqdm(range(num_epochs)):
            nn_model.train()
            # Forward pass
            y_logits = nn_model(X_train)
            y_proba = torch.softmax(y_logits, dim=1)
            y_labels = torch.argmax(y_proba, dim=1)
            loss = loss_fn(y_logits, y_train)
            acc = accuracy_fn(y_train, y_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            report[epoch] = {
                'Train Loss': loss.item(),
                'Train Accuracy': acc
            }
        return report
    except Exception as e:
        raise DUStatsException(e, sys)