import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DUStatsDataset(Dataset):
    def __init__(self, X:torch.Tensor, y:torch.Tensor):
        self.X = X.to(torch.float)
        self.y = y.to(torch.long)

        self.X = self.X.permute(0, 2, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_data_loader(X:torch.Tensor, y:torch.Tensor, batch_size:int=32, shuffle:bool=False):
    dataset = DUStatsDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)