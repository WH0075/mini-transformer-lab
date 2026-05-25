import torch
from torch.utils.data import Dataset


class ToyClassificationDataset(Dataset):
    
    def __init__(self, num_samples: int = 1000) -> None:
        self.x = torch.randn(num_samples, 2)
        self.y = (self.x[:, 0] + self.x[:, 1] > 0).long()
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index] 
        