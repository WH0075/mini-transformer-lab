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


class CharLanguageModelDataset(Dataset):

    def __init__(self, token_ids: list[int], block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        
        if len(token_ids) <= block_size:
            raise ValueError("len(token_ids) must be greater than block_size.")
        
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size
        
    def __len__(self) -> int:
        return len(self.data) - self.block_size
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index: index + self.block_size]
        y = self.data[index + 1: index + self.block_size + 1]

        return x, y
        
        