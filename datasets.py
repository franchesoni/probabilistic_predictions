import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class AbstractDataset(ABC):
    @abstractmethod
    def get_feature_dim(self):
        pass

    @abstractmethod
    def get_target_dim(self):
        pass

    @abstractmethod
    def upper_bound(self):
        pass

    @abstractmethod
    def lower_bound(self):
        pass

class BishopToy(AbstractDataset, torch.utils.data.Dataset):
    def __init__(self, seed=0, noise_strength=0.1, n_samples=10000):
        torch.manual_seed(seed)
        self.noise_strength = noise_strength
        self.n_samples = n_samples
        self.Y = torch.rand(size=(self.n_samples,))
        noise = (torch.rand(self.n_samples) * 2 - 1) * self.noise_strength
        self.X = self.Y + 0.3 * np.sin(2 * np.pi * self.Y) + noise

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def visualize_samples(self, dstfile='vis.png'):
        plt.figure()
        plt.scatter(self.X, self.Y, s=1)
        plt.savefig(dstfile)

    def get_feature_dim(self):
        return 1

    def get_target_dim(self):
        return 1

    def upper_bound(self):
        return 1

    def lower_bound(self):
        return 0
    

# ds = BishopToy(n_samples=1000)
# ds.visualize_samples()

def get_dataset(dataset_name, **kwargs):
    if dataset_name == 'bishop_toy':
        return BishopToy(**kwargs)
    else:
        raise ValueError('Unknown dataset name')


