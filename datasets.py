import os
import pickle
from abc import ABC, abstractmethod

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from ucimlrepo import fetch_ucirepo


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
    def __init__(self, noise_strength=0.1, n_samples=1000, split='train'):
        if split == 'train':
            seed = 0
        elif split == 'val':
            seed = 1
        elif split == 'test':
            seed = 2

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

    def visualize_samples(self, dstfile="vis.png"):
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


def dotdict_to_dict(d):
    if isinstance(d, dict):
        return {k: dotdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [dotdict_to_dict(i) for i in d]
    else:
        return d

# cache the fetching of datasets
def cached_fetch_ucirepo(name):
    # Directory for cache files
    cache_dir = "./cache_datasets/"
    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    goodname = name.replace(" ", "_").replace("-", "_").lower()
    cache_file = os.path.join(cache_dir, f"{goodname}.pkl")
    if os.path.exists(cache_file):
        # Load from cache
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    else:
        # Fetch and save to cache
        result = fetch_ucirepo(name=name)
        dictres = dotdict_to_dict(result)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(dictres, f)
        except Exception as e:
            print(f"Error when saving {name} to cache: {e}")
            os.remove(cache_file)
            with open(cache_file, "wb") as f:
                breakpoint()
        return result


class UCIRepoDataset(AbstractDataset, torch.utils.data.Dataset):
    def __init__(self, name, split="train", seed=0):
        dataset = cached_fetch_ucirepo(name=name)
        self.X = dataset["data"]["features"]
        if (dataset["variables"]["type"] == "Categorical").any():
            categorical_feature_names = dataset["variables"][
                dataset["variables"]["type"] == "Categorical"
            ]["name"]
            for cfname in categorical_feature_names:
                if cfname in self.X.columns:
                    self.X[cfname] = (
                        self.X[cfname].astype("category").cat.codes
                    )  # map to integers
        self.X = self.X.to_numpy()
        self.y = dataset["data"]["targets"].to_numpy().squeeze()
        if len(self.y.shape) > 1 and self.y.shape[1] > 1:
            print("Oups, seems that you have more than one target")
            self.y = self.y[:, 0]  # take the first as target
        # normalize the data
        min_per_col = self.X.min(axis=0, keepdims=True)  # 1, D
        max_per_col = self.X.max(axis=0, keepdims=True)  # 1, D
        self.X = (
            (self.X - min_per_col) / (max_per_col - min_per_col) 
        )  # inputs are in [0, 1]
        self.y = (self.y - self.y.min()) / (
            self.y.max() - self.y.min()
        )  # outputs are in [0, 1]
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.33, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=seed)
        if split == "train":
            self.X = X_train
            self.y = y_train
        elif split == "val":    
            self.X = X_val
            self.y = y_val
        elif split == "test":   
            self.X = X_test
            self.y = y_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_feature_dim(self):
        return self.X.shape[1]

    def get_target_dim(self):
        return 1

    def upper_bound(self):
        return 1

    def lower_bound(self):
        return 0


ucireponames = dict(
    abalone="Abalone",
    concrete="Concrete Compressive Strength",
    energy="Energy Efficiency",
    power="Combined Cycle Power Plant",
    parkinsons="Parkinsons Telemonitoring",
    liver="Liver Disorders",
)


def get_dataset(dataset_name, **kwargs):
    if dataset_name == "bishop_toy":
        return BishopToy(**kwargs)
    elif dataset_name in ucireponames:
        return UCIRepoDataset(ucireponames[dataset_name])


# # load once
# for ucireponame in ucireponames:
#     print(f"Fetching {ucireponame}")
#     ds = UCIRepoDataset(ucireponame)
#     print(f"  {ucireponame} has {len(ds)} samples")
#     print(f"  Feature dimension: {ds.get_feature_dim()}")
#     print(f"  Target dimension: {ds.get_target_dim()}")

# # fetch dataset
# fetch_ucirepo(name="Abalone")
# fetch_ucirepo(name="Concrete Compressive Strength")
# fetch_ucirepo(name="Energy Efficiency")
# fetch_ucirepo(name="Combined Cycle Power Plant")
# fetch_ucirepo(name="Parkinsons Telemonitoring")
# fetch_ucirepo(name="Liver Disorders")

# # these don't work
# fetch_ucirepo(name="boston housing")
# fetch_ucirepo(name="Yacht Hydrodynamics")
# fetch_ucirepo(name="Year Prediction MSD")

# # this is constant!!!
# fetch_ucirepo(name="Challenger USA Space Shuttle O-Ring")

# also add sklearn datasets: california housing, diabetes
# also add boston housing dataset

# ------------------------------
# COMPARISON AGAINST MCDROP

# Concrete Compressive Strength
# Energy Efficiency
# Year Prediction MSD
# boston housing
# Combined Cycle Power Plant
# Yacht Hydrodynamics

# extra:
# Parkinsons Telemonitoring
# abalone
# liver disorders

# not included:
# Kin8nm 8,192 8 0.10 ±0.00 0.10 ±0.00 0.10 ±0.00 0.90 ±0.01 0.90 ±0.01 0.95 ±0.01
# Naval Propulsion 11,934 16 0.01 ±0.00 0.01 ±0.00 0.01 ±0.00 3.73 ±0.12 3.73 ±0.01 3.80 ±0.01
# Protein Structure 45,730 9 4.84 ±0.03 4.73 ±0.01 4.36 ±0.01 -2.99 ±0.01 -2.97 ±0.00 -2.89 ±0.00
# Wine Quality Red 1,599 11 0.65 ±0.01 0.64 ±0.01 0.62 ±0.01 -0.98 ±0.01 -0.97 ±0.01 -0.93 ±0.01
# ------------------------------
