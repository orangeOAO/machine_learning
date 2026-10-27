from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"

class TemperatureDataset(Dataset):
    def __init__(self, values, seq_len, start, end):
        self.x = []
        self.y = []
        for i in range(start, end):
            self.x.append(values[i - seq_len:i])
            self.y.append(values[i])
        self.x = torch.tensor(np.array(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
def get_dataloaders(seq_len=30, batch_size=32):
    path = Path("data/daily-min-temperatures.csv")
    path.parent.mkdir(exist_ok=True)

    if not path.exists():
        print("Downloading dataset...")
        urlretrieve(DATA_URL, path)

    # 將一維資料轉換為二維陣列（時間步數，特徵數），其中 -1 代表自動計算時間步數。
    values = pd.read_csv(path)["Temp"].to_numpy(dtype=np.float32).reshape(-1, 1)
    n = len(values)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    if not 1 <= seq_len < train_end:
        raise ValueError("seq_len must be smaller than the training data length")

    # Only use training data to calculate normalization statistics.
    mean = values[:train_end].mean()
    std = values[:train_end].std()
    values = (values - mean) / std

    # Targets belong to separate time periods. Val/test can use earlier history.
    train_set = TemperatureDataset(values, seq_len, seq_len, train_end)
    val_set = TemperatureDataset(values, seq_len, train_end, val_end)
    test_set = TemperatureDataset(values, seq_len, val_end, n)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )

    print(
        f"Dataset: {n} steps | Train: {len(train_set)} | "
        f"Val: {len(val_set)} | Test: {len(test_set)}"
    )

    return train_loader, val_loader, test_loader, mean, std


if __name__ == "__main__":
    train_loader, val_loader, test_loader, mean, std = get_dataloaders()

    x, y = next(iter(train_loader))

    print("\n--- Dataset Example ---")
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))
    print("Input batch shape:", x.shape)
    print("Target batch shape:", y.shape)

    sample_x, sample_y = train_loader.dataset[0]

    print(
        "First 5 days (°C):",
        (sample_x[:5, 0].numpy() * std + mean).round(2)
    )
    print(
        "Next-day target (°C):",
        round(sample_y.item() * std + mean, 2)
    )