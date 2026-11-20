import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_datasets(csv_path: str, seed: int = 42):
    df = pd.read_csv(csv_path)
    drop_cols = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    y = df["diagnosis"].map({"M": 1.0, "B": 0.0}).values
    X = df.drop(columns=["diagnosis"]).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
    )
    return train_ds, test_ds, X_train.shape[1]
