import torch.nn as nn


def build_model(in_features: int = 30):
    model = nn.Linear(in_features, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    return model, loss_fn
