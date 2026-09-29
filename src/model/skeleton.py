"""Template for a custom model module.

The framework imports `Main_Model` from `src.model.<name>` where <name> is
`model.name` in the config, and calls it with `model.config` as kwargs.
"""

import torch
from torch import nn


class YourModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        raise NotImplementedError("Define your modules here.")

    def forward(self, x):
        # x: (B, T) -> list of (B, T), one per source
        raise NotImplementedError


def Main_Model(**kwargs):
    return YourModel(**kwargs)
