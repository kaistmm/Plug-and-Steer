"""Template for a custom dataset module.

The framework imports `Datasets` from `src.datasets.dataset.<name>` where
<name> is `datasets.{train,validation}.dataset.name`, and calls it with the
matching `config` block as kwargs.
"""

from torch.utils.data import Dataset


class Custom_Dataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        raise NotImplementedError("Load your file lists here.")

    def __getitem__(self, index):
        # Separation trainers expect (mix, [ref_1, ..., ref_n]);
        # audio-visual trainers expect (mix, [ref_1, ...], visual).
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


def Datasets(**kwargs):
    return Custom_Dataset(**kwargs)
