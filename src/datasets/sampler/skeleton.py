"""Template for a custom sampler module.

`Sampler` receives the dataset, the sampler `config` block, and every key of the
DDP `proc` dict (world_size, glob_rank, curr_gpu, ...).
"""


def Sampler(dataset, world_size, glob_rank, shuffle, **kwargs):
    raise NotImplementedError("Return a torch.utils.data.Sampler.")
