import torch

def Sampler(dataset, world_size, glob_rank, shuffle, **kwargs):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=glob_rank, shuffle=shuffle)

    return sampler
