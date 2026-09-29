import torch

def Optimizer(model_param,**kwargs):
    optimizer = torch.optim.Adam(model_param,**kwargs)

    return optimizer
