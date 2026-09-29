import torch

def Optimizer(model_param,**kwargs):
    optimizer = torch.optim.AdamW(model_param,**kwargs)

    return optimizer
