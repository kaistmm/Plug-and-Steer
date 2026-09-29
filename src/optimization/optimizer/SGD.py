import torch

def Optimizer(model_param,**kwargs):
    optimizer = torch.optim.SGD(model_param,**kwargs)

    return optimizer
