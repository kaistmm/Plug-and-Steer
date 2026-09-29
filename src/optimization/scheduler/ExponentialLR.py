import torch

def Scheduler(optimizer,**kwargs):
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,**kwargs)

    return scheduler
