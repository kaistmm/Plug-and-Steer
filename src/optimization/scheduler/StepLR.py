import torch

def Scheduler(optimizer,**kwargs):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,**kwargs)

    return scheduler
