"""Template for a custom LR scheduler module.

`scheduler.call_step` decides whether the framework steps it per epoch or per
optimizer update; `scheduler.loss_pass` decides whether validation loss is
passed to step() (needed by ReduceLROnPlateau).
"""


def Scheduler(optimizer, **kwargs):
    raise NotImplementedError("Return a torch.optim.lr_scheduler object.")
