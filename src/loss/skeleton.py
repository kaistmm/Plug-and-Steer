"""Template for a custom loss module.

The framework imports the symbol `Loss` from `src.loss.<name>` where <name> is
`trainer.loss.name` in the config. It may be either a function (as here) or a
class -- if it is a class, `trainer.loss.config` is passed to its constructor
instead of to each call.
"""

import torch


def Loss(ests, egs, **kwargs):
    """
    Args:
        ests: list of estimated sources, each (B, T)
        egs:  list of reference sources, each (B, T)
        kwargs: whatever `trainer.loss.config` holds
    Returns:
        (loss_tensor, loss_dict) -- loss_dict is scalars for tensorboard
    """
    raise NotImplementedError("Implement your loss here.")
