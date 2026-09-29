"""Cosine decay with warmup, stepped per optimizer step."""

import transformers


def Scheduler(optimizer, **kwargs):
    """Cosine decay with warmup and optional hard restarts.

    HuggingFace's schedule is a LambdaLR: it scales each param group from that
    group's own base lr, so a config using `optim.param_groups` keeps the ratio
    between groups for the whole run. A scheduler that writes one lr into every
    group -- as the hand-rolled CosineAnnealingWarmupRestarts that used to sit
    here did -- would silently undo that. It was never reachable and is gone;
    see archive/ if it is ever needed.

    kwargs: num_warmup_steps, num_training_steps, num_cycles.
    """
    return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, **kwargs)
