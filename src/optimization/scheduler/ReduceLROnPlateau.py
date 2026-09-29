"""ReduceLROnPlateau with an improvement tolerance that works on negative losses.

Behaviour is torch's, unchanged, except for two defaults.

WHY THE DEFAULTS DIFFER FROM TORCH
----------------------------------
torch defaults to `threshold_mode='rel'`, whose test for mode='min' is

    a < best * (1 - threshold)

That is written for positive metrics. Our losses are negative SI-SNR, and with a
negative `best` the multiplication moves the bar the WRONG WAY:

    best = -14.9, threshold = 1e-4  ->  bar = -14.8985

-14.8985 is *higher* than -14.9, so a value that is genuinely worse than the
current best (say -14.899) still registers as an improvement. Patience is reset
by noise, the learning rate never decays, and the run drags on indefinitely --
which is exactly what audio-only pre-training did here.

`threshold_mode='abs'` uses

    a < best - threshold

which behaves identically for positive and negative metrics: the candidate must
beat the best by at least `threshold`.

CHOOSING `threshold`
--------------------
Set it to the smallest improvement you would actually report. Results are
quoted to two decimals in dB, so 0.01 dB is the natural floor -- anything below
that is not a result, and should not buy another `patience` epochs of training.
Raise it (0.05 - 0.1) to converge faster when you only need a ballpark number.
"""

import torch


def Scheduler(optimizer, **kwargs):
    # Only fill in what the config did not specify, so an explicit
    # threshold_mode: rel is still honoured if someone really wants it.
    kwargs.setdefault('threshold_mode', 'abs')
    kwargs.setdefault('threshold', 0.01)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
