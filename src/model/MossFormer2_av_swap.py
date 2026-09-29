"""MossFormer2 steered by a frame-wise visual gate (stage 3).

Every MossFormer2 AV-TSE number in the paper came from this model. It was called
`MossFormer2_av_v3` while the AV variants were still numbered, and a second
wrapper named `MossFormer2_av_v3_global` imported the very same implementation --
they were never two models. Both names are gone; see archive/ for the history.

Averaging the gate over an utterance is an inference-time choice, not a different
model. The variant that *trained* a single utterance-level gate lives at
archive/src/model/MossFormer2_av_v4.py; it is not the paper's method.
"""

from src.model.mossformer2.mossformer2_av_swap import MossFormer2_AV


def Main_Model(**kwargs):
    return MossFormer2_AV(**kwargs)
