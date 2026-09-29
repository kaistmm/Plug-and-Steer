"""MossFormer2 with a visual residual adapter (stage 3, the `+ Residual` rows).

Serves both adaptation strategies, selected by `freeze`, the same way the other
three backbones do:

    freeze: true    partial adaptation -- backbone frozen, adapter trained
    freeze: false   full fine-tuning   -- backbone trained too, at a lower lr set
                                          through `optim.param_groups`

Until 2026-08-28 these were two classes, `MossFormer2_av` and `MossFormer2_av_res`,
because the frozen variant detached the graph at the adapter and carried an unused
`W`. Detaching is now driven by `freeze` (with the backbone frozen it changes no
gradient), and `W` is gone -- the residual adapter rewrites the feature rather than
routing it, so a steering matrix has no role here.
"""

from src.model.mossformer2.mossformer2_av import MossFormer2_AV


def Main_Model(**kwargs):
    return MossFormer2_AV(**kwargs)
