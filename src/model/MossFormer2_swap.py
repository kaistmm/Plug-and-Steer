import torch.nn.functional as F
from src.model.mossformer2.mossformer2_swap import MossFormer


'''
        in_channels=512,
        out_channels=512,
        num_blocks=24,
        kernel_size=16,
        norm="ln",
        num_spks=2,
        skip_around_intra=True,
        use_global_pos_enc=True,
        max_length=20000,
'''

def Main_Model(**kwargs):
    model = MossFormer(**kwargs)
    return model

