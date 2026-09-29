import torch.nn.functional as F
from src.model.spmamba.SPMamba import SPMamba



def Main_Model(**kwargs):
    model = SPMamba(**kwargs)
    return model

