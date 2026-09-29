import torch.nn.functional as F
from src.model.tfgridnet.tfgridnetv1_separator import TFGridNet



def Main_Model(**kwargs):
    model = TFGridNet(**kwargs)
    return model

