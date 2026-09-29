import torch.nn.functional as F
from src.model.tfgridnet.tfgridnetv1_separator_av import TFGridNet_AV



def Main_Model(**kwargs):
    model = TFGridNet_AV(**kwargs)
    return model

