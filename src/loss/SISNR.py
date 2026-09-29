import torch
from itertools import permutations

def sisnr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))


def calculate_sisnr_loss(ests, egs, pit=True, **kwargs):
    # spks x n x S
    refs = egs
    num_spks = len(refs)

    # P x N
    N = egs[0].size(0)
    if pit:
        num_spks = len(egs)
        batch_size = egs[0].shape[0]
        perms = list(permutations(range(num_spks)))
        
        all_perm_scores = []
        for p in perms:
            current_p_score = sum([sisnr(ests[s], egs[t]) for s, t in enumerate(p)]) / num_spks
            all_perm_scores.append(current_p_score)
        
        sisnr_mat = torch.stack(all_perm_scores, dim=0)
        
        max_per_sample, _ = torch.max(sisnr_mat, dim=0)
        loss = -torch.mean(max_per_sample)
        return loss

    else:
        sisnr_mat = torch.stack(
            [sisnr(ests[spk],refs[spk]) for spk in range(num_spks)])
        # si-snr
        return -torch.sum(sisnr_mat) / (num_spks*N)


def Loss(ests, egs, pit=False, **kwargs):
    """
    loss_tensor: final value of loss
    loss_dict: values of loss component
    """
    loss_tensor = calculate_sisnr_loss(ests, egs, pit, **kwargs)
    loss_dict = {"SISNR Loss": loss_tensor.item(),
                 }
    return loss_tensor, loss_dict
