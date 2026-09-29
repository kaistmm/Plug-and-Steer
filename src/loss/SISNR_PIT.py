import torch
from itertools import permutations

def sisnr(x, s, eps=1e-8):
    """
    input:
        mix: B x L
        est: B x L
    output: B
    """
    if x.shape != s.shape:
        raise RuntimeError(
            f"Dimension mismatch when calculate si-snr, {x.shape} vs {s.shape}")

    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)

    dot_product = torch.sum(x_zm * s_zm, dim=-1, keepdim=True)

    s_target_energy = torch.sum(s_zm ** 2, dim=-1, keepdim=True) + eps

    proj = dot_product * s_zm / s_target_energy

    e_noise = x_zm - proj


    target_energy = torch.sum(proj ** 2, dim=-1)
    noise_energy = torch.sum(e_noise ** 2, dim=-1) + eps

    sisnr_value = target_energy / noise_energy

    sisnr_value = 10 * torch.log10(sisnr_value + eps)

    return sisnr_value


# PIT code with a minor error. not used anymore
def _calculate_sisnr_loss(ests, egs, **kwargs):
    # spks x n x S
    refs = egs
    num_spks = len(refs)

    def sisnr_loss(permute):
        # for one permute
        return sum(
            [sisnr(ests[s], refs[t])
             for s, t in enumerate(permute)]) / len(permute)
             # average the value

    # P x N
    N = egs[0].size(0)
    sisnr_mat = torch.stack(
        [sisnr_loss(p) for p in permutations(range(num_spks))])
    max_perutt, _ = torch.max(sisnr_mat, dim=0)
    # si-snr
    return -torch.sum(max_perutt) / N


def calculate_sisnr_loss(ests, egs, **kwargs):
    """
    ests: [(B, T), (B, T), ...]
    egs:  [(B, T), (B, T), ...] - gt
    """
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

def Loss(ests, egs, **kwargs):
    """
    loss_tensor: final value of loss
    loss_dict: values of loss component
    """
    loss_tensor = calculate_sisnr_loss(ests, egs, **kwargs)
    loss_dict = {"SISNR Loss": loss_tensor.item(),
                 }
    return loss_tensor, loss_dict
