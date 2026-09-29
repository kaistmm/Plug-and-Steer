"""Helpers shared by the audio-visual trainers.

These used to be copy-pasted verbatim into every av_*_trainer file, which is how
the perturbation key mismatch below survived unnoticed.
"""

from itertools import permutations

import torch
import torch.nn.functional as F


def sisnr(x, s, eps=1e-8):
    """Scale-invariant SNR, per item. x, s: (B, T) -> (B,)"""
    if x.shape != s.shape:
        raise RuntimeError(f"Dimension mismatch when calculating si-snr, {x.shape} vs {s.shape}")

    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)

    dot = torch.sum(x_zm * s_zm, dim=-1, keepdim=True)
    s_energy = torch.sum(s_zm ** 2, dim=-1, keepdim=True) + eps
    proj = dot * s_zm / s_energy
    e_noise = x_zm - proj

    target_energy = torch.sum(proj ** 2, dim=-1)
    noise_energy = torch.sum(e_noise ** 2, dim=-1) + eps
    return 10 * torch.log10(target_energy / noise_energy + eps)


def align_prediction_order(ests, egs, output_orders=False):
    """Reorder `ests` to the permutation that best matches `egs` (per item).

    When `output_orders` is set, also return a (B, T) mask that is 1 for items
    whose best permutation was the swap -- this is the pseudo-label g-bar of
    Eq. 2 in the paper.
    """
    num_spks = len(egs)
    batch_size, T = egs[0].shape[0], egs[0].shape[1]
    device = egs[0].device

    perms = list(permutations(range(num_spks)))
    scores = torch.stack([
        sum(sisnr(ests[s], egs[t]) for s, t in enumerate(p)) / num_spks
        for p in perms
    ], dim=0)
    _, max_indices = torch.max(scores, dim=0)

    ordered = [torch.zeros_like(e) for e in ests]
    order_mask = torch.zeros((batch_size, T), device=device) if output_orders else None

    for b in range(batch_size):
        best_p = perms[max_indices[b]]
        if output_orders and best_p == (1, 0):
            order_mask[b, :] = 1.0
        for s, t in enumerate(best_p):
            ordered[t][b] = ests[s][b]

    return (ordered, order_mask) if output_orders else ordered


def random_mask_like(x, min_len, max_len, prob=0.5):
    """(B, T) 0/1 mask built from random-length intervals, each flipped w.p. `prob`."""
    B, T = x.shape
    mask = torch.zeros(B, T, device=x.device)
    for b in range(B):
        t = 0
        while t < T:
            remaining = T - t
            if remaining <= min_len:
                seg_len = remaining
            else:
                hi = min(max_len, remaining)
                seg_len = torch.randint(min_len, hi + 1, (1,), device=x.device).item()
            if torch.rand(1, device=x.device).item() < prob:
                mask[b, t:t + seg_len] = 1.0
            t += seg_len
    return mask


# Config key -> the key the model and the label generator both read.
PERTURBATIONS = {"swap": "swap_mask", "zero": "zero_mask", "addition": "addition_mask"}


def get_perturbation_cfg(x, cfg):
    """Build the per-batch perturbation masks described by `trainer.perturbation`."""
    out = {}
    for name, key in PERTURBATIONS.items():
        if name in cfg:
            c = cfg[name]
            out[key] = random_mask_like(x, c['min_len'], c['max_len'], c['prob']) * c['scaling']
    return out


def generate_gate_label(gate, order_mask, perturbation_cfg):
    """Frame-wise BCE target for the gate.

    A swap perturbation flips which channel currently holds the target, so the
    gate has to flip with it. NOTE: the lookup key here is `swap_mask`, matching
    what get_perturbation_cfg writes -- the original code checked for `swap`,
    which is the CONFIG-level name and never appears in the built dict, so the
    swap perturbation silently never reached the label.
    """
    T = gate.shape[1]
    swap_mask = perturbation_cfg.get("swap_mask", None)
    if swap_mask is not None:
        gate_label = order_mask * (1 - swap_mask) + (1 - order_mask) * swap_mask
    else:
        gate_label = order_mask

    gate_label = gate_label.float()
    if gate_label.shape[1] != T:
        gate_label = F.interpolate(gate_label.unsqueeze(1), size=T, mode="nearest").squeeze(1)
    return gate_label
