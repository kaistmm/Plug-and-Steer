"""Stage 1 of Plug-and-Steer: training the Latent Steering Matrix (LSM).

The backbone is frozen and only W is learned. The target is the frozen
backbone's OWN output with the two channels exchanged, so W is being asked to
perform the permutation in latent space:

    ref  = [AO(mix)[1], AO(mix)[0]]        (forced-swap target, g = 1)
    out  = AO(mix, transform=True)         (f' = (I + W) f)

MODEL SELECTION. `best.pt` is chosen on the loss against that swapped
prediction, NOT on PIT SI-SNR against ground truth. PIT is invariant to
permutation, i.e. blind to the one thing W exists to do -- under a PIT
criterion W = 0 (no swap at all) scores best. The PIT number is still computed
and logged, since it is what Fig. 3's preservation rate is derived from, but it
does not drive selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ddp_trainer_skeleton import DDP_Trainer_Skeleton


class Basic_Separation_Trainer(DDP_Trainer_Skeleton):
    def __init__(self, config, proc, train_loader, val_loader):
        super().__init__(config, proc, train_loader, val_loader)
        self.sample_rate = config['datasets']['validation']['dataset']['config']['sample_rate']
        tcfg = config['trainer']
        self.reg = tcfg.get('regularization', None)
        self.reg_w = tcfg.get('regularization_weight', 0.0)

    def _swapped_reference(self, mix):
        """The frozen backbone's output with channels exchanged."""
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            out = self.model(mix, transform=False)
            ref = [out[1].detach(), out[0].detach()]
        if was_training:
            self.model.train()
        return ref

    def _regularizers(self):
        W = self.model.module.W
        I = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
        orthog = F.mse_loss(torch.mm((I + W).t(), (I + W)), I)
        inverse = F.mse_loss(W * 2 + torch.mm(W, W), torch.zeros_like(W))
        return orthog, inverse

    def _compute_loss(self, data):
        mix = data[0].cuda(self.device)
        ref = self._swapped_reference(mix)

        with torch.autocast("cuda", dtype=self.precision_dtype, enabled=self.use_amp):
            output = self.model(mix, transform=True)
            loss, loss_dict = self.loss(output, ref, **self.loss_config)

            orthog_loss, inverse_loss = self._regularizers()
            if self.reg == "orthogonal":
                loss = loss + orthog_loss * self.reg_w
            elif self.reg == "inverse":
                loss = loss + inverse_loss * self.reg_w

        loss_dict["Orthogonal Loss"] = orthog_loss.item()
        loss_dict["Inverse Loss"] = inverse_loss.item()
        return loss, loss_dict

    def train(self):
        return self.run_train_epoch(self._compute_loss)

    def validate(self):
        total_pred_loss = 0.0     # vs the swapped backbone output -> selection
        total_pit_loss = 0.0      # vs ground truth, PIT           -> reporting
        total_sample = 0

        self.model.eval()
        with torch.no_grad():
            for batch, (mix, ref) in enumerate(self.val_loader):
                mix = mix.cuda(self.device)
                ref = [r.cuda(self.device) for r in ref]
                pred_ref = self._swapped_reference(mix)

                with torch.autocast("cuda", dtype=self.precision_dtype,
                                    enabled=self.precision_inference):
                    output = self.model(mix, transform=True)
                    pred_loss, _ = self.loss(output, pred_ref, **self.loss_config)
                    pit_loss, _ = self.loss(output, ref, pit=True, **self.loss_config)

                total_pred_loss += pred_loss.item() * mix.shape[0]
                total_pit_loss += pit_loss.item() * mix.shape[0]
                total_sample += mix.shape[0]

                if self.tb and batch < self.num_test_log:
                    tag = f"Sample{batch:02d}"
                    self.logging_audio(tag, 'Mix', mix, self.epoch, self.sample_rate)
                    for i in range(len(ref)):
                        self.logging_audio(tag, f'S{i+1}_GT', ref[i], self.epoch, self.sample_rate)
                        self.logging_audio(tag, f'S{i+1}_pt', pred_ref[i], self.epoch, self.sample_rate)
                        self.logging_audio(tag, f'S{i+1}_pred', output[i], self.epoch, self.sample_rate)

        pit_avg = self.calculate_ddp_loss(total_pit_loss, total_sample)
        if self.tb:
            self.logging("Validation", "Average Loss(PIT vs GT)", pit_avg, self.epoch)

        return self.calculate_ddp_loss(total_pred_loss, total_sample)


def Trainer(config, proc, train_loader, val_loader):
    return Basic_Separation_Trainer(config, proc, train_loader, val_loader)
