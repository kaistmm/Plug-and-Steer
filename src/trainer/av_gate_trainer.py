"""Stage 2 of Plug-and-Steer: training the visual steering module.

The backbone and the LSM are frozen; only the visual steering module learns.
It predicts a gate g_t that decides how much of W to apply, which is what routes
the target speaker to channel 0.

    L_total = sisnr_loss_weight * L_SI-SNR(s_LSM, s) + gate_loss_weight * L_BCE(g_t, g_bar)

This single file replaces eight near-identical trainers from the original
repository. They differed only in these two weights and in the gate's shape,
so both are configuration now:

    trainer.sisnr_loss_weight   trainer.gate_loss_weight   trainer.gate_mode
    ------------------------------------------------------------------------
    1.0                         w                          framewise   L_BCE + L_SI-SNR  (paper's default)
    0.0                         w                          framewise   L_BCE only
    1.0                         0.0                        framewise   L_SI-SNR only
    0.1                         w                          framewise   the "lowsnr" variant
    (the same three)                                       global      the "swap1" variant, utterance-level gate

`gate_mode: global` predicts one gate for the whole utterance instead of one per
frame; it pairs with the *_global / _v4 model variants.
"""

import torch
import torch.nn.functional as F
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment

from .ddp_trainer_skeleton import DDP_Trainer_Skeleton
from .av_common import align_prediction_order, get_perturbation_cfg, generate_gate_label


class AV_Gate_Trainer(DDP_Trainer_Skeleton):
    def __init__(self, config, proc, train_loader, val_loader):
        super().__init__(config, proc, train_loader, val_loader)
        tcfg = config['trainer']
        self.sample_rate = config['datasets']['validation']['dataset']['config']['sample_rate']
        self.perturbation = tcfg.get('perturbation', {})
        self.gate_w = tcfg.get('gate_loss_weight', 1.0)
        self.sisnr_w = tcfg.get('sisnr_loss_weight', 1.0)
        self.gate_mode = tcfg.get('gate_mode', 'framewise')
        assert self.gate_mode in ('framewise', 'global'), \
            f"gate_mode must be 'framewise' or 'global', got {self.gate_mode}"

        self.eval_metrics = tcfg.get('eval_metrics', True)
        if self.eval_metrics:
            self.dnsmos = DeepNoiseSuppressionMeanOpinionScore(self.sample_rate, False,
                                                               device=self.device)
            self.nisqa = NonIntrusiveSpeechQualityAssessment(self.sample_rate)

    def _gate_bce(self, gate, gate_label):
        if self.gate_mode == 'global':
            # One gate per utterance: collapse both the prediction and the
            # frame-wise label to a single value.
            return F.binary_cross_entropy_with_logits(
                gate.squeeze(-1).squeeze(-1), gate_label[:, 0].squeeze(-1))
        return F.binary_cross_entropy_with_logits(gate.squeeze(-1), gate_label)

    def _compute_loss(self, data):
        mix, ref, visual = data
        mix = mix.cuda(self.device)
        ref = [r.cuda(self.device) for r in ref]
        visual = visual.cuda(self.device)

        # Which channel currently carries the target, per item: the pseudo-label.
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            pred_ref = self.model(mix)
            _, order = align_prediction_order([f.detach() for f in pred_ref], ref,
                                              output_orders=True)
        if was_training:
            self.model.train()

        perturbation_cfg = get_perturbation_cfg(mix, self.perturbation)
        perturbation_cfg["order_mask"] = order.clone()

        with torch.autocast("cuda", dtype=self.precision_dtype, enabled=self.use_amp):
            output, gate = self.model(mix, visual,
                                      perturbation_cfg=perturbation_cfg, output_gate=True)
            sisnr_loss, loss_dict = self.loss(output, ref, **self.loss_config)
            gate_label = generate_gate_label(gate, order, perturbation_cfg)
            gate_loss = self._gate_bce(gate, gate_label)
            loss = self.sisnr_w * sisnr_loss + self.gate_w * gate_loss

        loss_dict["Gate Loss"] = gate_loss.item()
        loss_dict["SISNR Loss"] = sisnr_loss.item()
        return loss, loss_dict

    def train(self):
        return self.run_train_epoch(self._compute_loss)

    def validate(self):
        total_loss = total_pred_loss = total_dnsmos = total_nisqa = 0.0
        total_sample = 0

        self.model.eval()
        with torch.no_grad():
            for batch, (mix, ref, visual) in enumerate(self.val_loader):
                mix = mix.cuda(self.device)
                ref = [r.cuda(self.device) for r in ref]

                # AO output, permuted to match the reference, as the "what the
                # frozen backbone alone could do" comparison point.
                pred_ref = align_prediction_order([f.clone() for f in self.model(mix)], ref)
                pred_ref, ref = [pred_ref[0]], [ref[0]]

                with torch.autocast("cuda", dtype=self.precision_dtype,
                                    enabled=self.precision_inference):
                    output = self.model(mix, visual.cuda(self.device))
                    output = [output[0]]
                    loss, _ = self.loss(output, ref, **self.loss_config)
                    pred_loss, _ = self.loss(output, pred_ref, **self.loss_config)

                n = mix.shape[0]
                total_loss += loss.item() * n
                total_pred_loss += pred_loss.item() * n
                total_sample += n

                if self.eval_metrics:
                    peak = torch.max(torch.abs(output[0])).clamp(min=1e-8)
                    est = output[0] / peak
                    total_dnsmos += float(self.dnsmos(est)[3].item()) * n
                    total_nisqa += float(self.nisqa(est)[0].item()) * n

                if self.tb and batch < self.num_test_log:
                    tag = f"Sample{batch:02d}"
                    self.logging_audio(tag, 'Mix', mix, self.epoch, self.sample_rate)
                    for i in range(len(ref)):
                        self.logging_audio(tag, f'S{i+1}_GT', ref[i], self.epoch, self.sample_rate)
                        self.logging_audio(tag, f'S{i+1}_pt', pred_ref[i], self.epoch, self.sample_rate)
                        self.logging_audio(tag, f'S{i+1}_pred', output[i], self.epoch, self.sample_rate)

        pred_avg = self.calculate_ddp_loss(total_pred_loss, total_sample)
        if self.tb:
            self.logging("Validation", "Average Loss(vs prediction)", pred_avg, self.epoch)
            if self.eval_metrics:
                self.logging("Validation", "DNSMOS",
                             self.calculate_ddp_loss(total_dnsmos, total_sample), self.epoch)
                self.logging("Validation", "NISQA",
                             self.calculate_ddp_loss(total_nisqa, total_sample), self.epoch)

        return self.calculate_ddp_loss(total_loss, total_sample)


def Trainer(config, proc, train_loader, val_loader):
    return AV_Gate_Trainer(config, proc, train_loader, val_loader)
