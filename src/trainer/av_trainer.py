"""The residual-adapter AV-TSE baselines (Sec. 3.3 of the paper).

Used for both alternatives the paper compares LSM against:

  + Residual (frozen)   backbone frozen, visual adapter learns f' = f + Adapter(v)
  + Residual (fulltune) backbone unfrozen at a lower lr

The difference between the two lives entirely in the model config
(`freeze: true/false`) and the optimizer's parameter groups, so one trainer
covers both. There is no gate here -- the target is routed by the adapter, and
the loss is plain SI-SNR against the reference in its given order.
"""

import torch

from .ddp_trainer_skeleton import DDP_Trainer_Skeleton
from .av_common import align_prediction_order


class AV_Trainer(DDP_Trainer_Skeleton):
    def __init__(self, config, proc, train_loader, val_loader):
        super().__init__(config, proc, train_loader, val_loader)
        self.sample_rate = config['datasets']['validation']['dataset']['config']['sample_rate']
        self.eval_metrics = config['trainer'].get('eval_metrics', False)
        if self.eval_metrics:
            from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
            from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment
            self.dnsmos = DeepNoiseSuppressionMeanOpinionScore(self.sample_rate, False,
                                                               device=self.device)
            self.nisqa = NonIntrusiveSpeechQualityAssessment(self.sample_rate)

    def _compute_loss(self, data):
        mix, ref, visual = data
        mix = mix.cuda(self.device)
        ref = [r.cuda(self.device) for r in ref]

        with torch.autocast("cuda", dtype=self.precision_dtype, enabled=self.use_amp):
            output = self.model(mix, visual.cuda(self.device))
            loss, loss_dict = self.loss(output, ref, **self.loss_config)
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
    return AV_Trainer(config, proc, train_loader, val_loader)
