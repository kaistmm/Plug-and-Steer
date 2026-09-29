"""Audio-only separation training (the AO backbones the paper starts from).

Loss is PIT SI-SNR against the reference sources; nothing here is audio-visual.
"""

import torch

from .ddp_trainer_skeleton import DDP_Trainer_Skeleton


class Basic_Separation_Trainer(DDP_Trainer_Skeleton):
    def __init__(self, config, proc, train_loader, val_loader):
        super().__init__(config, proc, train_loader, val_loader)
        self.sample_rate = config['datasets']['validation']['dataset']['config']['sample_rate']

    def _compute_loss(self, data):
        mix, ref = data
        mix = mix.cuda(self.device)
        ref = [r.cuda(self.device) for r in ref]

        with torch.autocast("cuda", dtype=self.precision_dtype, enabled=self.use_amp):
            output = self.model(mix)
            loss, loss_dict = self.loss(output, ref, **self.loss_config)
        return loss, loss_dict

    def train(self):
        return self.run_train_epoch(self._compute_loss)

    def validate(self):
        total_loss = 0.0
        total_sample = 0

        self.model.eval()
        with torch.no_grad():
            for batch, (mix, ref) in enumerate(self.val_loader):
                mix = mix.cuda(self.device)
                ref = [r.cuda(self.device) for r in ref]

                with torch.autocast("cuda", dtype=self.precision_dtype,
                                    enabled=self.precision_inference):
                    output = self.model(mix)
                    loss, _ = self.loss(output, ref, **self.loss_config)

                total_loss += loss.item() * mix.shape[0]
                total_sample += mix.shape[0]

                if self.tb and batch < self.num_test_log:
                    tag = f"Sample{batch:02d}"
                    self.logging_audio(tag, 'Mix', mix, self.epoch, self.sample_rate)
                    for i in range(len(ref)):
                        self.logging_audio(tag, f'S{i+1}_GT', ref[i], self.epoch, self.sample_rate)
                        self.logging_audio(tag, f'S{i+1}_pred', output[i], self.epoch, self.sample_rate)

        return self.calculate_ddp_loss(total_loss, total_sample)


def Trainer(config, proc, train_loader, val_loader):
    return Basic_Separation_Trainer(config, proc, train_loader, val_loader)
