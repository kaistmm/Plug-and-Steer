"""Base class for every trainer in this framework.

Responsibilities kept here (so no trainer has to re-implement them):
  * building model / optimizer / scheduler from config, single or multi-model
  * the epoch loop, checkpointing, resume and early stop
  * DDP-safe helpers for gradient accumulation and NaN handling

A subclass only has to implement train() and validate().
"""

import os
import time
import logging
import importlib
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from ruamel.yaml import YAML

NUM_TEST_LOG = 5


def torch_load(path, map_location='cpu'):
    """torch.load that behaves the same before and after torch 2.6.

    2.6 flipped the default of `weights_only` to True, which breaks every
    checkpoint here (they carry optimizer/scheduler state, not just tensors).
    Pinning it explicitly keeps one code path across versions.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:                      # torch < 1.13 has no such kwarg
        return torch.load(path, map_location=map_location)


class DDP_Trainer_Skeleton:
    def __init__(self, config, proc, train_loader, val_loader):
        self.config = config
        self.proc = proc
        self.device = self.proc['curr_gpu']
        self.dist = self.proc['dist']
        self.glob_rank = self.proc['glob_rank']

        if self.glob_rank == 0:
            self.logger = logging.getLogger(self.config['logging']['name'])
        else:
            self.logger = logging.getLogger('tmp_log')

        tcfg = self.config['trainer']
        self.max_epoch = tcfg['epoch']
        self.early_stop = tcfg['early_stop']
        self.clip_norm = tcfg['clip_norm']

        self.precision_mode = tcfg.get('precision_mode', 'fp32')
        auto_cast_dtypes = {"bf16-mixed": torch.bfloat16, "fp16-mixed": torch.float16}
        if self.precision_mode in auto_cast_dtypes:
            self.precision_dtype = auto_cast_dtypes[self.precision_mode]
            self.use_amp = True
            self.precision_inference = tcfg.get('precision_inference', False)
        else:
            self.precision_dtype = None
            self.use_amp = False
            self.precision_inference = False

        self.grad_accum_num = tcfg.get('grad_accum_num', 0)
        # How many numbered checkpoints to keep. best.pt is never pruned.
        # 0 keeps everything (the old behaviour, which reached 44 GB per run).
        self.keep_last_ckpt = tcfg.get('keep_last_ckpt', 3)

        self.tb = self.config['logging']['tensorboard']
        self.log_freq = self.config['logging']['log_freq']
        self.num_test_log = self.config['logging'].get('num_test_log', NUM_TEST_LOG)

        self.best_loss = np.inf
        self.last_loss = np.inf
        self.epoch = 0
        self.last_epoch = 0
        self.global_step = 0

        self.base_path = self.proc['base_path']
        self.ckpt_dir = os.path.join(self.base_path, 'ckpt')

        self.train_loader = train_loader
        self.val_loader = val_loader

        self._build_model()
        self.dist.barrier()
        self._build_optimizer()
        self._build_scheduler()
        self._build_loss()

        if self.glob_rank == 0:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            yaml = YAML(typ='rt')
            yaml.preserve_quotes = True
            with open(os.path.join(self.base_path, 'config.yaml'), 'w') as f:
                yaml.dump(config, f)

        self.dist.barrier()
        if self.config['checkpoint']['resume']['enable']:
            ckpt_path = self.search_checkpoint(self.ckpt_dir,
                                               self.config['checkpoint']['resume']['type'])
            if ckpt_path is not None:
                self.load_checkpoint(ckpt_path,
                                     only_model=self.config['checkpoint']['resume']['only_model'])
        self.dist.barrier()

        if self.tb and self.glob_rank == 0:
            self.tb_dir = self.base_path
            os.makedirs(self.tb_dir, exist_ok=True)
            self.writer = SummaryWriter(self.tb_dir)

    # ------------------------------------------------------------------ build

    @staticmethod
    def _has_batchnorm(model):
        bn = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        return any(isinstance(m, bn) for m in model.modules())

    def _build_single_model(self, model_cfg, label=None):
        prefix = f'[{label}] ' if label else ''
        _model = importlib.import_module(
            "src.model." + model_cfg['name']).__getattribute__("Main_Model")
        model = _model(**(model_cfg.get('config') or {})).cuda(self.device)

        if self.glob_rank == 0:
            self.logger.info(f'{prefix}{model}')
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(f'{prefix}Total Parameters: {total / 1e6:.3f}M')
            self.logger.info(f'{prefix}Trainable Parameters: {trainable / 1e6:.3f}M')

        if len(self.proc['gpus']) > 1 and self._has_batchnorm(model):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.logger.info(f'{prefix}BatchNorm found. Synchronize BatchNorm')

        find_unused = self.config.get('DDP', {}).get('find_unused_parameters', False)
        model = DDP(model, device_ids=[self.device], find_unused_parameters=find_unused)
        self.logger.info(f'rank: {self.glob_rank} {prefix}model built')
        return model

    def _build_model(self):
        if 'models' in self.config:
            self.models = {k: self._build_single_model(c, label=k)
                           for k, c in self.config['models'].items()}
        else:
            self.model = self._build_single_model(self.config['model'])

    def _param_groups(self, model, cfg):
        """Split a model's trainable parameters into optimizer groups.

        Without `optim.param_groups` every parameter goes into one group and the
        optimizer's own `lr` applies -- the behaviour this had before. With it, each
        entry names the *leaf* module names it claims and the learning rate they get:

            optim:
              config: {lr: 5.0e-4}
              param_groups:
                - name: adapter
                  modules: [visual_module, vis_module, compress, decompress]
                  lr: 5.0e-4
                - name: backbone        # no `modules`: everything not claimed above
                  lr: 5.0e-5

        The schedulers in `src/optimization/scheduler/` are LambdaLR-based, so they
        scale each group from its own base lr and the ratio between groups holds for
        the whole run.
        """
        spec = cfg.get('param_groups')
        if not spec:
            return model.parameters()

        named = list(model.named_parameters())
        trainable = {id(p) for _, p in named if p.requires_grad}
        modules = dict(model.named_modules())

        claimed, groups = set(), []
        rest_spec = None
        for entry in spec:
            if 'modules' not in entry:
                if rest_spec is not None:
                    raise ValueError('optim.param_groups: only one catch-all entry allowed')
                rest_spec = entry
                continue
            leaves = set(entry['modules'])
            ids = {id(p)
                   for path, mod in modules.items() if path and path.split('.')[-1] in leaves
                   for p in mod.parameters() if p.requires_grad}
            if not ids:
                raise ValueError(f"optim.param_groups: no trainable parameter under "
                                 f"{sorted(leaves)} (group '{entry.get('name')}')")
            if claimed & ids:
                raise ValueError(f"optim.param_groups: group '{entry.get('name')}' overlaps "
                                 f"an earlier group")
            claimed |= ids
            groups.append(dict(entry, params=[p for _, p in named if id(p) in ids]))

        rest = [p for _, p in named if p.requires_grad and id(p) not in claimed]
        if rest_spec is not None:
            groups.append(dict(rest_spec, params=rest))
        elif rest:
            raise ValueError(f'optim.param_groups: {len(rest)} trainable parameters match no '
                             f'group and there is no catch-all entry')

        assigned = sum(len(g['params']) for g in groups)
        if assigned != len(trainable):
            raise ValueError(f'optim.param_groups: {assigned} parameters assigned but '
                             f'{len(trainable)} are trainable')
        for g in groups:
            g.pop('modules', None)
            g.pop('name', None)
        return groups

    def _build_optimizer(self):
        def make(cfg, model_or_params):
            _o = importlib.import_module(
                "src.optimization.optimizer." + cfg['name']).__getattribute__("Optimizer")
            params = (self._param_groups(model_or_params, cfg)
                      if isinstance(model_or_params, torch.nn.Module) else model_or_params)
            return _o(params, **cfg['config'])

        if 'optims' in self.config:
            self.optimizers = {}
            for key, cfg in self.config['optims'].items():
                self.optimizers[key] = make(cfg, self.models[cfg.get('model', key)])
            self.logger.info(f'rank: {self.glob_rank} optimizers built ({list(self.optimizers)})')
        else:
            self.optimizer = make(self.config['optim'], self.model)
            n = len(self.optimizer.param_groups)
            self.logger.info(f'rank: {self.glob_rank} optimizer built'
                             + (f' ({n} param groups: '
                                + ', '.join(f"{g['lr']:g}" for g in self.optimizer.param_groups)
                                + ')' if n > 1 else ''))

    def _build_scheduler(self):
        def make(cfg, optim):
            _s = importlib.import_module(
                "src.optimization.scheduler." + cfg['name']).__getattribute__("Scheduler")
            return _s(optim, **cfg['config'])

        if 'schedulers' in self.config:
            self.schedulers = {}
            for key, cfg in self.config['schedulers'].items():
                self.schedulers[key] = make(cfg, self.optimizers[cfg.get('optimizer', key)])
            self.logger.info(f'rank: {self.glob_rank} schedulers built ({list(self.schedulers)})')
        else:
            self.scheduler = make(self.config['scheduler'], self.optimizer)
            self.logger.info(f'rank: {self.glob_rank} scheduler built')

    def _build_loss(self):
        loss_cfg = self.config['trainer']['loss']
        _loss = importlib.import_module(
            "src.loss." + loss_cfg['name']).__getattribute__("Loss")
        loss_config = loss_cfg.get('config') or {}
        if isinstance(_loss, type):          # class-based loss
            self.loss = _loss(**loss_config)
            self.loss_config = {}
        else:                                # function-based loss
            self.loss = _loss
            self.loss_config = loss_config

    # ------------------------------------------------- grad-accum / NaN utils

    @property
    def accum_steps(self):
        return 1 if not self.grad_accum_num else int(self.grad_accum_num)

    def no_sync_ctx(self, model, do_step):
        """Suppress DDP gradient all-reduce on non-stepping micro-batches.

        Without this the gradients are all-reduced once per micro-batch instead
        of once per optimizer step, i.e. `accum_steps` times more collective
        traffic for an identical result.
        """
        if do_step or not isinstance(model, DDP):
            return nullcontext()
        return model.no_sync()

    def any_rank(self, flag):
        """True on every rank if `flag` is True on any rank.

        Every skip decision taken inside the training loop -- a NaN loss, a bad
        gradient -- must be agreed on collectively. If one rank alone skips its
        backward, the other ranks block forever in the DDP all-reduce and the
        job dies on the NCCL timeout instead of on the actual error.
        """
        t = torch.tensor([1.0 if flag else 0.0], device=self.device)
        self.dist.all_reduce(t, op=self.dist.ReduceOp.SUM)
        return t.item() > 0

    # ----------------------------------------------------- shared train epoch

    def run_train_epoch(self, compute_loss):
        """One epoch of accumulate-and-step, shared by every single-model trainer.

        `compute_loss(batch)` returns `(loss_tensor, loss_dict)`; everything
        around it -- accumulation windows, DDP sync suppression, the collective
        NaN decision, clipping, per-iteration scheduling and logging -- is
        handled here so the trainers only describe their own forward pass.
        """
        accum = self.accum_steps
        num_iters = len(self.train_loader)
        num_updates_est = (num_iters + accum - 1) // accum

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss, valid_batches, nan_count = 0.0, 0, 0
        local_loss, local_valid, update_step = 0.0, 0, 0
        total_time, local_start_time = 0.0, time.time()
        grad_norm, loss_dict = torch.tensor(0.0), {}

        for batch, data in enumerate(self.train_loader):
            micro_step = batch % accum
            if micro_step == 0:
                window_start = batch
                # The final window is short whenever num_iters is not a multiple
                # of accum. Dividing it by `accum` anyway would silently shrink
                # that update's gradient, so scale by what the window holds.
                window_size = min(accum, num_iters - window_start)
                self.optimizer.zero_grad(set_to_none=True)

            is_last = (batch == num_iters - 1)
            do_step = ((micro_step + 1) == window_size) or is_last

            loss, loss_dict = compute_loss(data)

            bad = bool(torch.isnan(loss) or torch.isinf(loss))
            # Agreed across ranks: a rank that skipped backward on its own would
            # leave the others waiting in the gradient all-reduce forever.
            if self.any_rank(bad):
                nan_count += 1
                if self.glob_rank == 0:
                    self.logger.warning(f'Epoch {self.epoch}, batch {batch}: '
                                        f'NaN/Inf loss, skipping this window')
                del loss
                self.optimizer.zero_grad(set_to_none=True)
                continue

            total_loss += loss.item()
            local_loss += loss.item()
            valid_batches += 1
            local_valid += 1

            with self.no_sync_ctx(self.model, do_step):
                (loss / window_size).backward()

            if not do_step:
                continue

            clip = self.clip_norm if self.clip_norm != 0 else 1e10
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            if self.any_rank(bool(torch.isnan(grad_norm) or torch.isinf(grad_norm))):
                nan_count += 1
                if self.glob_rank == 0:
                    self.logger.warning(f'Epoch {self.epoch}, batch {batch}: '
                                        f'NaN/Inf gradient, skipping optimizer step')
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.step()
                if self.config['scheduler']['call_step'] == "iteration":
                    if self.config['scheduler']['loss_pass']:
                        raise NotImplementedError(
                            'loss_pass is not supported for iteration-level scheduling')
                    self.scheduler.step()

            update_step += 1
            self.global_step += 1

            if self.glob_rank == 0 and (update_step % self.log_freq == 0):
                exec_time = time.time() - local_start_time
                total_time += exec_time
                est = time.strftime("%H:%M:%S",
                                    time.gmtime((total_time / update_step) * num_updates_est))
                taken = time.strftime("%H:%M:%S", time.gmtime(total_time))
                avg_loss = local_loss / max(1, local_valid)

                self.logger.info(
                    'Epoch: {} || (Training upd {}/{}) Avg loss: {:.4f}, lr: {:.2e}, '
                    'exectime: {:.3f}s || [{}/{}]'.format(
                        self.epoch, update_step, num_updates_est, avg_loss,
                        self.optimizer.param_groups[0]['lr'], exec_time, taken, est))

                if self.tb:
                    log_dict = {"Learning Rate": self.optimizer.param_groups[0]['lr'],
                                "Grad Norm": grad_norm.item(),
                                "Training Loss(Avg.)": avg_loss}
                    for i, group in enumerate(self.optimizer.param_groups[1:], start=1):
                        log_dict[f"Learning Rate {i}"] = group['lr']
                    log_dict.update(loss_dict)
                    self.logging_dict("Training", log_dict, self.global_step)

                local_loss, local_valid = 0.0, 0
                local_start_time = time.time()

        if nan_count and self.glob_rank == 0:
            self.logger.warning(f'Epoch {self.epoch}: {nan_count} NaN/Inf events skipped')

        return total_loss / max(1, valid_batches)

    # ------------------------------------------------------- to be overridden

    def train(self):
        raise NotImplementedError("This method is not implemented.")

    def validate(self):
        raise NotImplementedError("This method is not implemented.")

    # ------------------------------------------------------------- reductions

    def calculate_ddp_loss(self, total_loss, total_sample):
        loss_sum = torch.tensor(total_loss, dtype=torch.float64, device=self.device)
        sample_sum = torch.tensor(total_sample, dtype=torch.float64, device=self.device)
        self.dist.barrier()
        self.dist.all_reduce(loss_sum)
        self.dist.all_reduce(sample_sum)
        return (loss_sum / sample_sum).item() if sample_sum.item() > 0 else float('inf')

    def reduce_metric(self, value_sum, count):
        val = torch.tensor(value_sum, dtype=torch.float64, device=self.device)
        cnt = torch.tensor(count, dtype=torch.float64, device=self.device)
        self.dist.all_reduce(val)
        self.dist.all_reduce(cnt)
        return (val / cnt).item() if cnt.item() > 0 else 0.0

    # ---------------------------------------------------------------- logging

    def logging(self, stage, tag, value, iteration, rank_zero_only=True):
        if self.glob_rank == 0 or not rank_zero_only:
            self.writer.add_scalar(f'{stage}/{tag}', value, iteration)
            self.writer.flush()

    def logging_dict(self, stage, log_dict, iteration, rank_zero_only=True):
        if self.glob_rank == 0 or not rank_zero_only:
            for tag, value in log_dict.items():
                self.writer.add_scalar(f'{stage}/{tag}', value, iteration)
            self.writer.flush()

    def logging_audio(self, stage, tag, audio, iteration, sample_rate=8000, rank_zero_only=True):
        """audio: (T) or (B,T) -- only the first item of a batch is written."""
        if not (self.glob_rank == 0 or not rank_zero_only):
            return
        if audio.dim() == 2:
            audio = audio[0]
        elif audio.dim() != 1:
            self.logger.warning('Audio logging skipped: tensor must be 1D or 2D.')
            return
        peak = torch.max(torch.abs(audio))
        audio = (audio / peak if peak > 0 else audio).float().cpu().numpy()
        self.writer.add_audio(f'{stage}/{tag}', audio, iteration, sample_rate=sample_rate)
        self.writer.flush()

    # ------------------------------------------------------------ checkpoints

    def save_checkpoint(self, path, rank_zero_only=True):
        if not (self.glob_rank == 0 or not rank_zero_only):
            return
        ckpt = dict(epoch=self.epoch, best_loss=self.best_loss, global_step=self.global_step)
        if hasattr(self, 'models'):
            ckpt['models'] = {k: m.module.state_dict() for k, m in self.models.items()}
            ckpt['optimizers'] = {k: o.state_dict() for k, o in self.optimizers.items()}
            ckpt['schedulers'] = {k: s.state_dict() for k, s in self.schedulers.items()}
        else:
            ckpt['model'] = self.model.module.state_dict()
            ckpt['optimizer'] = self.optimizer.state_dict()
            ckpt['scheduler'] = self.scheduler.state_dict()
        torch.save(ckpt, path)

    def load_checkpoint(self, path, only_model=False):
        checkpoint = torch_load(path, map_location='cpu')
        if hasattr(self, 'models'):
            for k, m in self.models.items():
                m.module.load_state_dict(checkpoint['models'][k])
            if not only_model:
                for k, o in self.optimizers.items():
                    o.load_state_dict(checkpoint['optimizers'][k])
                for k, s in self.schedulers.items():
                    s.load_state_dict(checkpoint['schedulers'][k])
        else:
            self.model.module.load_state_dict(checkpoint['model'])
            if not only_model:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])

        if not only_model:
            # Resume at the epoch AFTER the one the checkpoint completed.
            self.last_epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint['best_loss']
            self.global_step = checkpoint['global_step']
            self.logger.info(f'Resume training from epoch {self.last_epoch}. '
                             f'Best loss: {self.best_loss}')
        del checkpoint

    def prune_checkpoints(self):
        """Keep only the newest `keep_last_ckpt` numbered checkpoints."""
        if self.glob_rank != 0 or not self.keep_last_ckpt:
            return
        ckpts = sorted(f for f in os.listdir(self.ckpt_dir)
                       if f.startswith('model') and f.endswith('.pt'))
        for f in ckpts[:-self.keep_last_ckpt]:
            try:
                os.remove(os.path.join(self.ckpt_dir, f))
            except OSError:
                pass

    def search_checkpoint(self, ckpt_dir, search_type="last"):
        if not os.path.isdir(ckpt_dir):
            self.logger.info(f'The checkpoint directory {ckpt_dir} does not exist.')
            return None
        if search_type == "last":
            ckpt_list = sorted(f for f in os.listdir(ckpt_dir)
                               if f.startswith('model') and f.endswith('.pt'))
            if not ckpt_list:
                return None
            self.logger.info(f'Found last checkpoint: {ckpt_list[-1]}')
            return os.path.join(ckpt_dir, ckpt_list[-1])
        if search_type == "best":
            best_path = os.path.join(ckpt_dir, 'best.pt')
            if os.path.isfile(best_path):
                self.logger.info('Found best checkpoint: best.pt')
                return best_path
        return None

    # --------------------------------------------------------------- schedule

    def step_epoch_schedulers(self, val_loss):
        if hasattr(self, 'schedulers'):
            for key, sched in self.schedulers.items():
                cfg = self.config['schedulers'][key]
                if cfg.get('call_step', 'epoch') == "epoch":
                    sched.step(val_loss) if cfg.get('loss_pass', False) else sched.step()
        else:
            cfg = self.config['scheduler']
            if cfg['call_step'] == "epoch":
                self.scheduler.step(val_loss) if cfg['loss_pass'] else self.scheduler.step()

    # ------------------------------------------------------------------- loop

    def run(self):
        loss_not_changed = 0
        early_stop_flag = torch.zeros(1, device=self.device)

        for epoch in range(self.last_epoch, self.max_epoch):
            self.epoch = epoch
            if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            # Datasets using `num_per_epoch` rotate their window here. Worker
            # processes are respawned per epoch, so they inherit the new value.
            for loader in (self.train_loader, self.val_loader):
                ds = getattr(loader, 'dataset', None)
                if hasattr(ds, 'set_epoch'):
                    ds.set_epoch(epoch)

            torch.cuda.empty_cache()
            self.logger.info(f'Precision Setting: {self.precision_mode}, '
                             f'AMP Inference: {self.precision_inference}')
            self.logger.info(f'Epoch: {self.epoch} || (Training) Training start')

            start_time = time.time()
            train_loss = self.train()
            exectime = time.time() - start_time
            self.logger.info(f'Epoch: {self.epoch} || (Training) Avg loss: {train_loss:.4f}, '
                             f'Exec time: {exectime / 60:.4f} min')

            torch.cuda.empty_cache()
            self.logger.info(f'Epoch: {self.epoch} || (Validation) Validation start')
            start_time = time.time()
            val_loss = self.validate()

            if self.tb:
                self.logging("Training", "Average Loss", train_loss, self.epoch)
                self.logging("Validation", "Average Loss", val_loss, self.epoch)

            # best_loss is decided identically on every rank (val_loss is
            # all-reduced), so the value stored in any checkpoint is consistent.
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                loss_not_changed = 0
            else:
                loss_not_changed += 1

            # Saved AFTER training so the numbered checkpoint contains this
            # epoch's work; the reload re-synchronises every rank from rank 0.
            self.save_checkpoint(os.path.join(self.ckpt_dir, 'model%05d.pt' % self.epoch))
            self.dist.barrier()
            self.load_checkpoint(os.path.join(self.ckpt_dir, 'model%05d.pt' % self.epoch),
                                 only_model=True)
            self.dist.barrier()

            if self.glob_rank == 0:
                if is_best:
                    self.save_checkpoint(os.path.join(self.ckpt_dir, 'best.pt'))
                    self.logger.info(f'Epoch: {self.epoch} || (Validation) '
                                     f'Best loss changed -> {val_loss:.4f}')
                else:
                    self.logger.info(f'Epoch: {self.epoch} || (Validation) '
                                     f'Best loss not changed: {self.best_loss:.4f}')
                self.prune_checkpoints()

                exectime = time.time() - start_time
                self.logger.info(f'Epoch: {self.epoch} || (Validation) Avg loss: {val_loss:.4f}, '
                                 f'Exec time: {exectime / 60:.4f} min')

                if loss_not_changed >= self.early_stop:
                    self.logger.info(f'Best loss is not changed for {loss_not_changed} epochs. '
                                     f'Best loss: {self.best_loss}')
                    early_stop_flag += 1

            self.dist.barrier()
            self.dist.all_reduce(early_stop_flag, op=self.dist.ReduceOp.SUM)
            if early_stop_flag.item() > 0:
                self.logger.info('Satisfying the early stop condition. Stop training')
                break

            self.step_epoch_schedulers(val_loss)

        torch.cuda.empty_cache()
        if self.tb and self.glob_rank == 0:
            self.writer.close()
