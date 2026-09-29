"""Distributed training entry point.

The launcher's only jobs are to resolve DDP topology, build the data loaders and
hand everything to a trainer. Model / optimizer / scheduler construction lives
inside the trainer (see src/trainer/ddp_trainer_skeleton.py), which is what lets
a trainer own several models without changing this file.
"""

import os
import json
import random
import logging
import warnings
import importlib
import socket
from datetime import datetime
from argparse import ArgumentParser

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from ruamel.yaml import YAML

LOG_LEVEL = logging.INFO


def setup_logger(node_id, logger_name, log_path, screen=True, file_save=True,
                 file_mode='w', level=logging.INFO):
    lg = logging.getLogger(logger_name)
    # Without this, a second setup_logger() call on the same name stacks another
    # pair of handlers and every line is emitted twice.
    lg.handlers.clear()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)

    if node_id == 0 and file_save:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, mode=file_mode)
        fh.setFormatter(formatter)
        lg.addHandler(fh)

    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

    lg.propagate = False


def parse_ddp_info(node_id, DDP_config):
    curr_node = list(DDP_config['devices'].values())[node_id]
    ddp_info = {}
    ddp_info['master_ip'] = list(DDP_config['devices'].values())[0]['ip']
    ddp_info['port'] = DDP_config['port']
    ddp_info['backend'] = DDP_config['backend']
    ddp_info['node_id'] = node_id
    ddp_info['gpus'] = curr_node['gpus']
    ddp_info['ip'] = curr_node['ip']
    ddp_info['world_size'] = sum(len(n['gpus']) for n in DDP_config['devices'].values())
    ddp_info['batch_size'] = curr_node['batch_size']
    ddp_info['rank_start'] = sum(len(n['gpus'])
                                 for n in list(DDP_config['devices'].values())[:node_id])
    return ddp_info


def find_free_port(start_port, addr="127.0.0.1", max_tries=1000):
    port = start_port
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((addr, port))
                return port
            except OSError:
                port += 1
    raise RuntimeError(f"No free port found from {start_port} within {max_tries} tries")


def set_seed(seed, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main_worker(local_rank, config):
    if config.get('trainer', {}).get('suppress_warnings', False):
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    proc = config.pop('ddp_info')
    proc['curr_gpu'] = local_rank
    proc['glob_rank'] = proc['rank_start'] + local_rank

    if proc['glob_rank'] == 0:
        setup_logger(proc['node_id'], config['logging']['name'], proc['logging_path'],
                     file_mode='a', level=LOG_LEVEL)
        logger = logging.getLogger(config['logging']['name'])
    else:
        setup_logger(proc['node_id'], 'tmp_log', None, False, False, level=LOG_LEVEL)
        logger = logging.getLogger('tmp_log')

    dist.init_process_group(backend=proc['backend'], init_method='env://',
                            world_size=proc['world_size'], rank=proc['glob_rank'])
    proc['dist'] = dist
    logger.info('rank: {} initialized'.format(proc['glob_rank']))

    torch.cuda.set_device(proc['curr_gpu'])

    seed = config.get('trainer', {}).get('seed', None)
    if seed is not None:
        # Offset by rank so the ranks do not draw identical augmentation noise
        # while the run as a whole stays reproducible.
        set_seed(seed + proc['glob_rank'])
    logger.info('rank: {} device set {}'.format(proc['glob_rank'], proc['curr_gpu']))

    proc['base_path'] = os.path.join(config['output_path'], config['name'])

    ### data loaders
    data_config = config['datasets']
    if seed is not None and 'train' in data_config:
        data_config['train']['dataset']['config'].setdefault('seed', seed)

    _trainset = importlib.import_module(
        "src.datasets.dataset." + data_config['train']['dataset']['name']).__getattribute__("Datasets")
    trainset = _trainset(**data_config['train']['dataset']['config'])

    _valset = importlib.import_module(
        "src.datasets.dataset." + data_config['validation']['dataset']['name']).__getattribute__("Datasets")
    valset = _valset(**data_config['validation']['dataset']['config'])

    debug_cfg = config.get('debug')
    if debug_cfg and debug_cfg.get('enable', False):
        rng = np.random.RandomState(debug_cfg.get('seed', 42))
        n_train = max(1, int(len(trainset) * debug_cfg.get('train_ratio', 0.05)))
        n_val = max(1, int(len(valset) * debug_cfg.get('val_ratio', 0.1)))
        trainset = Subset(trainset, rng.choice(len(trainset), n_train, replace=False).tolist())
        valset = Subset(valset, rng.choice(len(valset), n_val, replace=False).tolist())
        logger.info(f'[Debug] Subset: train {n_train}, val {n_val}')

    _train_sampler = importlib.import_module(
        "src.datasets.sampler." + data_config['train']['sampler']['name']).__getattribute__("Sampler")
    train_sampler = _train_sampler(trainset, **data_config['train']['sampler']['config'], **proc)

    train_loader = DataLoader(trainset, batch_size=proc['batch_size'],
                              sampler=train_sampler, **data_config['train']['loader'])

    _val_sampler = importlib.import_module(
        "src.datasets.sampler." + data_config['validation']['sampler']['name']).__getattribute__("Sampler")
    val_sampler = _val_sampler(valset, **data_config['validation']['sampler']['config'], **proc)

    val_loader = DataLoader(valset, batch_size=1, sampler=val_sampler,
                            **data_config['validation']['loader'])

    if proc['glob_rank'] != 0:
        logger.setLevel(logging.FATAL)

    logger.info(json.dumps(config.get('model', config.get('models', {})), indent='\t'))
    logger.info(f'Train Data: {len(trainset)}, Val Data: {len(valset)}')
    logger.info('Executing Trainer')

    _trainer = importlib.import_module(
        "src.trainer." + config['trainer']['name']).__getattribute__("Trainer")
    trainer = _trainer(config, proc, train_loader, val_loader)
    trainer.run()

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--node_id', '-n', default=0, type=int,
                        help="Node's order number in [0, num_of_nodes-1]")
    parser.add_argument('--config', '-c', required=True, type=str,
                        help="Path of config file")
    parser.add_argument('--tag', '-t', default=None, type=str,
                        help="Name of experiment")
    parser.add_argument('--output_path', default='./exps', type=str,
                        help="Directory experiments are written to")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    yaml = YAML(typ='safe', pure=True)
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f)

    if args.tag is not None:
        config['name'] = args.tag
    config['output_path'] = args.output_path

    node_nm = list(config['DDP']['devices'].keys())[args.node_id]
    start_time = datetime.now().strftime('%Y_%m_%d_%H%M')
    log_path = os.path.join(args.output_path, config['name'], start_time, f'{start_time}.log')
    setup_logger(args.node_id, config['logging']['name'], log_path, level=LOG_LEVEL)
    logger = logging.getLogger(config['logging']['name'])

    logger.info(f'Node {args.node_id} : {node_nm} executed')

    if len(config['DDP']['devices'][node_nm]['gpus']) == 0:
        logger.info('No specified GPUs. Detect available devices')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if visible is not None:
            # The process already sees only these devices, so index them locally.
            available_gpus = list(range(len(visible.split(','))))
        else:
            available_gpus = list(range(torch.cuda.device_count()))

        if len(available_gpus) == 0:
            logger.info('No available device detected. Kill the process.')
            raise SystemExit(1)
        logger.info(f'GPU {available_gpus} detected.')
        config['DDP']['devices'][node_nm]['gpus'] = available_gpus

    config['ddp_info'] = parse_ddp_info(args.node_id, config['DDP'])
    config['ddp_info']['start_time'] = start_time
    config['ddp_info']['logging_path'] = log_path
    logger.info(config['ddp_info'])

    os.environ['MASTER_ADDR'] = str(config['ddp_info']['master_ip'])
    if len(config['DDP']['devices']) == 1:
        # Single-node: a busy port is a local collision with another run of ours,
        # so stepping to the next free one is safe. Multi-node must not do this
        # -- the other nodes are told the configured port and cannot follow.
        port = find_free_port(config['ddp_info']['port'], '127.0.0.1')
        if port != config['ddp_info']['port']:
            logger.info(f"Port {config['ddp_info']['port']} is not available. Use port {port} instead.")
        os.environ['MASTER_PORT'] = str(port)
    else:
        os.environ['MASTER_PORT'] = str(config['ddp_info']['port'])

    os.environ['WORLD_SIZE'] = str(config['ddp_info']['world_size'])
    # NOTE: CUDA_VISIBLE_DEVICES is deliberately NOT overwritten here. The caller
    # owns device selection (`CUDA_VISIBLE_DEVICES=2,3 python train.py ...`) and
    # rewriting it from config silently moved runs onto the wrong GPUs.

    seed = config.get('trainer', {}).get('seed', None)
    if seed is not None:
        set_seed(seed)
        torch.backends.cudnn.benchmark = True
        logger.info(f'Random seed initialized: {seed}')

    try:
        mp.spawn(main_worker, args=(config,),
                 nprocs=len(config['ddp_info']['gpus']), join=True)
    except KeyboardInterrupt:
        logger.info('Interrupted. Tearing down the process group.')
        if dist.is_initialized():
            dist.destroy_process_group()
