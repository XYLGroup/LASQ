import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.distributed as dist
import numpy as np
import torchvision
import models
import datasets
import utils
import time
from models import DenoisingDiffusion
from torch.nn.parallel import DistributedDataParallel as DDP



def parse_args_and_config():
    parser = argparse.ArgumentParser(description='LASQ')
    parser.add_argument("--config", default='unsupervised.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--mode', type=str, default='training', help='training or evaluation')
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--image_folder", default='results/', type=str,
                        help="Location to save restored validation image patches")
    parser.add_argument('--seed', default=42, type=int, metavar='N',
                        help='Seed for initializing training (default: 230)')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Number of processes for distributed training')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--lambda_adv', type=float, default=0.00002,
                        help='Adversarial Loss')
    parser.add_argument('--lambda_scc', type=float, default=0.25,
                        help='SCC Loss')
    parser.add_argument('--N', type=int, default=10,
                        help='Number of the forward sampling')
    parser.add_argument('--eval_path', type=str, default='',
                        help='Path of the GT for evaluation')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        print(f"RANK {args.rank} LOCAL_RANK {args.local_rank} WORLD_SIZE {args.world_size}")

        dist.init_process_group(
            backend='nccl',
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        config.device = device
    else:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        config.device = device
        args.world_size = 1
        args.local_rank = 0

    print(f"Using device: {device} on rank {args.local_rank}")

    seed = args.seed + args.local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    print("=> using dataset '{}'".format(config.data.train_dataset))
    DATASET = datasets.__dict__[config.data.type](config)

    print("=> creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(args, config)

    diffusion.train(DATASET, args.local_rank, args.world_size)

    if args.world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()