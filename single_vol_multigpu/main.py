import os
import time
from itertools import chain

import torch
import torch.multiprocessing as mp
from config.config_utils import handle_reproducibility, load_config, parse_args
from datasets import KCoordDataset, seed_worker
from model import *
from torch.distributed import destroy_process_group, init_process_group
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from train_utils_tboard import *


def ddp_setup(rank, world_size):
    """
    Initialize the distributed process group (i.e., all the processes that are running on the GPUs).
    Typically, each GPU runs one process. Setting up a group is necessary so that the processes
    can communicate with one another.
    Args:
        rank: Unique identifier of each process (ranges from 0 to world_size-1).
        world_size: Total number of processes (in a group).
    """
    torch.cuda.set_device(rank)
    init_process_group("nccl", rank=rank, world_size=world_size)


MODEL_CLASSES = {
    "Siren2.0": Siren_skip,
    "Siren3.0": Siren_skip_emb,
    "Siren4.0": Siren_skip_hash
}

LOSS_CLASSES = {
    "MAE": MAELoss,
    "DMAE": DMAELoss,
    "MSE": MSELoss,
    "MSEDist": MSEDistLoss,
    "HDR": HDRLoss,
    "LogL2": LogL2Loss,
}

OPTIMIZER_CLASSES = {"Adam": Adam, "AdamW": AdamW, "SGD": SGD}

SCHEDULER_CLASSES = {"StepLR": StepLR}


def main(rank: int, world_size: int, config: dict):
    args = parse_args()
    config = load_config(args.config)
    config["device"] = args.device
    
    # Setup distributed data parallel environment
    ddp_setup(rank, world_size=world_size)

    # Seed reproducibility
    rs_numpy, rs_torch = handle_reproducibility(config["seed"])

    torch.set_default_dtype(torch.float32)
    
    ##################################################
    # Initialization
    ##################################################
    dataset_undersampled = KCoordDataset(**config["dataset"])
    dataset_fullysampled = KCoordDataset(**config["dataset_full"])
    loader_config = config["dataloader"]
    
    dataloader_undersampled = DataLoader(
        dataset_undersampled,
        batch_size=loader_config["batch_size"],
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        shuffle=False,
        sampler=DistributedSampler(dataset_undersampled),
        pin_memory=loader_config["pin_memory"],
    )
    
    dataloader_fullysampled = DataLoader(
        dataset_fullysampled,
        batch_size=loader_config["batch_size"],
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        shuffle=False,
        sampler=DistributedSampler(dataset_fullysampled),
        pin_memory=loader_config["pin_memory"],
    )

    model_params = config["model"]["params"]
    model = MODEL_CLASSES[config["model"]["id"]](**model_params)
    
    if "model_checkpoint" in config.keys():
        model_state_dict = torch.load(config["model_checkpoint"])["model_state_dict"]
        model.load_state_dict(model_state_dict)
        print(f"[GPU {rank}] Checkpoint loaded successfully.")

    loss_fn = LOSS_CLASSES[config["loss"]["id"]](**config["loss"]["params"])
    optimizer = OPTIMIZER_CLASSES[config["optimizer"]["id"]](
        model.parameters(), **config["optimizer"]["params"]
    )
    scheduler = SCHEDULER_CLASSES[config["scheduler"]["id"]](
        optimizer, **config["scheduler"]["params"]
    )
    
    if rank == 0:  # Print configuration on rank 0 for debugging
        print(f"model {model}")
        print(f"loss {loss_fn}")
        print(f"optimizer {optimizer}")
        print(f"scheduler {scheduler}")
        print(config)

    ##################################################
    # Training Process
    ##################################################
    print(f"[GPU {rank}] Starting training process...")
    t0 = time.time()

    trainer = Trainer(
        dataloader_consistency=dataloader_undersampled,
        dataloader_pisco=dataloader_fullysampled,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )
    trainer.train()

    save_config(config)

    t1 = time.time()
    print(f"[GPU {rank}] Time it took to train: {(t1-t0)/60} min")

    # Clean up the distributed process group
    destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    # Get world size (total number of GPUs)
    world_size = torch.cuda.device_count()
    assert world_size == int(os.environ["SLURM_GPUS_ON_NODE"]), (
        "Mismatch between available GPUs and SLURM allocation."
    )

    # Spawn the distributed processes
    print(f"Starting process on {world_size} GPUs...")
    mp.spawn(main, args=(world_size, config), nprocs=world_size)
