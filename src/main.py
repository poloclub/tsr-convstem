from typing import Any
import hydra
import logging
import os
import wandb
import torch
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, instantiate
from pathlib import Path
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from src.dataloader.pubtabnet import PubTabNet
from src.dataloader.dataloader import get_dataloader
from src.utils.utils import get_transform, printer, count_total_parameters
from src.trainer import Trainer

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    ddp_setup()
    device = int(os.environ["LOCAL_RANK"])
    cwd = Path(get_original_cwd())
    exp_dir = Path(os.getcwd())  # experiment directory
    (exp_dir / "snapshot").mkdir(parents=True, exist_ok=True)
    (exp_dir / "model").mkdir(parents=True, exist_ok=True)
    if device == 0 and cfg.trainer.mode == "train":
        wandb.init(project=cfg.wandb.project, name=cfg.name, resume=True)

    # vocab
    vocab_structure = torch.load(cwd / cfg.vocab.structure.dir)
    padding_idx = vocab_structure(["<pad>"])[0]

    # dataset
    train_transform, valid_transform = get_transform(
        img_size=cfg.trainer.img_size, mean=cfg.dataset.mean, std=cfg.dataset.std
    )

    if cfg.trainer.mode == "train":
        log.info(printer(device, "Loading training dataset"))
        train_dataset = PubTabNet(
            root_dir=cfg.dataset.root_dir,
            json_html=cfg.dataset.json_html,
            label_type=cfg.dataset.label_type,
            split="train",
            transform=train_transform,
        )

        train_dataloader = get_dataloader(
            train_dataset,
            vocab_structure=vocab_structure,
            max_seq_len=cfg.trainer.max_seq_len,
            batch_size=cfg.trainer.train.batch_size,
            sampler=DistributedSampler(train_dataset),
        )

    log.info(printer(device, "Loading validation dataset"))

    valid_dataset = PubTabNet(
        root_dir=cfg.dataset.root_dir,
        json_html=cfg.dataset.json_html,
        label_type=cfg.dataset.label_type,
        split="val",
        transform=valid_transform,
    )

    valid_dataloader = get_dataloader(
        valid_dataset,
        vocab_structure=vocab_structure,
        max_seq_len=cfg.trainer.max_seq_len,
        batch_size=cfg.trainer.valid.batch_size,
        sampler=DistributedSampler(valid_dataset),
    )

    # model
    log.info(printer(device, "Loading model ..."))
    max_seq_len = max(
        (cfg.trainer.img_size // cfg.model.backbone_downsampling_factor) ** 2,
        cfg.trainer.max_seq_len,
    )  # for positional embedding
    model = instantiate(
        cfg.model.model,
        max_seq_len=max_seq_len,
        vocab_size=len(vocab_structure),
        padding_idx=padding_idx,
    ).to(device)
    log.info(
        printer(device, f"Total parameters: {count_total_parameters(model) / 1e6:.2f}M")
    )

    # trainer
    trainer = Trainer(
        device=device,
        vocab_structure=vocab_structure,
        model=model,
        log=log,
        exp_dir=exp_dir,
        snapshot=exp_dir / "snapshot" / cfg.trainer.snapshot
        if cfg.trainer.snapshot
        else None,
        model_weights=Path(cfg.trainer.test.model) if cfg.trainer.test.model else None,
    )

    if cfg.trainer.mode == "train":
        log.info(printer(device, "Training starts ..."))
        trainer.train(
            train_dataloader, valid_dataloader, cfg.trainer.train, cfg.trainer.valid
        )

    if cfg.trainer.mode == "test":
        log.info(printer(device, "Evaluation starts ..."))
        trainer.test(valid_dataloader, cfg.trainer.test)

    destroy_process_group()


def ddp_setup():
    init_process_group(backend="nccl")


if __name__ == "__main__":
    main()
