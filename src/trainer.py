from typing import Tuple, List, Union, Dict
import torch
import time
import wandb
import json
from torch import nn, Tensor, autograd
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from omegaconf import DictConfig
from hydra.utils import instantiate
import logging
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import default_collate

from src.utils.utils import (
    Batch,
    subsequent_mask,
    printer,
    compute_grad_norm,
    to_html_table,
)

SNAPSHOT_KEYS = set(["EPOCH", "OPTIMIZER", "LR_SCHEDULER", "MODEL", "LOSS"])


class Trainer:
    def __init__(
        self,
        device: int,
        vocab_structure: Vocab,
        model: nn.Module,
        log: logging.Logger,
        exp_dir: Path,
        snapshot: Path = None,
        model_weights: Path = None,  # only for testing
    ) -> None:
        # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
        torch.cuda.set_device(device)  # master gpu takes up extra memory
        torch.cuda.empty_cache()

        self.device = device
        self.log = log
        self.exp_dir = exp_dir
        self.vocab_structure = vocab_structure
        self.padding_idx = vocab_structure(["<pad>"])[0]
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.padding_idx)

        assert snapshot is None or model_weights is None

        self.model = model
        if snapshot is not None and snapshot.is_file():
            self.snapshot = self.load_snapshot(snapshot)
            self.model.load_state_dict(self.snapshot["MODEL"])
            self.start_epoch = self.snapshot["EPOCH"]
        elif model_weights is not None and model_weights.is_file():
            self.load_model(model_weights)
        else:
            self.snapshot = None
            self.start_epoch = 0

        self.model = DDP(self.model, device_ids=[device])

    def batch_inference(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        if batch.target == "structure":
            structure_pred = self.model(
                batch.img,
                batch.structure_src,
                batch.structure_casual_mask,
                batch.structure_padding_mask,
            )

            loss = self.criterion(structure_pred.permute(0, 2, 1), batch.structure_tgt)
        else:
            raise NotImplementedError

        return loss, structure_pred

    def train_epoch(self, epoch: int, grad_clip: float, target: str):
        start = time.time()
        total_token = 0
        total_loss = 0.0

        # load data from dataloader
        for i, obj in enumerate(self.train_dataloader):
            batch = Batch(
                device=self.device,
                padding_idx=self.padding_idx,
                name=obj[1]["filename"],
                target=target,
                img=obj[0],
                structure=obj[1]["html_structure"] if target == "structure" else None,
            )

            with autograd.detect_anomaly():
                loss, _ = self.batch_inference(batch)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                self.optimizer.step()

            total_token += batch.ntoken_structure
            loss = loss.detach().cpu().data
            total_loss += loss * batch.ntoken_structure

            if i % 10 == 0:
                grad_norm = compute_grad_norm(self.model)
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                self.log.info(
                    printer(
                        self.device,
                        f"Epoch {epoch} Step {i + 1}/{len(self.train_dataloader)} | Loss {loss:.4f} ({total_loss / total_token:.4f}) | Grad norm: {grad_norm:.3f} | {total_token / elapsed:6.1f} tokens/s | lr {lr:5.1e}",
                    )
                )
        self.lr_scheduler.step()

        return total_loss, total_token

    def train(
        self,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        train_cfg: DictConfig,
        valid_cfg: DictConfig,
    ):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = instantiate(
            train_cfg.optimizer, params=self.model.parameters()
        )

        self.lr_scheduler = instantiate(
            train_cfg.lr_scheduler, optimizer=self.optimizer
        )

        if self.snapshot is not None:
            self.optimizer.load_state_dict(self.snapshot["OPTIMIZER"])
            self.lr_scheduler.load_state_dict(self.snapshot["LR_SCHEDULER"])

        best_loss = float("inf")
        self.model.train()
        for epoch in range(self.start_epoch, train_cfg.epochs):
            train_dataloader.sampler.set_epoch(epoch)
            epoch_loss, epoch_token = self.train_epoch(
                epoch,
                grad_clip=train_cfg.grad_clip,
                target=train_cfg.target,
            )
            valid_loss, valid_token = self.valid(valid_cfg)

            # reduce loss to gpu 0
            training_info = torch.tensor(
                [epoch_loss, epoch_token, valid_loss, valid_token], device=self.device
            )

            dist.reduce(
                training_info,
                dst=0,
                op=dist.ReduceOp.SUM,
            )

            if self.device == 0:
                grad_norm = compute_grad_norm(self.model)
                epoch_loss, epoch_token, valid_loss, valid_token = training_info
                epoch_loss, valid_loss = (
                    float(epoch_loss) / epoch_token,
                    float(valid_loss) / valid_token,
                )

                wandb.log(
                    {
                        "train loss": epoch_loss,
                        "valid loss": valid_loss,
                        "train_tokens": epoch_token,
                        "valid_tokens": valid_token,
                        "grad_norm": grad_norm,
                    },
                    step=epoch,
                )

                if epoch % train_cfg.save_every == 0:
                    self.save_snapshot(epoch, best_loss)
                if valid_loss < best_loss:
                    self.save_model(epoch)
                    best_loss = valid_loss

    def valid(self, cfg: DictConfig):
        total_token = 0
        total_loss = 0.0

        self.model.eval()
        for i, obj in enumerate(self.valid_dataloader):
            batch = Batch(
                device=self.device,
                padding_idx=self.padding_idx,
                name=obj[1]["filename"],
                target=cfg.target,
                img=obj[0],
                structure=obj[1]["html_structure"],
            )

            loss, _ = self.batch_inference(batch)

            total_token += batch.ntoken_structure
            loss = loss.detach().cpu().data
            total_loss += loss * batch.ntoken_structure

            if i % 10 == 0:
                self.log.info(
                    printer(
                        self.device,
                        f"Valid: Step {i + 1}/{len(self.valid_dataloader)} | Loss {loss:.4f} ({total_loss / total_token:.4f})",
                    )
                )

        return total_loss, total_token

    def test(self, test_dataloader: DataLoader, cfg: DictConfig):
        for i, obj in enumerate(test_dataloader):
            batch = Batch(
                device=self.device,
                padding_idx=self.padding_idx,
                name=obj[1]["filename"],
                target=cfg.target,
                img=obj[0],
                structure=obj[1]["html_structure"],
            )

            pred = self.greedy_decode(batch, cfg.max_seq_len)

            if i == 0:
                total_pred = pred
            else:
                for k, v in pred.items():
                    if isinstance(v, list):
                        total_pred[k].extend(v)
                    elif torch.is_tensor(v):
                        total_pred[k] = torch.cat((total_pred[k], v), dim=0)
                    else:
                        raise ValueError(
                            f"Unrecognized prediction type {type(v)} of {k}."
                        )

            if i % 10 == 0:
                self.log.info(
                    printer(
                        self.device,
                        f"Test: Step {i + 1}/{len(test_dataloader)}",
                    )
                )

        total_pred_list = [None for _ in range(dist.get_world_size())]
        dist.gather_object(
            total_pred, total_pred_list if dist.get_rank() == 0 else None, dst=0
        )

        if self.device == 0:
            for i, pred in enumerate(total_pred_list):
                pred = to_html_table(self.vocab_structure, pred)
                if i == 0:
                    total_pred = pred
                else:
                    total_pred.update(pred)

            self.log.info(
                printer(
                    self.device,
                    f"Converting {len(total_pred)} samples to html tables ...",
                )
            )

            with open(self.exp_dir / cfg.save_to, "w", encoding="utf-8") as f:
                json.dump(total_pred, f, indent=4)

        return total_pred

    def greedy_decode(self, batch: Batch, max_seq_len: int) -> Dict[str, Tensor]:
        pred = dict(filename=batch.name, structure_gt=batch.structure_tgt)
        self.model.eval()
        with torch.no_grad():
            memory = self.model.module.encode(batch.img)

            N = memory.shape[0]
            structure_pred = (
                torch.tensor(self.vocab_structure(["<sos>"]), dtype=torch.int)
                .repeat(N, 1)
                .to(self.device)
            )

            for _ in range(max_seq_len - 1):
                causal_mask = subsequent_mask(structure_pred.shape[1]).to(self.device)
                out = self.model.module.decode(
                    memory, structure_pred, tgt_mask=causal_mask, tgt_padding_mask=None
                )
                prob = self.model.module.generator(out[:, -1])
                _, next_word = torch.max(prob, dim=-1)
                structure_pred = torch.cat(
                    [structure_pred, next_word.unsqueeze(1)], dim=1
                )

            pred["structure_pred"] = structure_pred

        return pred

    def save_model(self, epoch: int):
        filename = Path(self.exp_dir) / "model" / f"epoch{epoch}_model.pt"
        torch.save(self.model.module.state_dict(), filename)
        self.log.info(printer(self.device, f"Saving model to {filename}"))
        filename = Path(self.exp_dir) / "model" / f"best.pt"
        torch.save(self.model.module.state_dict(), filename)

    def load_model(self, path: Union[str, Path]):
        self.model.load_state_dict(torch.load(path))
        self.log.info(printer(self.device, f"Loading model from {path}"))

    def save_snapshot(self, epoch: int, best_loss: float):
        state_info = {
            "EPOCH": epoch + 1,
            "OPTIMIZER": self.optimizer.state_dict(),
            "LR_SCHEDULER": self.lr_scheduler.state_dict(),
            "MODEL": self.model.module.state_dict(),
            "LOSS": best_loss,
        }

        snapshot_path = Path(self.exp_dir) / "snapshot" / f"epoch{epoch}_snapshot.pt"
        torch.save(state_info, snapshot_path)

        self.log.info(printer(self.device, f"Saving snapshot to {snapshot_path}"))

    def load_snapshot(self, path: Path):
        self.log.info(printer(self.device, f"Loading snapshot from {path}"))
        snapshot = torch.load(path)
        assert SNAPSHOT_KEYS.issubset(snapshot.keys())
        return snapshot
