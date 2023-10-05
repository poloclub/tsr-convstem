from typing import Dict, Tuple, List, Union
import torch
import math
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler, StepLR, LambdaLR
from torch.optim.optimizer import Optimizer
from torchvision import transforms
from torchtext.vocab import Vocab

printer = lambda device, output: f"[GPU {device}] " + output

html_table = (
    lambda table: f"""<html>
            <head>
            <meta charset="UTF-8">
            <style>
            table, th, td {{
                border: 1px solid black;
                font-size: 10px;
            }}
            </style>
            </head>
            <body>
            <table frame="hsides" rules="groups" width="100%%">
                {table}
            </table>
            </body>
            </html>"""
)


def combine_html_table_structure_text(
    structure: Tuple[str], content: Tuple[str] = None
) -> str:
    """Prepare html code for table visualization"""
    code = structure.copy()
    to_insert = [i for i, tag in enumerate(structure) if tag in ("<td>", ">")]
    if content is None:
        for i in to_insert[::-1]:
            code.insert(i + 1, "placeholder")
    else:
        raise NotImplementedError

    return html_table("".join(code))


def to_html_table(
    vocab_structure: Vocab, pred: Dict[str, Union[Tensor, List]]
) -> Dict[str, Dict]:
    """Prepare table's html code + bbox"""
    html_table = dict()

    structure_pred = to_tokens(vocab_structure, pred["structure_pred"])
    structure_gt = to_tokens(vocab_structure, pred["structure_gt"])

    for name, i, j in zip(pred["filename"], structure_pred, structure_gt):
        table = dict(
            structure_pred=combine_html_table_structure_text(i),
            structure_gt=combine_html_table_structure_text(j),
        )
        html_table[name] = table

    return html_table


def to_tokens(vocab: Vocab, batch_int: Tensor) -> List[List[str]]:
    batch_token = list()
    if batch_int.device != "cpu":
        batch_int = batch_int.cpu()
    for obj in batch_int.numpy():
        obj = vocab.lookup_tokens(obj)

        if obj[0] == "<sos>":
            obj = obj[1:]

        try:
            i = obj.index("<eos>")
        except ValueError:
            i = len(obj)

        batch_token.append(obj[: i + 1])

    return batch_token


def count_total_parameters(model: nn.Module) -> int:
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_parameters


def compute_grad_norm(model: nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def subsequent_mask(size: int) -> Tensor:
    attn_shape = (size, size)
    output = torch.triu(torch.ones(attn_shape), diagonal=1).to(torch.bool)
    return output


# adpated from https://github.com/huggingface/transformers/blob/v4.33.0/src/transformers/optimization.py
def cosine_schedule_with_warmup(
    step: int,
    *,
    warmup: int,
    total_step: int,
    cycle: float = 0.5,
):
    if step < warmup:
        if step == 0:
            step = 1
        return float(step) / float(max(1, warmup))
    progress = float(step - warmup) / float(max(1, total_step - warmup))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(cycle) * 2.0 * progress)))


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(
        self,
        device: torch.device,
        padding_idx: int,
        name: List[str],
        target: str,
        img: Tensor,
        structure: Tensor = None,
        cell_txt: Tensor = None,
        bbox: Tensor = None,
    ) -> None:
        self.name = name
        self.target = target
        self.img = img.to(device)

        if structure is not None:
            self.structure_src = structure[:, :-1].to(device)
            self.structure_tgt = structure[:, 1:].type(torch.LongTensor).to(device)
            self.structure_casual_mask = subsequent_mask(
                self.structure_src.shape[-1]
            ).to(device)
            self.structure_padding_mask = (
                (self.structure_src == padding_idx).to(torch.bool).to(device)
            )

            self.ntoken_structure = (self.structure_tgt != padding_idx).data.sum()

        if cell_txt is not None:
            raise NotImplementedError

        if bbox is not None:
            raise NotImplementedError


def get_transform(img_size: int, mean: Tuple[float], std: Tuple[float]):
    if max(mean) > 1:
        assert max(std) > 1
        mean = [i / 255.0 for i in mean]
        std = [i / 255.0 for i in std]
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return train_transform, valid_transform
