import jsonlines
from typing import Union, Optional, List
from pathlib import Path
import torch
from torchtext.vocab import build_vocab_from_iterator


def build_pubtabnet_vocab(
    json_file_dir: Union[Path, str],
    save_dir: Union[Path, str],
    special_tokens: Optional[List[str]],
):
    html_structure = list()
    with jsonlines.open(json_file_dir) as f:
        for obj in f:
            html_structure.append(obj["html"]["structure"]["tokens"])

    vocab_structure = build_vocab_from_iterator(
        html_structure, min_freq=1, specials=special_tokens
    )

    vocab_structure.set_default_index(vocab_structure["<unk>"])

    print(vocab_structure.get_itos())

    torch.save(vocab_structure, Path(save_dir) / "vocab_structure.pt")

