from typing import Any
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch
from torchtext.vocab import Vocab
from torch.utils.data import default_collate


class Collator:
    def __init__(self, vocab_structure: Vocab, max_seq_len: int) -> None:
        self.vocab_structure = vocab_structure
        self.max_seq_len = max_seq_len

    def __call__(self, batch) -> Any:
        return self._collate_batch(batch, self.vocab_structure, self.max_seq_len)

    def _collate_batch(self, batch, vocab_structure: Vocab, max_seq_len: int):
        sos_id, eos_id, pad_id = vocab_structure(["<sos>", "<eos>", "<pad>"])
        img_list, label_list = list(), list()
        for obj in batch:
            img_list.append(obj["image"])
            label = dict(filename=obj["filename"])

            # structure
            if "structure" in obj.keys():
                html_structure = torch.tensor(
                    [sos_id, *vocab_structure(obj["structure"]), eos_id],
                    dtype=torch.int32,
                )
                html_structure = F.pad(
                    html_structure,
                    pad=(0, max_seq_len - len(html_structure)),
                    mode="constant",
                    value=pad_id,
                )
                label["html_structure"] = html_structure

            # cell text
            if "text" in obj.keys():
                raise NotImplementedError

            if "bbox" in obj.keys():
                raise NotImplementedError

            label_list.append(label)

        img_list = default_collate(img_list)
        label_list = default_collate(label_list)

        return img_list, label_list


def get_dataloader(
    dataset: Dataset,
    vocab_structure: Vocab,
    max_seq_len: int,
    batch_size: int,
    sampler=None,
) -> DataLoader:
    collate_fn = Collator(vocab_structure, max_seq_len)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=sampler,
    )

    return dataloader


