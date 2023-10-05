from typing import Any, Literal, Union
from pathlib import Path
import jsonlines
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


# average html annotation length: train: 181.327 149.753
# samples train: 500777, val: 9115
class PubTabNet(Dataset):
    def __init__(
        self,
        root_dir: Union[Path, str],
        json_html: Union[Path, str],
        label_type: Literal["structure", "structure+content+bbox"],  # will add more options
        split: Literal["train", "val"],
        transform: transforms = None,
    ) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.split = split
        self.label_type = label_type
        self.transform = transform

        self.img_html_pair = self.process_json_annotations(
            Path(root_dir) / json_html, split
        )

    def process_json_annotations(self, json_file_dir: Path, split: str):
        img_html_pair = list()
        with jsonlines.open(json_file_dir) as f:
            for obj in f:
                if obj["split"] == split:
                    img_html_pair.append((obj["filename"], obj["html"]))

        return img_html_pair

    def __len__(self):
        return len(self.img_html_pair)

    def __getitem__(self, index: int) -> Any:
        obj = self.img_html_pair[index]

        img = Image.open(Path(self.root_dir) / self.split / obj[0])
        if self.transform:
            img = self.transform(img)

        sample = dict(filename=obj[0], image=img)

        if self.label_type == "structure":
            sample["structure"] = obj[1]["structure"]["tokens"]
        else:
            raise NotImplementedError

        return sample


