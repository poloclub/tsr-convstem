from typing import Optional, Tuple
import math
import torch
from torch import nn, Tensor
from torchvision.ops.misc import Conv2dNormActivation


class ImgCnnBackbone(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        output_channels: int,
        d_model: int,
        drop_layer: Tuple = None,
    ) -> None:
        super().__init__()

        # drop layers for classification & maxpooling for higher feature resolution
        layers = list(backbone.children())
        nlayer = len(layers)
        keep_layer = set([i for i in range(nlayer)]) - set(drop_layer)
        backbone = [layers[i] for i in keep_layer]
        self.backbone = nn.Sequential(*backbone)
        self.proj = nn.Linear(output_channels, d_model)
        self.channels = output_channels

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = x.flatten(start_dim=-2).transpose(1, 2)
        assert x.shape[-1] == self.channels, "Image channels size mismatch."
        x = self.proj(x)
        return x


class ImgLinearBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        patch_size: int,
    ) -> None:
        super().__init__()

        self.conv_proj = nn.Conv2d(
            3, out_channels=d_model, kernel_size=patch_size, stride=patch_size
        )
        self.d_model = d_model

        self.init_weight()

    def init_weight(self):
        fan_in = (
            self.conv_proj.in_channels
            * self.conv_proj.kernel_size[0]
            * self.conv_proj.kernel_size[1]
        )
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_proj(x)
        x = x.flatten(start_dim=-2).transpose(1, 2)
        return x


class ImgConvStemBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        downsample_factor: int,
        output_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()

        assert downsample_factor % 2 == 0
        assert output_channels % (downsample_factor // 2) == 0
        input_channels = output_channels // (downsample_factor // 2)

        layers = [
            Conv2dNormActivation(
                3, input_channels, kernel_size=kernel_size, stride=2, padding=1
            )
        ]

        while input_channels != output_channels:
            layers.append(
                Conv2dNormActivation(
                    input_channels,
                    input_channels * 2,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                )
            )
            input_channels = input_channels * 2

        layers.append(nn.Conv2d(output_channels, d_model, kernel_size=1))

        self.conv_stem = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_stem(x)
        x = x.flatten(start_dim=-2).transpose(1, 2)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        nlayer: int,
        ff_ratio: int = 4,
    ) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead=nhead,
            dim_feedforward=ff_ratio * d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        nlayer: int,
        ff_ratio: int = 4,
    ) -> None:
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward=ff_ratio * d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, nlayer)

    def forward(
        self, x: Tensor, memory: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor
    ) -> Tensor:
        x = self.decoder(
            x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask
        )
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # assume x is batch first
        out = self.embedding(torch.arange(x.shape[1], device=x.device))
        return self.dropout(out + x)


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int,
    ) -> None:
        super().__init__()
        assert vocab_size > 0
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)
