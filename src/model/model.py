import torch
from torch import Tensor, nn

from src.model.components import (
    ImgCnnBackbone,
    ImgLinearBackbone,
    ImgConvStemBackbone,
    Encoder,
    Decoder,
    PositionEmbedding,
    TokenEmbedding,
)

"""
image encoder + transformer encoder & decoder
image transformer encoder + text transformer decoder
"""


class EncoderDecoder(nn.Module):
    """Image encoder + Table structure decoder"""

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        vocab_size: int,
        d_model: int,
        padding_idx: int,
        max_seq_len: int,
        dropout: float,
    ):
        super().__init__()

        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.token_embed = TokenEmbedding(
            vocab_size=vocab_size, d_model=d_model, padding_idx=padding_idx
        )
        self.pos_embed = PositionEmbedding(
            max_seq_len=max_seq_len, d_model=d_model, dropout=dropout
        )
        self.generator = nn.Linear(d_model, vocab_size)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def encode(self, src: Tensor) -> Tensor:
        src_feature = self.backbone(src)
        src_feature = self.pos_embed(src_feature)
        memory = self.encoder(src_feature)
        return memory

    def decode(
        self, memory: Tensor, tgt: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor
    ) -> Tensor:
        tgt_feature = self.pos_embed(self.token_embed(tgt))
        tgt = self.decoder(tgt_feature, memory, tgt_mask, tgt_padding_mask)

        return tgt

    def forward(
        self, src: Tensor, tgt: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor
    ) -> Tensor:
        memory = self.encode(src)
        tgt = self.decode(memory, tgt, tgt_mask, tgt_padding_mask)
        tgt = self.generator(tgt)

        return tgt
