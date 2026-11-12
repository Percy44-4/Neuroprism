import torch
from torch import Tensor, nn
from torch.nn import functional as F


def combine_attention_mask(
    length: int,
    device: torch.device,
    padding_mask: Tensor | None = None,
) -> Tensor:
    positions = torch.arange(length, device=device)
    allowed = positions[None, :] <= positions[:, None]
    allowed = allowed[None, None]
    if padding_mask is not None:
        allowed = allowed & padding_mask[:, None, None, :].bool()
    return allowed


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("attention_width_not_divisible")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.dropout = dropout
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.output_dropout = nn.Dropout(dropout)

    def split_heads(self, value: Tensor) -> Tensor:
        batch, length, _ = value.shape
        return value.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        mask = combine_attention_mask(hidden.shape[1], hidden.device, attention_mask)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], self.width)
        return self.output_dropout(self.output(attended))
