from torch import Tensor, nn
from torch.nn import functional as F

from .attention import MultiHeadSelfAttention


class PrecisionLayerNorm(nn.LayerNorm):
    def forward(self, value: Tensor) -> Tensor:
        weight = None if self.weight is None else self.weight.float()
        bias = None if self.bias is None else self.bias.float()
        return F.layer_norm(value.float(), self.normalized_shape, weight, bias, self.eps).to(value.dtype)


class FeedForward(nn.Module):
    def __init__(self, width: int, ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.expand = nn.Linear(width, width * ratio)
        self.activation = nn.GELU(approximate="tanh")
        self.contract = nn.Linear(width * ratio, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = self.activation(self.expand(hidden))
        return self.dropout(self.contract(hidden))


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.attention_norm = PrecisionLayerNorm(width, eps=layer_norm_epsilon)
        self.attention = MultiHeadSelfAttention(width, heads, dropout)
        self.feed_forward_norm = PrecisionLayerNorm(width, eps=layer_norm_epsilon)
        self.feed_forward = FeedForward(width, mlp_ratio, dropout)

    def forward(self, hidden: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden), attention_mask)
        return hidden + self.feed_forward(self.feed_forward_norm(hidden))


class TransformerStack(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ResidualAttentionBlock(width, heads, mlp_ratio, dropout, layer_norm_epsilon)
            for _ in range(layers)
        )

    def forward(self, hidden: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        return hidden
