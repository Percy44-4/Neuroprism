import torch
from torch import Tensor, nn


class LearnedPositionEmbedding(nn.Module):
    def __init__(self, length: int, width: int, scale: float = 0.02) -> None:
        super().__init__()
        self.length = length
        self.weight = nn.Parameter(torch.empty(length, width))
        nn.init.normal_(self.weight, std=scale)

    def forward(self, hidden: Tensor) -> Tensor:
        length = hidden.shape[1]
        if length > self.length:
            raise ValueError("position_capacity_exceeded")
        return hidden + self.weight[:length].to(hidden.dtype)
