from math import gcd, pi, sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FeatureTokenProjection(nn.Module):
    def __init__(self, input_dim: int, token_count: int, token_dim: int) -> None:
        super().__init__()
        self.token_count = token_count
        self.token_dim = token_dim
        self.input_normalization = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, token_count * token_dim)
        self.token_normalization = nn.LayerNorm(token_dim)
        self.position_embedding = nn.Parameter(torch.randn(token_count, token_dim) * 0.02)

    def forward(self, features: Tensor) -> Tensor:
        projected = self.projection(self.input_normalization(features))
        tokens = projected.reshape(features.shape[0], self.token_count, self.token_dim)
        return self.token_normalization(tokens + self.position_embedding)


class SceneAttention(nn.Module):
    def __init__(self, channels: int, heads: int, context_channels: int | None = None) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("invalid_scene_attention_heads")
        context_channels = channels if context_channels is None else context_channels
        self.heads = heads
        self.head_dim = channels // heads
        self.query_projection = nn.Linear(channels, channels, bias=False)
        self.key_projection = nn.Linear(context_channels, channels, bias=False)
        self.value_projection = nn.Linear(context_channels, channels, bias=False)
        self.output_projection = nn.Linear(channels, channels)

    def split_heads(self, tokens: Tensor) -> Tensor:
        batch, length, _ = tokens.shape
        return tokens.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, queries: Tensor, context: Tensor | None = None) -> Tensor:
        context = queries if context is None else context
        query = self.split_heads(self.query_projection(queries))
        key = self.split_heads(self.key_projection(context))
        value = self.split_heads(self.value_projection(context))
        scores = torch.matmul(query, key.transpose(-1, -2)) / sqrt(self.head_dim)
        probabilities = scores.softmax(dim=-1)
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).reshape(queries.shape[0], queries.shape[1], -1)
        return self.output_projection(attended)


class SceneFeedForward(nn.Module):
    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)
        self.input_projection = nn.Linear(channels, expansion * channels)
        self.output_projection = nn.Linear(expansion * channels, channels)

    def forward(self, tokens: Tensor) -> Tensor:
        hidden = self.input_projection(self.normalization(tokens))
        return tokens + self.output_projection(F.gelu(hidden))


class SceneTransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int, expansion: int = 4) -> None:
        super().__init__()
        self.attention_normalization = nn.LayerNorm(channels)
        self.attention = SceneAttention(channels, heads)
        self.feed_forward = SceneFeedForward(channels, expansion)

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.attention(self.attention_normalization(tokens))
        return self.feed_forward(tokens)


class FourierCoordinateEmbedding(nn.Module):
    def __init__(self, channels: int, frequencies: int = 4) -> None:
        super().__init__()
        self.frequencies = frequencies
        self.projection = nn.Linear(4 + 4 * frequencies, channels)

    def forward(self, height: int, width: int, reference: Tensor) -> Tensor:
        rows = torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype)
        columns = torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype)
        row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
        coordinates = torch.stack((row_grid, column_grid, 1 - row_grid, 1 - column_grid), dim=-1)
        periodic = []
        for exponent in range(self.frequencies):
            frequency = (2 ** exponent) * pi
            periodic.extend((
                (frequency * row_grid).sin(),
                (frequency * row_grid).cos(),
                (frequency * column_grid).sin(),
                (frequency * column_grid).cos(),
            ))
        encoded = torch.cat((coordinates, torch.stack(periodic, dim=-1)), dim=-1)
        return self.projection(encoded)


class ConditionedSpatialResidual(nn.Module):
    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        self.input_normalization = nn.GroupNorm(gcd(16, channels), channels, eps=1e-6)
        self.output_normalization = nn.GroupNorm(gcd(16, channels), channels, eps=1e-6)
        self.condition_projection = nn.Linear(condition_dim, 2 * channels)
        self.input_convolution = nn.Conv2d(channels, channels, 3, padding=1)
        self.output_convolution = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, images: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.condition_projection(condition).chunk(2, dim=-1)
        hidden = self.input_normalization(images)
        hidden = hidden * (1 + scale[..., None, None]) + shift[..., None, None]
        hidden = self.input_convolution(F.silu(hidden))
        hidden = self.output_convolution(F.silu(self.output_normalization(hidden)))
        return images + hidden


class TextureTemporalMixer(nn.Module):
    def __init__(self, channels: int, kernel: int = 3) -> None:
        super().__init__()
        self.radius = kernel // 2
        self.normalization = nn.LayerNorm(channels)
        self.depthwise = nn.Conv3d(channels, channels, (kernel, 1, 1), groups=channels)
        self.channel_projection = nn.Conv3d(channels, channels, 1)
        self.mix = nn.Parameter(torch.full((channels,), 0.1))

    def forward(self, video: Tensor) -> Tensor:
        hidden = self.normalization(video).permute(0, 4, 1, 2, 3)
        hidden = F.pad(hidden, (0, 0, 0, 0, self.radius, self.radius), mode="replicate")
        hidden = self.channel_projection(F.silu(self.depthwise(hidden)))
        hidden = hidden.permute(0, 2, 3, 4, 1)
        return video + self.mix * hidden
