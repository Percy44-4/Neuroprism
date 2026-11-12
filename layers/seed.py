from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class DecoderSeed:
    grid: Tensor
    frame_context: Tensor


class ReadoutCrossAttention(nn.Module):
    def __init__(self, query_dim: int, source_dim: int, heads: int = 8) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(query_dim)
        self.source_norm = nn.LayerNorm(source_dim)
        self.source_projection = nn.Linear(source_dim, query_dim)
        self.attention = nn.MultiheadAttention(query_dim, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(query_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(query_dim, 4 * query_dim),
            nn.GELU(),
            nn.Linear(4 * query_dim, query_dim),
        )

    def forward(self, queries: Tensor, source: Tensor) -> Tensor:
        memory = self.source_projection(self.source_norm(source))
        attended = self.attention(
            self.query_norm(queries),
            memory,
            memory,
            need_weights=False,
        )[0]
        queries = queries + attended
        return queries + self.feed_forward(self.output_norm(queries))


class QueryGridSeed(nn.Module):
    def __init__(
        self,
        frames: int = 6,
        resolution: int = 8,
        source_dim: int = 768,
        channels: int = 256,
    ) -> None:
        super().__init__()
        self.frames = frames
        self.resolution = resolution
        self.channels = channels
        self.frame_positions = nn.Parameter(torch.randn(frames, channels) * 0.02)
        self.row_positions = nn.Parameter(torch.randn(resolution, channels) * 0.02)
        self.column_positions = nn.Parameter(torch.randn(resolution, channels) * 0.02)
        self.query_offsets = nn.Parameter(torch.randn(frames, resolution, resolution, channels) * 0.02)
        self.grid_attention = ReadoutCrossAttention(channels, source_dim)
        self.grid_refinement = ReadoutCrossAttention(channels, source_dim)
        self.frame_queries = nn.Parameter(torch.randn(frames, channels) * 0.02)
        self.frame_attention = ReadoutCrossAttention(channels, source_dim)
        self.grid_norm = nn.LayerNorm(channels)
        self.frame_norm = nn.LayerNorm(channels)

    def position_queries(self, batch: int) -> Tensor:
        temporal = self.frame_positions[:, None, None, :]
        vertical = self.row_positions[None, :, None, :]
        horizontal = self.column_positions[None, None, :, :]
        positions = temporal + vertical + horizontal + self.query_offsets
        positions = positions.reshape(1, self.frames * self.resolution ** 2, self.channels)
        return positions.expand(batch, -1, -1)

    def forward(self, features: Tensor) -> DecoderSeed:
        if features.ndim != 3:
            raise ValueError("readout_features")
        batch = features.shape[0]
        queries = self.position_queries(batch)
        grid_tokens = self.grid_attention(queries, features)
        grid_tokens = self.grid_refinement(grid_tokens, features)
        grid_tokens = self.grid_norm(grid_tokens)
        grid = grid_tokens.reshape(
            batch,
            self.frames,
            self.resolution,
            self.resolution,
            self.channels,
        ).permute(0, 4, 1, 2, 3)
        frame_queries = self.frame_queries.unsqueeze(0).expand(batch, -1, -1)
        frame_context = self.frame_attention(frame_queries, features)
        return DecoderSeed(grid, self.frame_norm(frame_context))
