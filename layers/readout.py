from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import HierarchyConfig, ReadoutConfig
from ..structures import SceneState


EvidenceChannel = Literal["spatial", "semantic", "language", "dynamic"]


class CrossAttentionBlock(nn.Module):
    def __init__(self, config: ReadoutConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.hidden_dim)
        self.memory_norm = nn.LayerNorm(config.hidden_dim)
        self.attention = nn.MultiheadAttention(
            config.hidden_dim, config.attention_heads, dropout=config.dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
        )

    def forward(self, queries: Tensor, memory: Tensor, memory_bias: Tensor) -> Tensor:
        normalized_memory = self.memory_norm(memory)
        attended = self.attention(
            self.query_norm(queries), normalized_memory, normalized_memory,
            attn_mask=memory_bias, need_weights=False,
        )[0]
        queries = queries + attended
        return queries + self.feed_forward(queries)


class SharedCrossAttentionReader(nn.Module):
    def __init__(
        self,
        config: ReadoutConfig = ReadoutConfig(),
        hierarchy: HierarchyConfig = HierarchyConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.texture_projection = nn.Linear(hierarchy.texture_shape[-1], config.hidden_dim)
        self.object_projection = nn.Linear(hierarchy.object_shape[-1], config.hidden_dim)
        self.semantic_projection = nn.Linear(hierarchy.semantic_shape[0], config.hidden_dim)
        self.queries = nn.Parameter(torch.randn(len(config.channels), config.query_tokens, config.hidden_dim) * 0.02)
        self.texture_positions = nn.Parameter(torch.randn(hierarchy.texture_shape[0] * 64, config.hidden_dim) * 0.02)
        self.object_positions = nn.Parameter(torch.randn(hierarchy.object_shape[0] * hierarchy.object_shape[1], config.hidden_dim) * 0.02)
        self.level_embeddings = nn.Parameter(torch.randn(3, config.hidden_dim) * 0.02)
        self.layers = nn.ModuleList(CrossAttentionBlock(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, state: SceneState, channel: EvidenceChannel) -> Tensor:
        batch, frames, height, width, channels = state.texture.shape
        texture = state.texture.permute(0, 1, 4, 2, 3).reshape(batch * frames, channels, height, width)
        texture = F.adaptive_avg_pool2d(texture, (8, 8)).flatten(start_dim=2).transpose(1, 2)
        texture = self.texture_projection(texture.reshape(batch, frames * 64, channels))
        texture = texture + self.texture_positions + self.level_embeddings[0]
        objects = self.object_projection(state.objects.flatten(start_dim=1, end_dim=2))
        objects = objects + self.object_positions + self.level_embeddings[1]
        semantics = self.semantic_projection(state.semantics).unsqueeze(1) + self.level_embeddings[2]
        if channel in ("spatial", "dynamic"):
            memory = torch.cat((texture, objects, semantics), dim=1)
            weights = memory.new_ones(memory.shape[1])
            weights[:texture.shape[1]] = 1.0 / texture.shape[1]
            weights[texture.shape[1]:-1] = 1.0 / objects.shape[1]
            weights[-1] = 0.25
        else:
            memory = torch.cat((objects, semantics), dim=1)
            weights = memory.new_full((memory.shape[1],), 0.25 / objects.shape[1])
            weights[-1] = 1.0
        index = self.config.channels.index(channel)
        queries = self.queries[index].unsqueeze(0).expand(batch, -1, -1)
        memory_bias = weights.log().unsqueeze(0).expand(queries.shape[1], -1)
        for layer in self.layers:
            queries = layer(queries, memory, memory_bias)
        return self.norm(queries)
