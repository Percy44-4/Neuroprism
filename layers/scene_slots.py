from math import sqrt

import torch
from torch import Tensor, nn

from ..config import HierarchyConfig, SceneNetworkConfig
from .scene import FeatureTokenProjection, SceneFeedForward, SceneTransformerBlock


class SlotRecurrentUpdate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(channels, 3 * channels)
        self.state_projection = nn.Linear(channels, 3 * channels)

    def forward(self, evidence: Tensor, previous: Tensor) -> Tensor:
        input_reset, input_update, input_candidate = self.input_projection(evidence).chunk(3, dim=-1)
        state_reset, state_update, state_candidate = self.state_projection(previous).chunk(3, dim=-1)
        reset = (input_reset + state_reset).sigmoid()
        update = (input_update + state_update).sigmoid()
        candidate = (input_candidate + reset * state_candidate).tanh()
        return (1.0 - update) * candidate + update * previous


class IterativeSlotAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        slots: int,
        iterations: int = 3,
        expansion: int = 2,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.slots = slots
        self.iterations = iterations
        self.epsilon = epsilon
        self.slot_templates = nn.Parameter(torch.randn(1, slots, channels) * 0.02)
        self.input_normalization = nn.LayerNorm(channels)
        self.slot_normalization = nn.LayerNorm(channels)
        self.key_projection = nn.Linear(channels, channels, bias=False)
        self.value_projection = nn.Linear(channels, channels, bias=False)
        self.query_projection = nn.Linear(channels, channels, bias=False)
        self.update = SlotRecurrentUpdate(channels)
        self.feed_forward = SceneFeedForward(channels, expansion)
        self.output_normalization = nn.LayerNorm(channels)

    def initial_slots(self, memory: Tensor, condition: Tensor) -> Tensor:
        templates = self.slot_templates.expand(memory.shape[0], -1, -1)
        return templates + condition.unsqueeze(1)

    def assignments(self, slots: Tensor, keys: Tensor) -> Tensor:
        queries = self.query_projection(self.slot_normalization(slots))
        logits = torch.matmul(queries, keys.transpose(-1, -2)) / sqrt(self.channels)
        competition = logits.softmax(dim=1) + self.epsilon
        return competition / competition.sum(dim=-1, keepdim=True)

    def forward(self, memory: Tensor, condition: Tensor) -> Tensor:
        normalized_memory = self.input_normalization(memory)
        keys = self.key_projection(normalized_memory)
        values = self.value_projection(normalized_memory)
        slots = self.initial_slots(memory, condition)
        for _ in range(self.iterations):
            weights = self.assignments(slots, keys)
            evidence = torch.matmul(weights, values)
            slots = self.update(
                evidence.reshape(-1, self.channels),
                slots.reshape(-1, self.channels),
            ).reshape(memory.shape[0], self.slots, self.channels)
            slots = self.feed_forward(slots)
        return self.output_normalization(slots)


class SlotTemporalRefinement(nn.Module):
    def __init__(self, channels: int, frames: int, heads: int, expansion: int) -> None:
        super().__init__()
        self.frame_positions = nn.Parameter(torch.randn(1, frames, channels) * 0.02)
        self.transformer = SceneTransformerBlock(channels, heads, expansion)
        self.normalization = nn.LayerNorm(channels)

    def forward(self, slots: Tensor) -> Tensor:
        batch, frames, count, channels = slots.shape
        tracks = slots.permute(0, 2, 1, 3).reshape(batch * count, frames, channels)
        tracks = self.transformer(tracks + self.frame_positions)
        tracks = self.normalization(tracks)
        return tracks.reshape(batch, count, frames, channels).permute(0, 2, 1, 3)


class ObjectSlotEncoder(nn.Module):
    def __init__(
        self,
        hierarchy: HierarchyConfig,
        config: SceneNetworkConfig = SceneNetworkConfig(),
    ) -> None:
        super().__init__()
        self.shape = hierarchy.object_shape
        frames, slots, channels = self.shape
        self.visual_tokens = FeatureTokenProjection(hierarchy.embedding_dim, config.visual_tokens, channels)
        self.text_tokens = FeatureTokenProjection(hierarchy.embedding_dim, config.text_tokens, channels)
        self.modality_positions = nn.Parameter(torch.randn(2, channels) * 0.02)
        self.frame_positions = nn.Parameter(torch.randn(1, frames, 1, channels) * 0.02)
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(2 * hierarchy.embedding_dim),
            nn.Linear(2 * hierarchy.embedding_dim, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.slot_attention = IterativeSlotAttention(
            channels,
            slots,
            config.slot_iterations,
            config.slot_mlp_expansion,
            config.attention_epsilon,
        )
        self.temporal_refinement = SlotTemporalRefinement(
            channels,
            frames,
            config.slot_attention_heads,
            config.feed_forward_expansion,
        )

    def forward(self, visual: Tensor, text: Tensor) -> Tensor:
        batch = visual.shape[0]
        frames, count, channels = self.shape
        visual_tokens = self.visual_tokens(visual) + self.modality_positions[0]
        text_tokens = self.text_tokens(text) + self.modality_positions[1]
        memory = torch.cat((visual_tokens, text_tokens), dim=1)
        memory = memory[:, None] + self.frame_positions
        memory = memory.reshape(batch * frames, -1, channels)
        condition = self.condition_projection(torch.cat((visual, text), dim=-1))
        condition = condition.repeat_interleave(frames, dim=0)
        slots = self.slot_attention(memory, condition)
        slots = slots.reshape(batch, frames, count, channels)
        return self.temporal_refinement(slots)


class SemanticObjectPredictor(nn.Module):
    def __init__(
        self,
        hierarchy: HierarchyConfig,
        config: SceneNetworkConfig = SceneNetworkConfig(),
    ) -> None:
        super().__init__()
        self.shape = hierarchy.object_shape
        frames, slots, channels = self.shape
        self.semantic_tokens = FeatureTokenProjection(
            hierarchy.semantic_shape[0], config.semantic_tokens, channels
        )
        self.frame_positions = nn.Parameter(torch.randn(1, frames, 1, channels) * 0.02)
        self.condition_projection = nn.Linear(hierarchy.semantic_shape[0], channels)
        self.slot_attention = IterativeSlotAttention(
            channels,
            slots,
            config.slot_iterations,
            config.slot_mlp_expansion,
            config.attention_epsilon,
        )
        self.temporal_refinement = SlotTemporalRefinement(
            channels,
            frames,
            config.slot_attention_heads,
            config.feed_forward_expansion,
        )

    def forward(self, semantics: Tensor) -> Tensor:
        batch = semantics.shape[0]
        frames, slots, channels = self.shape
        memory = self.semantic_tokens(semantics)[:, None] + self.frame_positions
        memory = memory.reshape(batch * frames, -1, channels)
        condition = self.condition_projection(semantics).repeat_interleave(frames, dim=0)
        objects = self.slot_attention(memory, condition).reshape(batch, frames, slots, channels)
        return self.temporal_refinement(objects)
