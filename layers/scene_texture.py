import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import HierarchyConfig, SceneNetworkConfig
from .scene import ConditionedSpatialResidual, FeatureTokenProjection, FourierCoordinateEmbedding, SceneAttention, TextureTemporalMixer


class TextureRefinementStage(nn.Module):
    def __init__(
        self,
        channels: int,
        condition_dim: int,
        layers: int,
        coordinate_frequencies: int,
        temporal_kernel: int,
    ) -> None:
        super().__init__()
        self.coordinates = FourierCoordinateEmbedding(channels, coordinate_frequencies)
        self.spatial_blocks = nn.ModuleList(
            ConditionedSpatialResidual(channels, condition_dim) for _ in range(layers)
        )
        self.temporal_mixer = TextureTemporalMixer(channels, temporal_kernel)

    def forward(self, video: Tensor, condition: Tensor, size: tuple[int, int]) -> Tensor:
        batch, frames, height, width, channels = video.shape
        images = video.permute(0, 1, 4, 2, 3).reshape(batch * frames, channels, height, width)
        images = F.interpolate(images, size=size, mode="bilinear", align_corners=False)
        coordinates = self.coordinates(size[0], size[1], images).permute(2, 0, 1)
        images = images + coordinates.unsqueeze(0)
        frame_condition = condition.repeat_interleave(frames, dim=0)
        for block in self.spatial_blocks:
            images = block(images, frame_condition)
        video = images.reshape(batch, frames, channels, *size).permute(0, 1, 3, 4, 2)
        return self.temporal_mixer(video)


class TextureGridDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hierarchy: HierarchyConfig,
        config: SceneNetworkConfig = SceneNetworkConfig(),
    ) -> None:
        super().__init__()
        self.shape = hierarchy.texture_shape
        self.seed_size = hierarchy.seed_resolution
        self.hidden_dim = config.texture_width
        frames, height, width, output_dim = self.shape
        self.context = FeatureTokenProjection(input_dim, config.texture_memory_tokens, config.texture_width)
        self.seed = nn.Parameter(torch.randn(1, frames, self.seed_size, self.seed_size, config.texture_width) * 0.02)
        self.frame_positions = nn.Embedding(frames, config.texture_width)
        self.seed_coordinates = FourierCoordinateEmbedding(config.texture_width, config.coordinate_frequencies)
        self.cross_attention = SceneAttention(config.texture_width, config.texture_attention_heads)
        self.query_normalization = nn.LayerNorm(config.texture_width)
        self.context_normalization = nn.LayerNorm(config.texture_width)
        self.condition_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, config.texture_width), nn.SiLU())
        self.stage_sizes = self.resolutions(height, width)
        self.stages = nn.ModuleList(
            TextureRefinementStage(
                config.texture_width,
                config.texture_width,
                config.texture_residual_layers,
                config.coordinate_frequencies,
                config.temporal_kernel,
            )
            for _ in self.stage_sizes
        )
        self.output_normalization = nn.LayerNorm(config.texture_width)
        self.output_projection = nn.Linear(config.texture_width, output_dim)

    def resolutions(self, height: int, width: int) -> tuple[tuple[int, int], ...]:
        sizes = [(min(self.seed_size, height), min(self.seed_size, width))]
        while sizes[-1] != (height, width):
            current_height, current_width = sizes[-1]
            sizes.append((min(current_height * 2, height), min(current_width * 2, width)))
        return tuple(sizes)

    def forward(self, features: Tensor) -> Tensor:
        batch = features.shape[0]
        frames = self.shape[0]
        context = self.context(features)
        condition = self.condition_projection(features)
        seed = self.seed.expand(batch, -1, -1, -1, -1)
        coordinates = self.seed_coordinates(self.seed_size, self.seed_size, features)
        frame_ids = torch.arange(frames, device=features.device)
        frame_positions = self.frame_positions(frame_ids).reshape(1, frames, 1, 1, self.hidden_dim)
        seed = seed + coordinates[None, None] + frame_positions
        queries = seed.reshape(batch, -1, self.hidden_dim)
        attended = self.cross_attention(self.query_normalization(queries), self.context_normalization(context))
        video = (queries + attended).reshape_as(seed)
        for stage, size in zip(self.stages, self.stage_sizes):
            video = stage(video, condition, size)
        return self.output_projection(self.output_normalization(video))
