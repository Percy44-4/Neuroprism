import torch
from torch import Tensor, nn

from ..structures import BrainEmbeddings, GaussianScene, SceneState
from ..config import HierarchyConfig, SceneNetworkConfig
from .scene_semantics import GlobalSemanticEncoder
from .scene_slots import ObjectSlotEncoder
from .scene_texture import TextureGridDecoder


class GaussianChannelHead(nn.Module):
    def __init__(self, channels: int, config: SceneNetworkConfig) -> None:
        super().__init__()
        self.minimum_log_variance = config.minimum_log_variance
        self.maximum_log_variance = config.maximum_log_variance
        self.normalization = nn.LayerNorm(channels, eps=config.norm_epsilon)
        self.mean_projection = nn.Linear(channels, channels)
        self.variance_projection = nn.Linear(channels, channels)
        nn.init.zeros_(self.variance_projection.weight)
        nn.init.constant_(self.variance_projection.bias, -2.0)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.normalization(features)
        mean = self.mean_projection(normalized)
        log_variance = self.variance_projection(normalized)
        log_variance = log_variance.clamp(self.minimum_log_variance, self.maximum_log_variance)
        return mean, log_variance


class SceneGaussianEncoder(nn.Module):
    def __init__(
        self,
        hierarchy: HierarchyConfig,
        config: SceneNetworkConfig = SceneNetworkConfig(),
    ) -> None:
        super().__init__()
        combined_dim = 2 * hierarchy.embedding_dim
        self.texture_features = TextureGridDecoder(combined_dim, hierarchy, config)
        self.object_features = ObjectSlotEncoder(hierarchy, config)
        self.semantic_features = GlobalSemanticEncoder(combined_dim, hierarchy.semantic_shape[0], config)
        self.texture_distribution = GaussianChannelHead(hierarchy.texture_shape[-1], config)
        self.object_distribution = GaussianChannelHead(hierarchy.object_shape[-1], config)
        self.semantic_distribution = GaussianChannelHead(hierarchy.semantic_shape[0], config)

    def forward(self, brain: BrainEmbeddings) -> GaussianScene:
        combined = torch.cat((brain.visual, brain.text), dim=-1)
        texture = self.texture_features(combined)
        objects = self.object_features(brain.visual, brain.text)
        semantics = self.semantic_features(combined)
        texture_mean, texture_variance = self.texture_distribution(texture)
        object_mean, object_variance = self.object_distribution(objects)
        semantic_mean, semantic_variance = self.semantic_distribution(semantics)
        return GaussianScene(
            SceneState(texture_mean, object_mean, semantic_mean),
            SceneState(texture_variance, object_variance, semantic_variance),
        )
