from math import pi

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import HierarchyConfig, SceneNetworkConfig
from .scene import ConditionedSpatialResidual, FourierCoordinateEmbedding, TextureTemporalMixer


class GaussianObjectRasterizer(nn.Module):
    def __init__(
        self,
        object_dim: int,
        output_dim: int,
        height: int,
        width: int,
        minimum_scale: float = 0.05,
        maximum_scale: float = 0.75,
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.minimum_scale = minimum_scale
        self.maximum_scale = maximum_scale
        self.object_normalization = nn.LayerNorm(object_dim)
        self.center_projection = nn.Linear(object_dim, 2)
        self.scale_projection = nn.Linear(object_dim, 2)
        self.rotation_projection = nn.Linear(object_dim, 1)
        self.presence_projection = nn.Linear(object_dim, 1)
        self.feature_projection = nn.Linear(object_dim, output_dim)
        rows = torch.linspace(-1.0, 1.0, height)
        columns = torch.linspace(-1.0, 1.0, width)
        coordinates = torch.stack(torch.meshgrid(rows, columns, indexing="ij"), dim=-1)
        self.register_buffer("coordinates", coordinates, persistent=False)

    def spatial_assignments(self, normalized_objects: Tensor) -> Tensor:
        centers = self.center_projection(normalized_objects).tanh()
        scales = self.minimum_scale + (
            self.maximum_scale - self.minimum_scale
        ) * self.scale_projection(normalized_objects).sigmoid()
        rotation = self.rotation_projection(normalized_objects).tanh() * pi
        log_presence = F.logsigmoid(self.presence_projection(normalized_objects))
        coordinates = self.coordinates.to(normalized_objects.dtype)
        displacement = coordinates[None, None, None] - centers[..., None, None, :]
        cosine = rotation.cos()[..., None]
        sine = rotation.sin()[..., None]
        first = cosine * displacement[..., 0] + sine * displacement[..., 1]
        second = -sine * displacement[..., 0] + cosine * displacement[..., 1]
        scaled_first = first / scales[..., 0, None, None]
        scaled_second = second / scales[..., 1, None, None]
        distance = scaled_first.square() + scaled_second.square()
        logits = -0.5 * distance + log_presence[..., None]
        return logits.softmax(dim=2)

    def forward(self, objects: Tensor) -> Tensor:
        normalized = self.object_normalization(objects)
        assignments = self.spatial_assignments(normalized)
        features = self.feature_projection(normalized)
        return torch.einsum("bfnhw,bfnc->bfhwc", assignments, features)


class ObjectTexturePredictor(nn.Module):
    def __init__(
        self,
        hierarchy: HierarchyConfig,
        config: SceneNetworkConfig = SceneNetworkConfig(),
    ) -> None:
        super().__init__()
        _, height, width, channels = hierarchy.texture_shape
        object_dim = hierarchy.object_shape[-1]
        self.rasterizer = GaussianObjectRasterizer(
            object_dim,
            channels,
            height,
            width,
            config.raster_minimum_scale,
            config.raster_maximum_scale,
        )
        self.coordinates = FourierCoordinateEmbedding(channels, config.coordinate_frequencies)
        self.spatial_blocks = nn.ModuleList(
            ConditionedSpatialResidual(channels, object_dim)
            for _ in range(config.texture_residual_layers)
        )
        self.temporal_mixer = TextureTemporalMixer(channels, config.temporal_kernel)
        self.output_normalization = nn.LayerNorm(channels)
        self.output_projection = nn.Linear(channels, channels)

    def forward(self, objects: Tensor) -> Tensor:
        texture = self.rasterizer(objects)
        batch, frames, height, width, channels = texture.shape
        coordinates = self.coordinates(height, width, texture)
        texture = texture + coordinates[None, None]
        images = texture.permute(0, 1, 4, 2, 3).reshape(batch * frames, channels, height, width)
        condition = objects.mean(dim=2).reshape(batch * frames, -1)
        for block in self.spatial_blocks:
            images = block(images, condition)
        texture = images.reshape(batch, frames, channels, height, width).permute(0, 1, 3, 4, 2)
        texture = self.temporal_mixer(texture)
        return self.output_projection(self.output_normalization(texture))
