from torch import Tensor, nn
from torch.nn import functional as F

from ..config import EvidenceConfig
from ..structures import ContinuousEvidence
from ..layers.decoding import FrameConditionedResidual, VideoDecoderBottleneck, VideoUpsamplingStage
from ..layers.seed import QueryGridSeed
from ..layers.temporal import FrameNormalization


class GaussianEvidenceOutput(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, bounded_mean: bool) -> None:
        super().__init__()
        self.bounded_mean = bounded_mean
        self.mean_norm = FrameNormalization(input_channels)
        self.mean_hidden = nn.Conv3d(
            input_channels,
            input_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        self.mean_output = nn.Conv3d(input_channels, output_channels, kernel_size=1)
        self.variance_norm = FrameNormalization(input_channels)
        self.variance_hidden = nn.Conv3d(
            input_channels,
            input_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        self.variance_output = nn.Conv3d(input_channels, output_channels, kernel_size=1)

    def forward(self, video: Tensor, target_size: tuple[int, int]) -> ContinuousEvidence:
        mean_hidden = self.mean_hidden(F.silu(self.mean_norm(video)))
        mean = self.mean_output(F.silu(mean_hidden))
        variance_hidden = self.variance_hidden(F.silu(self.variance_norm(video)))
        log_variance = self.variance_output(F.silu(variance_hidden))
        if video.shape[-2:] != target_size:
            shape = (video.shape[2], *target_size)
            mean = F.interpolate(mean, size=shape, mode="trilinear", align_corners=False)
            log_variance = F.interpolate(log_variance, size=shape, mode="trilinear", align_corners=False)
        if self.bounded_mean:
            mean = mean.sigmoid()
        log_variance = log_variance.clamp(-8, 6)
        return ContinuousEvidence(
            mean.permute(0, 2, 3, 4, 1),
            log_variance.permute(0, 2, 3, 4, 1),
        )


class SpatialEvidenceDecoder(nn.Module):
    def __init__(self, config: EvidenceConfig = EvidenceConfig(), hidden_dim: int = 768) -> None:
        super().__init__()
        self.config = config
        self.seed = QueryGridSeed(config.frames, source_dim=hidden_dim)
        self.bottleneck = VideoDecoderBottleneck(256, config.frames, hidden_dim)
        widths = (256, 256, 128, 96, 64, 32)
        self.stages = nn.ModuleList(
            VideoUpsamplingStage(
                widths[index],
                widths[index + 1],
                config.frames,
                source_dim=hidden_dim,
                use_attention=index < 2,
            )
            for index in range(5)
        )
        self.refinement = FrameConditionedResidual(32, 32)
        self.output = GaussianEvidenceOutput(32, 1, bounded_mean=True)

    def forward(self, features: Tensor) -> ContinuousEvidence:
        seed = self.seed(features)
        hidden = self.bottleneck(seed.grid, seed.frame_context, features)
        for stage in self.stages:
            hidden = stage(hidden, seed.frame_context, features)
        hidden = self.refinement(hidden, seed.frame_context)
        return self.output(hidden, (self.config.height, self.config.width))


class DynamicEvidenceDecoder(nn.Module):
    def __init__(self, config: EvidenceConfig = EvidenceConfig(), hidden_dim: int = 768) -> None:
        super().__init__()
        self.config = config
        self.seed = QueryGridSeed(config.frames, source_dim=hidden_dim)
        self.bottleneck = VideoDecoderBottleneck(256, config.frames, hidden_dim)
        self.stages = nn.ModuleList((
            VideoUpsamplingStage(256, 128, config.frames, source_dim=hidden_dim, use_attention=True),
            VideoUpsamplingStage(128, 64, config.frames, source_dim=hidden_dim, use_attention=True),
        ))
        self.refinement = FrameConditionedResidual(64, 64)
        self.output = GaussianEvidenceOutput(64, 4, bounded_mean=False)

    def forward(self, features: Tensor) -> ContinuousEvidence:
        seed = self.seed(features)
        hidden = self.bottleneck(seed.grid, seed.frame_context, features)
        for stage in self.stages:
            hidden = stage(hidden, seed.frame_context, features)
        hidden = self.refinement(hidden, seed.frame_context)
        return self.output(hidden, (self.config.height // 8, self.config.width // 8))
