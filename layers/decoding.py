from torch import Tensor, nn
from torch.nn import functional as F

from .seed import ReadoutCrossAttention
from .temporal import FrameNormalization, TemporalMixer


class FrameConditionedResidual(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, context_dim: int = 256) -> None:
        super().__init__()
        self.input_norm = FrameNormalization(input_channels)
        self.input_conv = nn.Conv3d(
            input_channels,
            output_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        self.context_projection = nn.Linear(context_dim, 2 * output_channels)
        self.output_norm = FrameNormalization(output_channels)
        self.output_conv = nn.Conv3d(
            output_channels,
            output_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        self.shortcut = (
            nn.Conv3d(input_channels, output_channels, kernel_size=1)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, video: Tensor, frame_context: Tensor) -> Tensor:
        hidden = self.input_conv(F.silu(self.input_norm(video)))
        scale, shift = self.context_projection(frame_context).chunk(2, dim=-1)
        scale = scale.transpose(1, 2)[:, :, :, None, None]
        shift = shift.transpose(1, 2)[:, :, :, None, None]
        hidden = self.output_norm(hidden) * (1 + scale) + shift
        hidden = self.output_conv(F.silu(hidden))
        return hidden + self.shortcut(video)


class ReadoutSpatialAttention(nn.Module):
    def __init__(self, channels: int, source_dim: int = 768) -> None:
        super().__init__()
        self.norm = FrameNormalization(channels)
        self.cross_attention = ReadoutCrossAttention(channels, source_dim)
        self.output_projection = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, video: Tensor, readout: Tensor) -> Tensor:
        batch, channels, frames, height, width = video.shape
        normalized = self.norm(video)
        positions = normalized.permute(0, 2, 3, 4, 1).reshape(batch, frames * height * width, channels)
        attended = self.cross_attention(positions, readout)
        attended = attended.reshape(batch, frames, height, width, channels).permute(0, 4, 1, 2, 3)
        return video + self.output_projection(attended)


class VideoUpsamplingStage(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        frames: int,
        source_dim: int = 768,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.first_residual = FrameConditionedResidual(input_channels, output_channels)
        self.second_residual = FrameConditionedResidual(output_channels, output_channels)
        self.temporal = TemporalMixer(output_channels, frames, use_attention)
        self.cross_attention = (
            ReadoutSpatialAttention(output_channels, source_dim)
            if use_attention
            else None
        )
        self.upsample_convolution = nn.Conv3d(
            output_channels,
            output_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )

    def forward(self, video: Tensor, frame_context: Tensor, readout: Tensor) -> Tensor:
        hidden = self.first_residual(video, frame_context)
        hidden = self.second_residual(hidden, frame_context)
        if self.cross_attention is not None:
            hidden = self.cross_attention(hidden, readout)
        hidden = self.temporal(hidden)
        hidden = F.interpolate(hidden, scale_factor=(1, 2, 2), mode="nearest")
        return self.upsample_convolution(hidden)


class VideoDecoderBottleneck(nn.Module):
    def __init__(self, channels: int, frames: int, source_dim: int) -> None:
        super().__init__()
        self.input_residual = FrameConditionedResidual(channels, channels)
        self.attention = ReadoutSpatialAttention(channels, source_dim)
        self.temporal = TemporalMixer(channels, frames, True)
        self.output_residual = FrameConditionedResidual(channels, channels)

    def forward(self, video: Tensor, frame_context: Tensor, readout: Tensor) -> Tensor:
        hidden = self.input_residual(video, frame_context)
        hidden = self.attention(hidden, readout)
        hidden = self.temporal(hidden)
        return self.output_residual(hidden, frame_context)
