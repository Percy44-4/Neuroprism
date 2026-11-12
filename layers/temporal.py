import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FrameNormalization(nn.Module):
    def __init__(self, channels: int, groups: int = 16) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(math.gcd(channels, groups), channels, eps=1e-6)

    def forward(self, video: Tensor) -> Tensor:
        batch, channels, frames, height, width = video.shape
        flat = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        flat = self.norm(flat)
        return flat.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


class TemporalConvolution(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = FrameNormalization(channels)
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
        )
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1)
        self.scale = nn.Parameter(torch.full((channels,), 0.1))

    def forward(self, video: Tensor) -> Tensor:
        hidden = self.depthwise(F.silu(self.norm(video)))
        hidden = self.pointwise(F.silu(hidden))
        return video + self.scale[None, :, None, None, None] * hidden


class TemporalAttention(nn.Module):
    def __init__(self, channels: int, frames: int, heads: int = 8) -> None:
        super().__init__()
        self.frames = frames
        self.positions = nn.Parameter(torch.randn(frames, channels) * 0.02)
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, channels),
        )
        self.output_scale = nn.Parameter(torch.full((channels,), 0.1))

    def sequence_transform(self, sequences: Tensor) -> Tensor:
        positioned = self.norm(sequences) + self.positions.unsqueeze(0)
        attended = self.attention(positioned, positioned, positioned, need_weights=False)[0]
        hidden = sequences + attended
        hidden = hidden + self.feed_forward(self.output_norm(hidden))
        return sequences + self.output_scale * (hidden - sequences)

    def forward(self, video: Tensor) -> Tensor:
        batch, channels, frames, height, width = video.shape
        if frames != self.frames:
            raise ValueError("decoder_frame_count")
        sequences = video.permute(0, 3, 4, 2, 1).reshape(batch * height * width, frames, channels)
        transformed = [self.sequence_transform(chunk) for chunk in sequences.split(4096, dim=0)]
        hidden = torch.cat(transformed, dim=0)
        return hidden.reshape(batch, height, width, frames, channels).permute(0, 4, 3, 1, 2)


class TemporalMixer(nn.Module):
    def __init__(self, channels: int, frames: int, use_attention: bool) -> None:
        super().__init__()
        self.convolution = TemporalConvolution(channels)
        self.attention = TemporalAttention(channels, frames) if use_attention else None

    def forward(self, video: Tensor) -> Tensor:
        hidden = self.convolution(video)
        if self.attention is not None:
            hidden = self.attention(hidden)
        return hidden
