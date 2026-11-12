import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import BrainEncoderConfig
from ..structures import BrainEmbeddings


class ResidualBrainBlock(nn.Module):
    def __init__(self, width: int = 4096, dropout: float = 0.15) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(width)
        self.expand = nn.Linear(width, width)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.contract = nn.Linear(width, width)

    def forward(self, features: Tensor) -> Tensor:
        hidden = self.input_norm(features)
        hidden = self.expand(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.contract(hidden)
        return features + hidden


class BrainTokenMixer(nn.Module):
    def __init__(self, tokens: int, width: int, dropout: float) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(width)
        self.token_network = nn.Sequential(
            nn.Linear(tokens, tokens * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tokens * 2, tokens),
        )
        self.channel_norm = nn.LayerNorm(width)
        self.channel_network = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        mixed = self.token_norm(tokens).transpose(1, 2)
        mixed = self.token_network(mixed).transpose(1, 2)
        tokens = tokens + mixed
        return tokens + self.channel_network(self.channel_norm(tokens))


class BrainEmbeddingProjector(nn.Module):
    def __init__(self, input_width: int, hidden_width: int, output_width: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_width)
        self.layers = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, output_width),
        )

    def forward(self, features: Tensor) -> Tensor:
        return F.normalize(self.layers(self.input_norm(features)), dim=-1)


class MindEye2Encoder(nn.Module):
    def __init__(self, config: BrainEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.subject_voxels = config.subject_voxels
        width = config.width
        embedding_dim = config.embedding_dim
        self.subject_projection = nn.ModuleList(
            nn.Linear(count, width) for count in self.subject_voxels
        )
        self.input_norm = nn.LayerNorm(width)
        self.backbone = nn.ModuleList(
            ResidualBrainBlock(width, self.config.dropout)
            for _ in range(self.config.blocks)
        )
        self.token_projection = nn.Linear(width, self.config.tokens * self.config.token_width)
        self.token_position = nn.Parameter(
            torch.randn(1, self.config.tokens, self.config.token_width) * 0.02
        )
        self.token_mixers = nn.ModuleList(
            BrainTokenMixer(self.config.tokens, self.config.token_width, self.config.dropout)
            for _ in range(2)
        )
        flattened_width = self.config.tokens * self.config.token_width
        self.visual = BrainEmbeddingProjector(
            flattened_width, self.config.projection_width, embedding_dim,
        )
        self.text = BrainEmbeddingProjector(
            flattened_width, self.config.projection_width, embedding_dim,
        )

    def project_subjects(self, fmri: Tensor, subject_ids: Tensor) -> Tensor:
        fmri = fmri.reshape(fmri.shape[0], -1)
        subject_ids = subject_ids.reshape(-1)
        if torch.any(subject_ids < 0) or torch.any(subject_ids >= len(self.subject_voxels)):
            raise ValueError("invalid_subject_index")
        encoded = fmri.new_zeros((fmri.shape[0], self.config.width))
        for index, projection in enumerate(self.subject_projection):
            selected = subject_ids == index
            count = self.subject_voxels[index]
            encoded[selected] = projection(fmri[selected, :count])
        return self.input_norm(encoded)

    def forward_features(self, fmri: Tensor, subject_ids: Tensor) -> Tensor:
        hidden = self.project_subjects(fmri, subject_ids)
        for block in self.backbone:
            hidden = block(hidden)
        tokens = self.token_projection(hidden)
        tokens = tokens.reshape(hidden.shape[0], self.config.tokens, self.config.token_width)
        tokens = tokens + self.token_position
        for mixer in self.token_mixers:
            tokens = mixer(tokens)
        return tokens

    def forward(self, fmri: Tensor, subject_ids: Tensor) -> BrainEmbeddings:
        tokens = self.forward_features(fmri, subject_ids)
        features = tokens.flatten(start_dim=1)
        return BrainEmbeddings(self.visual(features), self.text(features))
