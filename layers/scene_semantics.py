import torch
from torch import Tensor, nn

from ..config import SceneNetworkConfig
from .scene import FeatureTokenProjection, SceneTransformerBlock


class GlobalSemanticEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: SceneNetworkConfig = SceneNetworkConfig(),
    ) -> None:
        super().__init__()
        self.tokens = FeatureTokenProjection(input_dim, config.semantic_tokens, output_dim)
        self.summary_token = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
        self.summary_condition = nn.Linear(input_dim, output_dim)
        self.layers = nn.ModuleList(
            SceneTransformerBlock(
                output_dim,
                config.semantic_attention_heads,
                config.feed_forward_expansion,
            )
            for _ in range(config.semantic_layers)
        )
        self.summary_normalization = nn.LayerNorm(output_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim, 2 * output_dim),
            nn.GELU(),
            nn.Linear(2 * output_dim, output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        tokens = self.tokens(features)
        summary = self.summary_token + self.summary_condition(features).unsqueeze(1)
        tokens = torch.cat((summary, tokens), dim=1)
        for layer in self.layers:
            tokens = layer(tokens)
        pooled = self.summary_normalization(tokens[:, 0])
        return pooled + self.output_projection(pooled)
