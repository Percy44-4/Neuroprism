from torch import Tensor, nn

from ..config import LanguageConfig
from ..layers.embeddings import LearnedPositionEmbedding
from ..layers.transformer import PrecisionLayerNorm, TransformerStack


class CausalLanguageModel(nn.Module):
    def __init__(self, config: LanguageConfig = LanguageConfig()) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.width)
        self.position_embedding = LearnedPositionEmbedding(config.max_length, config.width, config.initializer_range)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.transformer = TransformerStack(
            config.width,
            config.layers,
            config.heads,
            config.mlp_ratio,
            config.dropout,
            config.layer_norm_epsilon,
        )
        self.final_norm = PrecisionLayerNorm(config.width, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.width, config.vocabulary_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self.initialize)
        residual_scale = config.initializer_range / (2 * config.layers) ** 0.5
        for block in self.transformer.blocks:
            nn.init.normal_(block.attention.output.weight, std=residual_scale)
            nn.init.normal_(block.feed_forward.contract.weight, std=residual_scale)

    def initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embedding

    def forward(
        self,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide_tokens_or_embeddings")
        hidden = self.token_embedding(input_ids) if inputs_embeds is None else inputs_embeds
        hidden = self.embedding_dropout(self.position_embedding(hidden))
        hidden = self.transformer(hidden, attention_mask)
        return self.lm_head(self.final_norm(hidden))
