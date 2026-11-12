import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..backbones.language import CausalLanguageModel
from ..config import EvidenceConfig, LanguageConfig, ReadoutConfig
from ..layers.attention import MultiHeadSelfAttention
from ..layers.readout import SharedCrossAttentionReader
from ..structures import (
    ContinuousEvidence,
    DiscreteUncertainty,
    EvidencePrediction,
    LanguageEvidence,
    SceneState,
)
from .decoding import DynamicEvidenceDecoder, SpatialEvidenceDecoder


class SpatialEvidenceHead(nn.Module):
    def __init__(self, config: EvidenceConfig = EvidenceConfig(), hidden_dim: int = 768) -> None:
        super().__init__()
        self.decoder = SpatialEvidenceDecoder(config, hidden_dim)

    def forward(self, features: Tensor) -> ContinuousEvidence:
        return self.decoder(features)


class DynamicEvidenceHead(nn.Module):
    def __init__(self, config: EvidenceConfig = EvidenceConfig(), hidden_dim: int = 768) -> None:
        super().__init__()
        self.decoder = DynamicEvidenceDecoder(config, hidden_dim)

    def forward(self, features: Tensor) -> ContinuousEvidence:
        return self.decoder(features)


class SemanticEvidenceHead(nn.Module):
    def __init__(self, hidden_dim: int = 768, concepts: int = 51) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, concepts)
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.classifier(features.mean(dim=1))


class GPT2LanguageHead(nn.Module):
    def __init__(
        self,
        config: ReadoutConfig = ReadoutConfig(),
        language_config: LanguageConfig = LanguageConfig(),
    ) -> None:
        super().__init__()
        if language_config.prefix_length < 1 or config.max_language_tokens < 1:
            raise ValueError("language_sequence_length")
        if language_config.prefix_length + config.max_language_tokens - 1 > language_config.max_length:
            raise ValueError("language_position_capacity")
        self.config = config
        self.decoder = CausalLanguageModel(language_config)
        self.prefix_length = language_config.prefix_length
        self.prefix_projection = nn.Linear(config.hidden_dim, language_config.width)

    def prefix(self, features: Tensor) -> Tensor:
        pooled = F.adaptive_avg_pool1d(features.transpose(1, 2), self.prefix_length).transpose(1, 2)
        return self.prefix_projection(pooled)

    def sequence_logits(self, features: Tensor, tokens: Tensor, attention_mask: Tensor) -> Tensor:
        prefix = self.prefix(features)
        words = self.decoder.get_input_embeddings()(tokens[:, :-1])
        embedded = torch.cat((prefix, words), dim=1)
        mask = torch.cat((attention_mask.new_ones((tokens.shape[0], self.prefix_length)), attention_mask[:, :-1]), dim=1)
        outputs = self.decoder(inputs_embeds=embedded, attention_mask=mask)
        return outputs[:, self.prefix_length - 1:]

    def forward(self, features: Tensor) -> LanguageEvidence:
        batch = features.shape[0]
        prefix = self.prefix(features)
        generated = torch.empty((batch, 0), device=features.device, dtype=torch.long)
        active = torch.ones(batch, device=features.device, dtype=torch.bool)
        logits_sequence, masks = [], []
        for _ in range(self.config.max_language_tokens):
            words = self.decoder.get_input_embeddings()(generated)
            outputs = self.decoder(inputs_embeds=torch.cat((prefix, words), dim=1))
            logits = outputs[:, -1]
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(active, next_token, self.decoder.config.pad_token_id)
            logits_sequence.append(logits)
            masks.append(active.long())
            generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)
            active = active & (next_token != self.decoder.config.eos_token_id)
        return LanguageEvidence(torch.stack(logits_sequence, dim=1), generated, torch.stack(masks, dim=1))

    def sequence_nll_samples(
        self,
        features: Tensor,
        token_ids: Tensor,
        attention_mask: Tensor,
        samples: int,
    ) -> Tensor:
        if samples < 1:
            raise ValueError("language_mc_samples")
        modes = [(module, module.training) for module in self.decoder.modules()]
        values = []
        try:
            for module, _ in modes:
                if isinstance(module, (nn.Dropout, MultiHeadSelfAttention)):
                    module.training = True
            for _ in range(samples):
                logits = self.sequence_logits(features, token_ids, attention_mask)
                nll = F.cross_entropy(logits.transpose(1, 2), token_ids, reduction="none")
                values.append((nll * attention_mask).sum(dim=1))
        finally:
            for module, mode in modes:
                module.training = mode
        return torch.stack(values)


class EvidenceField(nn.Module):
    def __init__(
        self,
        reader: SharedCrossAttentionReader,
        spatial: SpatialEvidenceHead,
        semantic: SemanticEvidenceHead,
        language: GPT2LanguageHead,
        dynamic: DynamicEvidenceHead,
    ) -> None:
        super().__init__()
        self.reader = reader
        self.spatial = spatial
        self.semantic = semantic
        self.language = language
        self.dynamic = dynamic

    def forward(self, state: SceneState) -> EvidencePrediction:
        return EvidencePrediction(
            spatial=self.spatial(self.reader(state, "spatial")),
            semantic_logits=self.semantic(self.reader(state, "semantic")),
            language=self.language(self.reader(state, "language")),
            dynamic=self.dynamic(self.reader(state, "dynamic")),
        )

    def uncertainty(self, state: SceneState, evidence: EvidencePrediction) -> DiscreteUncertainty:
        probabilities = evidence.semantic_logits.sigmoid()
        semantic = -(probabilities * F.logsigmoid(evidence.semantic_logits)).sum(dim=-1)
        features = self.reader(state, "language")
        sequence_nll = self.language.sequence_nll_samples(
            features,
            evidence.language.token_ids,
            evidence.language.attention_mask,
            self.reader.config.language_mc_samples,
        )
        return DiscreteUncertainty(semantic, sequence_nll.var(dim=0, unbiased=False))
