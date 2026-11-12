from dataclasses import dataclass

from torch import Tensor


@dataclass
class BrainEmbeddings:
    visual: Tensor
    text: Tensor


@dataclass
class SceneState:
    texture: Tensor
    objects: Tensor
    semantics: Tensor

    def levels(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.texture, self.objects, self.semantics


@dataclass
class GaussianScene:
    mean: SceneState
    log_variance: SceneState


@dataclass
class ContinuousEvidence:
    mean: Tensor
    log_variance: Tensor


@dataclass
class LanguageEvidence:
    logits: Tensor
    token_ids: Tensor
    attention_mask: Tensor


@dataclass
class EvidencePrediction:
    spatial: ContinuousEvidence
    semantic_logits: Tensor
    language: LanguageEvidence
    dynamic: ContinuousEvidence


@dataclass
class DiscreteUncertainty:
    semantic: Tensor
    language: Tensor


@dataclass
class ChannelReliability:
    spatial: Tensor
    semantic: Tensor
    language: Tensor
    dynamic: Tensor


@dataclass
class GatedConditions:
    semantic_context: Tensor
    control_map: Tensor
    dynamic_bias: Tensor


@dataclass
class ModelPrediction:
    evidence: EvidencePrediction
    uncertainty: DiscreteUncertainty
    reliability: ChannelReliability
