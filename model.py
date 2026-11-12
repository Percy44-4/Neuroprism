from torch import Tensor, nn

from .backbones.brain import MindEye2Encoder
from .config import NeuroPrismConfig
from .heads.evidence import (
    DynamicEvidenceHead,
    EvidenceField,
    GPT2LanguageHead,
    SemanticEvidenceHead,
    SpatialEvidenceHead,
)
from .layers.hierarchy import (
    BottomUpEvidence,
    GaussianSceneInitializer,
    PredictiveCodingInference,
    TopDownPredictor,
)
from .layers.readout import SharedCrossAttentionReader
from .layers.reliability import DiscreteReliabilityCalibration, ReliabilityEstimator
from .structures import ModelPrediction


class NeuroPrism(nn.Module):
    def __init__(self, config: NeuroPrismConfig = NeuroPrismConfig()) -> None:
        super().__init__()
        if config.brain.embedding_dim != config.hierarchy.embedding_dim:
            raise ValueError("brain_hierarchy_embedding_dimension")
        if config.hierarchy.texture_shape[0] != config.hierarchy.object_shape[0]:
            raise ValueError("hierarchy_frame_count")
        channels = ("spatial", "semantic", "language", "dynamic")
        if len(config.readout.channels) != len(channels) or set(config.readout.channels) != set(channels):
            raise ValueError("evidence_channels")
        self.config = config
        self.brain = MindEye2Encoder(config.brain)
        self.hierarchy = PredictiveCodingInference(
            GaussianSceneInitializer(config.hierarchy, config.scene),
            BottomUpEvidence(config.hierarchy, config.scene),
            TopDownPredictor(config.hierarchy, config.scene),
            config.hierarchy,
        )
        self.evidence = EvidenceField(
            SharedCrossAttentionReader(config.readout, config.hierarchy),
            SpatialEvidenceHead(config.evidence, config.readout.hidden_dim),
            SemanticEvidenceHead(config.readout.hidden_dim, config.evidence.concept_count),
            GPT2LanguageHead(config.readout, config.language),
            DynamicEvidenceHead(config.evidence, config.readout.hidden_dim),
        )
        self.reliability = ReliabilityEstimator(DiscreteReliabilityCalibration())
        self.brain.requires_grad_(False)
        self.brain.eval()

    def train(self, mode: bool = True) -> "NeuroPrism":
        super().train(mode)
        self.brain.eval()
        return self

    def forward(self, fmri: Tensor, subject_ids: Tensor) -> ModelPrediction:
        brain = self.brain(fmri, subject_ids)
        state = self.hierarchy(brain)
        evidence = self.evidence(state)
        uncertainty = self.evidence.uncertainty(state, evidence)
        reliability = self.reliability(evidence, uncertainty)
        return ModelPrediction(evidence, uncertainty, reliability)
