import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..structures import ChannelReliability, DiscreteUncertainty, EvidencePrediction


class DiscreteReliabilityCalibration(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.slopes = nn.Parameter(torch.ones(2))
        self.offsets = nn.Parameter(torch.zeros(2))

    def variance(self, uncertainty: DiscreteUncertainty) -> Tensor:
        proxies = torch.stack((uncertainty.semantic, uncertainty.language), dim=-1)
        return F.softplus(self.slopes * proxies + self.offsets).clamp_min(1e-6)

    def forward(self, uncertainty: DiscreteUncertainty) -> tuple[Tensor, Tensor]:
        precision = self.variance(uncertainty).reciprocal()
        return precision[:, 0], precision[:, 1]


class ReliabilityEstimator(nn.Module):
    def __init__(self, calibration: DiscreteReliabilityCalibration) -> None:
        super().__init__()
        self.calibration = calibration

    def forward(
        self, evidence: EvidencePrediction, uncertainty: DiscreteUncertainty
    ) -> ChannelReliability:
        semantic, language = self.calibration(uncertainty)
        spatial = (-evidence.spatial.log_variance).exp().flatten(start_dim=1).mean(dim=1)
        dynamic = (-evidence.dynamic.log_variance).exp().flatten(start_dim=1).mean(dim=1)
        return ChannelReliability(spatial, semantic, language, dynamic)
