from torch import Tensor, nn
from torch.nn import functional as F

from ..structures import ChannelReliability, EvidencePrediction, GatedConditions


def broadcast_sample_weight(weight: Tensor, target: Tensor) -> Tensor:
    return weight.reshape(weight.shape[0], *((1,) * (target.ndim - 1)))


class PrecisionGatedCompiler(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        evidence: EvidencePrediction,
        reliability: ChannelReliability,
        language_context: Tensor,
        semantic_context: Tensor,
    ) -> GatedConditions:
        if language_context.ndim != 3 or language_context.shape != semantic_context.shape:
            raise ValueError("condition_context_shape")
        if language_context.shape[0] != evidence.semantic_logits.shape[0]:
            raise ValueError("condition_batch_size")
        spatial_gate = reliability.spatial / (1 + reliability.spatial)
        dynamic_gate = reliability.dynamic / (1 + reliability.dynamic)
        text_gate = reliability.language / (reliability.language + reliability.semantic)
        weight = broadcast_sample_weight(text_gate, language_context)
        context = F.normalize(weight * language_context + (1 - weight) * semantic_context, dim=-1)
        control = 0.5 * (
            1 + broadcast_sample_weight(spatial_gate, evidence.spatial.mean) * evidence.spatial.mean
        )
        dynamic = broadcast_sample_weight(dynamic_gate, evidence.dynamic.mean) * evidence.dynamic.mean
        return GatedConditions(context, control, dynamic)
