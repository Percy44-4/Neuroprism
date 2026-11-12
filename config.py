from dataclasses import dataclass, field


@dataclass(frozen=True)
class BrainEncoderConfig:
    subject_voxels: tuple[int, ...] = (13447, 14828, 9114)
    width: int = 4096
    embedding_dim: int = 768
    blocks: int = 4
    dropout: float = 0.15
    tokens: int = 16
    token_width: int = 256
    projection_width: int = 2048


@dataclass(frozen=True)
class HierarchyConfig:
    step_size: float = 0.05
    agreement_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    shrinkage_weights: tuple[float, float, float] = (0.001, 0.001, 0.001)
    steps: int = 5
    texture_shape: tuple[int, int, int, int] = (6, 32, 32, 256)
    object_shape: tuple[int, int, int] = (6, 8, 512)
    semantic_shape: tuple[int] = (768,)
    embedding_dim: int = 768
    seed_resolution: int = 8


@dataclass(frozen=True)
class SceneNetworkConfig:
    texture_width: int = 128
    texture_residual_layers: int = 2
    texture_memory_tokens: int = 16
    texture_attention_heads: int = 4
    coordinate_frequencies: int = 4
    visual_tokens: int = 32
    text_tokens: int = 8
    slot_iterations: int = 3
    slot_mlp_expansion: int = 2
    slot_attention_heads: int = 8
    semantic_tokens: int = 8
    semantic_layers: int = 3
    semantic_attention_heads: int = 8
    feed_forward_expansion: int = 4
    temporal_kernel: int = 3
    norm_epsilon: float = 1e-6
    attention_epsilon: float = 1e-8
    minimum_log_variance: float = -8.0
    maximum_log_variance: float = 4.0
    raster_minimum_scale: float = 0.05
    raster_maximum_scale: float = 0.75


@dataclass(frozen=True)
class ReadoutConfig:
    language_mc_samples: int = 8
    layers: int = 4
    attention_heads: int = 8
    hidden_dim: int = 768
    channels: tuple[str, ...] = ("spatial", "semantic", "language", "dynamic")
    query_tokens: int = 32
    max_language_tokens: int = 32
    dropout: float = 0.1


@dataclass(frozen=True)
class LanguageConfig:
    vocabulary_size: int = 50257
    width: int = 768
    layers: int = 12
    heads: int = 12
    max_length: int = 1024
    mlp_ratio: int = 4
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    eos_token_id: int = 50256
    pad_token_id: int = 50256
    prefix_length: int = 8


@dataclass(frozen=True)
class EvidenceConfig:
    frames: int = 6
    height: int = 256
    width: int = 256
    concept_count: int = 51


@dataclass(frozen=True)
class NeuroPrismConfig:
    brain: BrainEncoderConfig = field(default_factory=BrainEncoderConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    scene: SceneNetworkConfig = field(default_factory=SceneNetworkConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
