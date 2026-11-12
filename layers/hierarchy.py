import torch
from torch import Tensor, nn

from ..config import HierarchyConfig, SceneNetworkConfig
from ..structures import BrainEmbeddings, GaussianScene, SceneState
from .scene_posterior import SceneGaussianEncoder
from .scene_raster import ObjectTexturePredictor
from .scene_semantics import GlobalSemanticEncoder
from .scene_slots import ObjectSlotEncoder, SemanticObjectPredictor
from .scene_texture import TextureGridDecoder


class GaussianSceneInitializer(nn.Module):
    def __init__(
        self,
        config: HierarchyConfig = HierarchyConfig(),
        scene: SceneNetworkConfig = SceneNetworkConfig(),
    ) -> None:
        super().__init__()
        self.encoder = SceneGaussianEncoder(config, scene)

    def forward(self, brain: BrainEmbeddings) -> GaussianScene:
        return self.encoder(brain)


class BottomUpEvidence(nn.Module):
    def __init__(
        self,
        config: HierarchyConfig = HierarchyConfig(),
        scene: SceneNetworkConfig = SceneNetworkConfig(),
    ) -> None:
        super().__init__()
        self.texture_network = TextureGridDecoder(config.embedding_dim, config, scene)
        self.object_network = ObjectSlotEncoder(config, scene)
        self.semantic_network = GlobalSemanticEncoder(
            config.embedding_dim, config.semantic_shape[0], scene
        )

    def texture(self, visual: Tensor) -> Tensor:
        return self.texture_network(visual)

    def objects(self, visual: Tensor, text: Tensor) -> Tensor:
        return self.object_network(visual, text)

    def semantics(self, text: Tensor) -> Tensor:
        return self.semantic_network(text)

    def forward(self, brain: BrainEmbeddings) -> SceneState:
        return SceneState(
            self.texture(brain.visual),
            self.objects(brain.visual, brain.text),
            self.semantics(brain.text),
        )


class TopDownPredictor(nn.Module):
    def __init__(
        self,
        config: HierarchyConfig = HierarchyConfig(),
        scene: SceneNetworkConfig = SceneNetworkConfig(),
    ) -> None:
        super().__init__()
        self.object_network = SemanticObjectPredictor(config, scene)
        self.texture_network = ObjectTexturePredictor(config, scene)

    def objects_from_semantics(self, semantics: Tensor) -> Tensor:
        return self.object_network(semantics)

    def texture_from_objects(self, objects: Tensor) -> Tensor:
        return self.texture_network(objects)


def squared_norm_per_sample(value: Tensor) -> Tensor:
    return value.flatten(start_dim=1).square().sum(dim=1)


class PredictiveCodingInference(nn.Module):
    def __init__(
        self,
        initializer: GaussianSceneInitializer,
        bottom_up: BottomUpEvidence,
        top_down: TopDownPredictor,
        config: HierarchyConfig,
    ) -> None:
        super().__init__()
        self.initializer = initializer
        self.bottom_up = bottom_up
        self.top_down = top_down
        self.config = config

    def energy(self, state: SceneState, observation: SceneState) -> Tensor:
        predictions = (
            self.top_down.texture_from_objects(state.objects),
            self.top_down.objects_from_semantics(state.semantics),
        )
        terms = []
        for level, (latent, observed) in enumerate(
            zip(state.levels(), observation.levels())
        ):
            agreement = squared_norm_per_sample(latent - observed)
            if level < 2:
                agreement = agreement + squared_norm_per_sample(latent - predictions[level])
            terms.append(
                self.config.agreement_weights[level] * agreement
                + self.config.shrinkage_weights[level] * squared_norm_per_sample(latent)
            )
        return torch.stack(terms).sum(dim=0)

    def forward(self, brain: BrainEmbeddings) -> SceneState:
        with torch.enable_grad():
            posterior = self.initializer(brain)
            observation = self.bottom_up(brain)
            state = SceneState(*(
                (mean + (0.5 * log_variance).exp() * torch.randn_like(mean)).requires_grad_(True)
                for mean, log_variance in zip(
                    posterior.mean.levels(), posterior.log_variance.levels()
                )
            ))
            for _ in range(self.config.steps):
                energy = self.energy(state, observation)
                gradients = torch.autograd.grad(
                    energy.sum(), state.levels(), create_graph=self.training
                )
                updated = tuple(
                    latent - self.config.step_size * gradient
                    for latent, gradient in zip(state.levels(), gradients)
                )
                if not self.training:
                    updated = tuple(value.detach().requires_grad_(True) for value in updated)
                state = SceneState(*updated)
        return state
