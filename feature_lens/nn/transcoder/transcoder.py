from __future__ import annotations

from feature_lens.nn.transcoder import sae_training  # noqa
from .sae_training.sparse_autoencoder import SparseAutoencoder
from .sae_training.config import LanguageModelSAERunnerConfig


class Transcoder(SparseAutoencoder):
    """Dummy wrapper around base SAE class"""

    cfg: TranscoderConfig


class TranscoderConfig(LanguageModelSAERunnerConfig):
    """Dummy wrapper around base config class"""

    # Some syntactic sugar to align naming conventions with SAELens

    @property
    def hook_name(self):
        return self.hook_point

    @property
    def hook_layer(self):
        return self.hook_point_layer

    @property
    def hook_head_index(self):
        return self.hook_point_head_index
