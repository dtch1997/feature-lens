from feature_lens.nn.transcoder import sae_training  # noqa
from .sae_training.sparse_autoencoder import SparseAutoencoder


class Transcoder(SparseAutoencoder):
    """Dummy wrapper around base SAE class"""

    pass
