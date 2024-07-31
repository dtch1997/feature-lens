import pytest

from feature_lens.core.types import Model
from feature_lens.utils.load_pretrained import load_model
from feature_lens.utils.device import set_device, get_device
from sae_lens import SAE, SAEConfig
from feature_lens.nn.transcoder import Transcoder, TranscoderConfig

set_device("cpu")


@pytest.fixture
def model() -> Model:
    return load_model("solu-1l")

@pytest.fixture
def expansion_factor() -> int:
    return 2

@pytest.fixture
def sae(model: Model, expansion_factor: int) -> SAE:
    cfg = SAEConfig(
        architecture="standard",
        d_in=model.cfg.d_model,
        d_sae=expansion_factor * model.cfg.d_model,
        activation_fn_str="relu",
        apply_b_dec_to_input=False,
        finetuning_scaling_factor=False,
        context_size=1024,
        model_name="solu-1l",
        hook_name="blocks.0.attn.hook_z",
        hook_layer=0,
        hook_head_index=None,
        prepend_bos=False,
        dataset_path="n/a",
        dataset_trust_remote_code=False,
        normalize_activations="n/a",
        device=get_device(),
        dtype="float32",
        sae_lens_training_version="n/a",
    )
    return SAE(cfg)

@pytest.fixture
def transcoder(model: Model, expansion_factor: int) -> Transcoder:
    # TODO: replace with inferece-only transcoder that doesn't have irrelevant config options
    cfg = TranscoderConfig(
        is_transcoder=True,
        model_name = model.cfg.model_name,
        hook_point = "blocks.0.ln2",
        hook_point_layer = 0,
        hook_point_head_index=None, 
        out_hook_point="blocks.0.hook_mlp_out",
        out_hook_point_layer=0,
        d_in = model.cfg.d_model,
        d_out = model.cfg.d_model,
        expansion_factor = expansion_factor,
        device = get_device(),
        # irrelevant features
        feature_sampling_method=None,
    )
    return Transcoder(cfg)