import pytest

from feature_lens.core.types import Model
from feature_lens.utils.load_pretrained import load_model
from feature_lens.utils.device import set_device, get_device
from sae_lens import SAE, SAEConfig

set_device("cpu")


@pytest.fixture
def solu1l_model() -> Model:
    return load_model("solu-1l")


@pytest.fixture
def solu1l_sae(solu1l_model: Model) -> SAE:
    cfg = SAEConfig(
        architecture="standard",
        d_in=solu1l_model.cfg.d_model,
        d_sae=2 * solu1l_model.cfg.d_model,
        activation_fn_str="relu",
        apply_b_dec_to_input=False,
        finetuning_scaling_factor=False,
        context_size=1024,
        model_name="solu-1l",
        hook_name="blocks.0.hook_resid_pre",
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
