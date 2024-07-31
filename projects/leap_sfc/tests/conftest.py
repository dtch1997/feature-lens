import pytest

from feature_lens.data.handler import Dataset, build_handler, DataHandler
from feature_lens.data.toy_datasets import make_dataset
from feature_lens.data.metric import LogitDiff, MetricFunction
from feature_lens.core.types import Model
from feature_lens.utils.load_pretrained import load_model
from feature_lens.utils.device import set_device, get_device
from sae_lens import SAE, SAEConfig
from feature_lens.nn.transcoder import Transcoder, TranscoderConfig

from projects.leap_sfc.leap.model_handler import ModelHandler

set_device("cpu")


@pytest.fixture
def model() -> Model:
    return load_model("solu-1l")


@pytest.fixture
def metric_fn() -> MetricFunction:
    return LogitDiff()


@pytest.fixture
def dataset() -> Dataset:
    return make_dataset("ioi")


@pytest.fixture
def data_handler(model: Model, dataset: Dataset) -> DataHandler:
    return build_handler(model, dataset)


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
    sae = SAE(cfg)
    sae.turn_off_forward_pass_hook_z_reshaping()
    return sae


@pytest.fixture
def transcoder(model: Model, expansion_factor: int) -> Transcoder:
    # TODO: replace with inferece-only transcoder that doesn't have irrelevant config options
    cfg = TranscoderConfig(
        is_transcoder=True,
        model_name=model.cfg.model_name,
        hook_point="blocks.0.ln2",
        hook_point_layer=0,
        hook_point_head_index=None,
        out_hook_point="blocks.0.hook_mlp_out",
        out_hook_point_layer=0,
        d_in=model.cfg.d_model,
        d_out=model.cfg.d_model,
        expansion_factor=expansion_factor,
        device=get_device(),
        # irrelevant features
        feature_sampling_method=None,
    )
    return Transcoder(cfg)


@pytest.fixture()
def model_handler(model: Model, sae: SAE, transcoder: Transcoder):
    return ModelHandler(
        model,
        {sae.cfg.hook_name: sae},
        {transcoder.cfg.hook_name: transcoder},
    )
