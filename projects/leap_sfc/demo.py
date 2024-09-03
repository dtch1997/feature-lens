from simple_parsing import ArgumentParser

from feature_lens.utils.load_pretrained import load_model, load_sae, load_transcoder
from feature_lens.utils.device import get_device
from feature_lens.data.handler import build_handler
from feature_lens.data.toy_datasets import make_dataset
from feature_lens.data.metric import LogitDiff
from sae_lens import SAE, SAEConfig
from feature_lens.nn.transcoder import Transcoder, TranscoderConfig

from .leap.leap import LeapAlgo
from .leap.model_handler import ModelHandler
from .leap.cache_handler import CacheHandler
from .leap.run_with_cache import run_with_cache

from dataclasses import dataclass


def _build_solu1l_model_handler() -> ModelHandler:
    """A simple model handler for testing"""
    model = load_model(name="solu-1l")

    # Attention-out SAE
    expansion_factor = 2
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
    saes = {"blocks.0.attn.hook_z": sae}

    # Transcoder
    cfg = TranscoderConfig(
        is_transcoder=True,
        model_name=model.cfg.model_name,
        hook_point="blocks.0.ln2.hook_normalized",
        hook_point_layer=0,
        hook_point_head_index=None,
        out_hook_point="blocks.0.hook_mlp_out",
        out_hook_point_layer=0,
        d_in=model.cfg.d_model,
        d_out=model.cfg.d_model,
        expansion_factor=expansion_factor,
        device=get_device(),
        # irrelevant
        feature_sampling_method=None,  # type: ignore
    )
    transcoder = Transcoder(cfg)
    transcoders = {"blocks.0.ln2.hook_normalized": transcoder}

    return ModelHandler(model, saes, transcoders)


def _build_gpt2_model_handler() -> ModelHandler:
    model = load_model()

    # Attention-out SAEs
    saes = {}
    release = "gpt2-small-hook-z-kk"
    for layer in range(12):
        hook_name = f"blocks.{layer}.attn.hook_z"
        id = f"blocks.{layer}.hook_z"
        sae = load_sae(release=release, sae_id=id)
        saes[hook_name] = sae

    # MLP transcoders
    transcoders = {}
    release = "gpt2-small-mlp-tc"
    for layer in range(12):
        hook_name = f"blocks.{layer}.ln2.hook_normalized"
        id = f"blocks.{layer}.mlp"
        transcoder = load_transcoder(release=release, sae_id=id)
        transcoders[hook_name] = transcoder

    return ModelHandler(model, saes, transcoders)


def build_model_handler(model_name: str = "solu-1l") -> ModelHandler:
    if model_name == "solu-1l":
        return _build_solu1l_model_handler()
    elif model_name == "gpt2":
        return _build_gpt2_model_handler()
    else:
        raise ValueError(f"Unknown model name: {model_name}")


@dataclass
class LeapDemoConfig:
    model_name: str = "solu-1l"
    dataset: str = "ioi"
    threshold: float = 0.1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(LeapDemoConfig, "config")
    args = parser.parse_args()
    config = args.config

    # Load in the models
    model_handler = build_model_handler(config.model_name)
    dataset = make_dataset(config.dataset)
    data_handler = build_handler(model_handler.model, dataset)
    metric_fn = LogitDiff()

    # Run the model with cache
    clean_cache = run_with_cache(model_handler, data_handler, metric_fn)
    corrupt_cache = run_with_cache(
        model_handler, data_handler, metric_fn, input="corrupt"
    )
    cache_handler = CacheHandler(clean_cache, corrupt_cache)

    # Set up LEAP algorithm
    leap = LeapAlgo(model_handler, cache_handler, config.threshold)
    leap.run_leap()
    graph = leap.graph
    del leap

    # TODO: visualize and save the graph
