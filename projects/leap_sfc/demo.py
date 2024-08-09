from simple_parsing import ArgumentParser

from feature_lens.utils.load_pretrained import load_model, load_sae, load_transcoder
from feature_lens.data.handler import build_handler
from feature_lens.data.toy_datasets import make_dataset
from feature_lens.data.metric import LogitDiff

from .leap.leap import LeapAlgo
from .leap.model_handler import ModelHandler
from .leap.cache_handler import CacheHandler
from .leap.run_with_cache import run_with_cache

from dataclasses import dataclass


def build_model_handler() -> ModelHandler:
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


@dataclass
class LeapDemoConfig:
    dataset: str = "ioi"
    threshold: float = 0.1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(LeapDemoConfig, "config")
    args = parser.parse_args()
    config = args.config

    # Load in the models
    model_handler = build_model_handler()
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
