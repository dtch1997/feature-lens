"""Script to evaluate sparsity of SAEs and transcoders in GPT-2 model."""

from feature_lens.utils.load_pretrained import load_model, load_sae, load_transcoder
from feature_lens.data.toy_datasets import make_dataset
from feature_lens.data.handler import build_handler
from feature_lens.data.metric import LogitDiff

from projects.leap_sfc.leap.model_handler import ModelHandler
from projects.leap_sfc.leap.run_with_cache import run_with_cache, SAE_NODE_ACT_NAME


def gpt2_model_handler() -> ModelHandler:
    model = load_model("gpt2-small")

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


def test_sparsity():
    model_handler = gpt2_model_handler()
    dataset = make_dataset("ioi")
    data_handler = build_handler(model_handler.model, dataset)
    metric_fn = LogitDiff()
    cache = run_with_cache(model_handler, data_handler, metric_fn)

    for layer in range(12):
        # Attention
        hook_name = f"blocks.{layer}.attn.hook_z.{SAE_NODE_ACT_NAME}"
        act = cache[hook_name]
        sparsity = (act != 0).float().mean()
        print(f"Layer {layer} Att-SAE sparsity: {sparsity}")

        # Transcoder
        hook_name = f"blocks.{layer}.ln2.hook_normalized.{SAE_NODE_ACT_NAME}"
        act = cache[hook_name]
        sparsity = (act != 0).float().mean()
        print(f"Layer {layer} MLP-TC sparsity: {sparsity}")


if __name__ == "__main__":
    test_sparsity()
