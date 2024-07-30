from sae_lens import SAE, HookedSAETransformer
from typing import cast
from feature_lens.utils.device import get_device
from feature_lens.nn.transcoder import (
    Transcoder,
    load_pretrained as _load_mlp_transcoder,
)


def load_model(name: str = "gpt2-small") -> HookedSAETransformer:
    model = HookedSAETransformer.from_pretrained(
        name,
        device=get_device(),  # type: ignore
    )
    model = cast(HookedSAETransformer, model)
    model.set_use_split_qkv_input(True)
    model.set_use_hook_mlp_in(True)
    # NOTE: can't use this in 2.2.0 due to a bug in TransformerLens
    # See: https://github.com/TransformerLensOrg/TransformerLens/issues/667
    # model.set_use_attn_result(True)
    # TODO: remove in 2.2.1
    return model


def load_sae(
    release: str = "gpt2-small-res-jb", sae_id: str = "blocks.8.hook_resid_pre"
) -> SAE:
    sae, _, _ = SAE.from_pretrained(
        release=release,  # see other options in sae_lens/pretrained_saes.yaml
        sae_id=sae_id,  # won't always be a hook point
        device=get_device(),  # type: ignore
    )
    sae.use_error_term = True
    return sae


def load_transcoder(
    release: str = "gpt2-small-mlp-tc", sae_id: str = "blocks.8.mlp.hook_mlp_in"
) -> Transcoder:
    if not release == "gpt2-small-mlp-tc":
        raise ValueError("Only gpt2-small-mlp-tc is supported")

    layer = int(sae_id.split(".")[1])
    filenames = [
        f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.ln2.hook_normalized_24576.pt"
    ]
    transcoders = _load_mlp_transcoder(filenames)
    assert len(transcoders) == 1
    return list(transcoders.values())[0]
