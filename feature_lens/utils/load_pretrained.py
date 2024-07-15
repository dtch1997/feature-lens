from sae_lens import SAE, HookedSAETransformer
from feature_lens.utils.device_manager import DeviceManager


def load_model(name: str = "gpt2-small") -> HookedSAETransformer:
    model = HookedSAETransformer.from_pretrained(
        name, device=DeviceManager.instance().get_device()
    )
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
        device=DeviceManager.instance().get_device(),
    )
    sae.use_error_term = True
    return sae
