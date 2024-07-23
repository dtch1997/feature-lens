import torch

from transformer_lens import ActivationCache
from jaxtyping import Float
from feature_lens.core.types import Model, HookName
from feature_lens.utils.data_handler import DataHandler
from feature_lens.utils.device import get_device


def get_sae_cache_for_target_feature_as_metric(
    model: Model,
    handler: DataHandler,
    target_hook_name: HookName,
    target_feature: int,
    input="clean",
):
    """Get the activations when running the model on the input, and gradients w.r.t a target feature."""
    cache_dict, fwd, bwd = model.get_caching_hooks(
        names_filter=lambda name: "sae" in name,
        incl_bwd=True,
        device=get_device(),  # type: ignore
    )

    with model.hooks(
        fwd_hooks=fwd,
        bwd_hooks=bwd,
    ):
        _ = handler.get_logits(model, input=input)  # type: ignore
        # Mean across examples
        # Sum across token positions
        metric = (
            cache_dict[target_hook_name][:, :, target_feature].sum(dim=1).mean(dim=0)
        )
        metric.backward()

    cache = ActivationCache(cache_dict, model)
    return cache


def compute_attribution(
    hook_name: HookName,
    *,
    clean_cache: ActivationCache,
    corrupt_cache: ActivationCache | None = None,
) -> Float[torch.Tensor, "batch seq ..."]:
    """Compute the attribution scores at a given hook point."""
    clean_acts = clean_cache[hook_name]
    clean_grads = clean_cache[hook_name + "_grad"]

    if corrupt_cache is None:
        corrupt_acts = torch.zeros_like(clean_acts)
    else:
        corrupt_acts = corrupt_cache[hook_name]

    attrib = (corrupt_acts - clean_acts) * clean_grads
    return attrib
