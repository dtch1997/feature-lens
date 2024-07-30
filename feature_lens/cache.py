# Monkey patch get_caching_hooks to enable delta

from functools import partial
from typing import (
    Callable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch

from transformer_lens.utils import Slice, SliceInput
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint, HookedRootModule

from feature_lens.core.types import Model
from feature_lens.utils.device import get_device
from feature_lens.utils.data_handler import DataHandler, InputType

# Define type aliases
NamesFilter = Optional[Union[Callable[[str], bool], Sequence[str]]]
DeviceType = Optional[torch.device]


# NOTE: Redefinition of base HookedRootModule's get_caching_hooks
# This is to support backpropagation of intermediate activations
def get_caching_hooks(
    self,
    names_filter: NamesFilter = None,
    incl_bwd: bool = False,
    device: DeviceType = None,
    remove_batch_dim: bool = False,
    cache: Optional[dict] = None,
    pos_slice: Union[Slice, SliceInput] = None,
) -> Tuple[dict, list, list]:
    """Creates hooks to cache activations. Note: It does not add the hooks to the model.

    Args:
        names_filter (NamesFilter, optional): Which activations to cache. Can be a list of strings (hook names) or a filter function mapping hook names to booleans. Defaults to lambda name: True.
        incl_bwd (bool, optional): Whether to also do backwards hooks. Defaults to False.
        device (_type_, optional): The device to store on. Keeps on the same device as the layer if None.
        remove_batch_dim (bool, optional): Whether to remove the batch dimension (only works for batch_size==1). Defaults to False.
        cache (Optional[dict], optional): The cache to store activations in, a new dict is created by default. Defaults to None.

    Returns:
        cache (dict): The cache where activations will be stored.
        fwd_hooks (list): The forward hooks.
        bwd_hooks (list): The backward hooks. Empty if incl_bwd is False.
    """
    if cache is None:
        cache = {}

    pos_slice = Slice.unwrap(pos_slice)

    if names_filter is None:
        names_filter = lambda name: True
    elif isinstance(names_filter, str):
        filter_str = names_filter
        names_filter = lambda name: name == filter_str
    elif isinstance(names_filter, list):
        filter_list = names_filter
        names_filter = lambda name: name in filter_list
    elif callable(names_filter):
        names_filter = names_filter
    else:
        raise ValueError("names_filter must be a string, list of strings, or function")
    assert callable(names_filter)  # Callable[[str], bool]

    self.is_caching = True

    def save_hook(tensor: torch.Tensor, hook: HookPoint, is_backward: bool = False):
        # for attention heads the pos dimension is the third from last
        if hook.name is None:
            raise RuntimeError("Hook should have been provided a name")

        hook_name = hook.name
        if is_backward:
            hook_name += "_grad"
        # NOTE: our change here!
        resid_stream = tensor
        # NOTE: end change
        if remove_batch_dim:
            resid_stream = resid_stream[0]

        if (
            hook.name.endswith("hook_q")
            or hook.name.endswith("hook_k")
            or hook.name.endswith("hook_v")
            or hook.name.endswith("hook_z")
            or hook.name.endswith("hook_result")
        ):
            pos_dim = -3
        else:
            # for all other components the pos dimension is the second from last
            # including the attn scores where the dest token is the second from last
            pos_dim = -2

        if (
            tensor.dim() >= -pos_dim
        ):  # check if the residual stream has a pos dimension before trying to slice
            resid_stream = pos_slice.apply(resid_stream, dim=pos_dim)
        cache[hook_name] = resid_stream

    fwd_hooks = []
    bwd_hooks = []
    for name, _ in self.hook_dict.items():
        if names_filter(name):
            fwd_hooks.append((name, partial(save_hook, is_backward=False)))
            if incl_bwd:
                bwd_hooks.append((name, partial(save_hook, is_backward=True)))

    return cache, fwd_hooks, bwd_hooks


def get_sae_cache(
    model: Model, handler: DataHandler, input: InputType = "clean"
) -> ActivationCache:
    cache_dict, fwd, bwd = model.get_caching_hooks(
        names_filter=lambda name: "sae" in name,
        incl_bwd=True,
        device=get_device(),  # type: ignore
    )

    with model.hooks(
        fwd_hooks=fwd,
        bwd_hooks=bwd,
    ):
        logits = handler.get_logits(model, input=input)
        metric = handler.get_metric(logits).mean()
        metric.backward()

    cache = ActivationCache(cache_dict, model)
    return cache


# Monkey patch the get_caching_hooks method
HookedRootModule.get_caching_hooks = get_caching_hooks
