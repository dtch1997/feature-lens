from projects.leap_sfc.leap.types import ActivationCache, Head
from feature_lens.data.handler import InputType
from jaxtyping import Float

import torch

SAE_NODE_ACT_NAME = "hook_sae_acts_post"  # post-ReLU activations of SAE
SAE_NODE_GRAD_OUT_NAME = (
    "hook_sae_acts_post_grad"  # gradient of metric w.r.t node activation
)
SAE_NODE_GRAD_IN_NAME = (
    "hook_sae_acts_post_grad_input"  # gradient of node activation w.r.t input
)


class CacheHandler:
    """Handles a clean and corrupt cache

    Assumes that the relevant SAE activations have already been added to the respective caches
    """

    clean_cache: ActivationCache
    corrupt_cache: ActivationCache

    def __init__(self, clean_cache: ActivationCache, corrupt_cache: ActivationCache):
        self.clean_cache = clean_cache
        self.corrupt_cache = corrupt_cache

        # TODO: validate that clean and corrupt cache have same shapes

    @property
    def seq_len(self) -> int:
        """Returns the number of tokens in the cached activations"""
        return self.clean_cache["hook_embed"].shape[1]

    def get_cache(self, input: InputType = "clean") -> ActivationCache:
        """Returns the cache"""
        if input == "clean":
            return self.clean_cache
        elif input == "corrupt":
            return self.corrupt_cache

    def get_layernorm_scale(
        self, head: Head, input: InputType = "clean"
    ) -> Float[torch.Tensor, "batch seq"]:
        """Returns the cached layernorm scale at the layernorm after a head"""
        if head.head_type == "mlp":
            return self.get_cache(input)[f"blocks.{head.layer}.ln2.hook_scale"]
        elif head.head_type == "att":
            return self.get_cache(input)[f"blocks.{head.layer}.ln1.hook_scale"]
        else:
            raise ValueError(f"Unknown head type: {head.head_type}")

    def get_act(
        self, head: Head, input: InputType = "clean"
    ) -> Float[torch.Tensor, "batch seq d_sae"]:
        """Returns the activation of all nodes at a head"""
        if head.head_type in ("mlp", "att"):
            return self.get_cache(input)[head.hook_name_in + f".{SAE_NODE_ACT_NAME}"]
        else:
            raise ValueError(f"Unknown head type: {head.head_type}")

    def get_grad_metric_wrt_act(
        self, head: Head, input: InputType = "clean"
    ) -> Float[torch.Tensor, "batch seq d_sae"]:
        """Returns the gradient of the metric with respect to all nodes at a head"""
        if head.head_type in ("mlp", "att"):
            return self.get_cache(input)[
                head.hook_name_out + f".{SAE_NODE_GRAD_OUT_NAME}"
            ]
        else:
            raise ValueError(f"Unknown head type: {head.head_type}")
