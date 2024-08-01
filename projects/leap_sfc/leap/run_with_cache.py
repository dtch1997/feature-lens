import torch

from einops import einsum, rearrange
from feature_lens.data.handler import DataHandler, InputType
from feature_lens.data.metric import MetricFunction
from feature_lens.cache import get_cache

from .types import SAE, Transcoder, Float, ActivationCache
from .model_handler import ModelHandler
from .utils import iter_all_heads


def run_with_cache(
    model_handler: ModelHandler,
    data_handler: DataHandler,
    metric_fn: MetricFunction,
    input: InputType = "clean",
) -> ActivationCache:
    """Runs the model on some data, populating a cache

    In order to get the cache for SAEs, we do not actually splice those SAEs in;
    rather, we just compute the SAE activations and gradients analytically
    """
    # First, run the model with cache as per normal
    # to get the activations and gradients at the model's hook points
    cache = get_cache(
        model_handler.model, data_handler, metric_fn, input=input, incl_bwd=True
    )

    # To get node activations for a head:
    # - Read the activations at hook_name_in
    # - Simulate forward pass using matmul and ReLU

    for head in iter_all_heads(model_handler.n_layers):
        input_act = cache[head.hook_name_in]
        sae = model_handler.get_sae_for_head(head)
        act_post = get_sae_act_post(sae, input_act)
        sae_act_post_hook_name = head.hook_name_in + ".hook_sae_acts_post"
        cache.cache_dict[sae_act_post_hook_name] = act_post

    # To get node gradients (w.r.t metric) for a head:
    # - Read the gradients at hook_name_out
    # - Multiply by the decoder weights; this gives gradient by chain rule
    for head in iter_all_heads(model_handler.n_layers):
        grad_metric_wrt_output = cache[head.hook_name_out + "_grad"]
        sae = model_handler.get_sae_for_head(head)
        grad_metric_wrt_act = get_sae_act_post_grad_metric(sae, grad_metric_wrt_output)
        sae_grad_metric_wrt_act_hook_name = (
            head.hook_name_out + ".hook_sae_acts_post_grad"
        )
        cache.cache_dict[sae_grad_metric_wrt_act_hook_name] = grad_metric_wrt_act

    return cache


def get_sae_act_post(
    sae: SAE | Transcoder, input_act: Float[torch.Tensor, "batch seq d_model"]
) -> Float[torch.Tensor, "batch seq d_sae"]:
    """Get the post-ReLU activations of the SAE on a given input

    Apply the SAE's weights and biases to the input, then apply ReLU.
    """
    # NOTE: currently hardcoded to matmul and ReLU

    if len(input_act.shape) == 3:
        sae_pre_act = (
            einsum(
                input_act,
                sae.W_enc,
                "batch seq d_model, d_model d_sae -> batch seq d_sae",
            )
            + sae.b_enc
        )
        return torch.relu(sae_pre_act)

    elif len(input_act.shape) == 4:
        # Concatenate the heads together
        input_act = rearrange(
            input_act, "batch seq n_head d_head -> batch seq (n_head d_head)"
        )
        sae_pre_act = (
            einsum(
                input_act,
                sae.W_enc,
                "batch seq d_model, d_model d_sae -> batch seq d_sae",
            )
            + sae.b_enc
        )
        return torch.relu(sae_pre_act)

    else:
        raise ValueError(
            f"Expected 3D or 4D input; got input of shape {input_act.shape}"
        )


def get_sae_act_post_grad_metric(
    sae: SAE | Transcoder,
    grad_metric_wrt_output: Float[torch.Tensor, "batch seq d_out"],
):
    """Get the gradient of the metric w.r.t the post-ReLU activations of the SAE"""
    return einsum(
        grad_metric_wrt_output,
        sae.W_dec,
        "batch seq d_out, d_sae d_out -> batch seq d_sae",
    )
