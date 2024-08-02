import torch

from einops import einsum, rearrange
from feature_lens.data.handler import DataHandler, InputType
from feature_lens.data.metric import MetricFunction
from feature_lens.cache import get_cache

from .types import SAE, Transcoder, Float, ActivationCache
from .model_handler import ModelHandler
from .utils import iter_all_heads

from .types import Head
from .cache_handler import SAE_NODE_ACT_NAME, SAE_NODE_GRAD_OUT_NAME


def run_with_cache(
    model_handler: ModelHandler,
    data_handler: DataHandler,
    metric_fn: MetricFunction,
    input: InputType = "clean",
) -> ActivationCache:
    """Runs the model on some data, populating a cache

    In order to get the cache for SAEs, we do not actually splice those SAEs in;
    rather, we just compute the SAE activations and gradients analytically

    At the end of this, the cache will contain:
    - Activations of SAEs,
    """
    # First, run the model with cache as per normal
    # to get the activations and gradients at the model's hook points
    cache = get_cache(
        model_handler.model, data_handler, metric_fn, input=input, incl_bwd=True
    )

    # Activations of the SAEs
    for head in iter_all_heads(model_handler.n_layers):
        input_act = cache[head.hook_name_in]
        sae = model_handler.get_sae_for_head(head)
        act_post = get_sae_act_post(sae, input_act)
        sae_act_post_hook_name = head.hook_name_in + f".{SAE_NODE_ACT_NAME}"
        cache.cache_dict[sae_act_post_hook_name] = act_post

    # Gradients of the metric w.r.t the SAE activations
    for head in iter_all_heads(model_handler.n_layers):
        grad_metric_wrt_output = cache[head.hook_name_out + "_grad"]
        sae = model_handler.get_sae_for_head(head)
        grad_metric_wrt_act = get_sae_act_post_grad_metric(sae, grad_metric_wrt_output)
        sae_grad_metric_wrt_act_hook_name = (
            head.hook_name_out + f".{SAE_NODE_GRAD_OUT_NAME}"
        )
        cache.cache_dict[sae_grad_metric_wrt_act_hook_name] = grad_metric_wrt_act

    # NOTE: Gradients of the SAE activations w.r.t the input to the head
    # have to be calculated online due to memory concerns
    # (i.e. we can't cache them)

    return cache


def _get_sae_cls_for_head_type(head: Head) -> type[SAE | Transcoder]:
    if head == "att":
        return Transcoder
    elif head == "mlp":
        return SAE
    else:
        raise ValueError(f"Unknown head type: {head}")


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
    grad_metric_wrt_output: torch.Tensor,  # Different shape depending on head type
):
    if isinstance(sae, SAE):
        return _get_sae_act_post_grad_metric(sae, grad_metric_wrt_output)
    elif isinstance(sae, Transcoder):
        return _get_transcoder_act_post_grad_metric(sae, grad_metric_wrt_output)


def _get_sae_act_post_grad_metric(
    sae: SAE,
    grad_metric_wrt_output: Float[torch.Tensor, "batch seq n_head d_head"],
) -> Float[torch.Tensor, "batch seq d_sae"]:
    """Get the gradient of the metric w.r.t the post-ReLU activations of an attention-out SAE

    Equivalent to splicing the SAE into the model and backpropagating the gradient
    The (i, j, k)th entry is the derivative of the metric w.r.t the kth SAE feature activation at the jth token position in the ith example
    """
    reshaped_grad = rearrange(
        grad_metric_wrt_output, "batch seq n_head d_head -> batch seq (n_head d_head)"
    )
    return einsum(
        reshaped_grad,
        sae.W_dec,
        "batch seq d_out, d_sae d_out -> batch seq d_sae",
    )


def _get_transcoder_act_post_grad_metric(
    sae: Transcoder,
    grad_metric_wrt_output: Float[torch.Tensor, "batch seq d_out"],
) -> Float[torch.Tensor, "batch seq d_sae"]:
    """Get the gradient of the metric w.r.t the post-ReLU activations of an MLP transcoder

    Equivalent to splicing the transcoder into the model and backpropagating the gradient
    The (i, j, k)th entry is the derivative of the metric w.r.t the kth SAE feature activation at the jth token position in the ith example"""
    return einsum(
        grad_metric_wrt_output,
        sae.W_dec,
        "batch seq d_out, d_sae d_out -> batch seq d_sae",
    )
