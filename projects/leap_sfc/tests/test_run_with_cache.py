import torch

from projects.leap_sfc.leap.types import SAE, Transcoder
from projects.leap_sfc.leap.run_with_cache import (
    run_with_cache,
    get_sae_act_post,
    get_sae_act_post_grad_metric,
)


def test_run_with_cache(model_handler, data_handler, metric_fn):
    run_with_cache(model_handler, data_handler, metric_fn)


def test_get_sae_act_post_for_sae(sae: SAE):
    input_act = torch.randn(1, 4, 8, 64)
    act_post = get_sae_act_post(sae, input_act)
    assert act_post.shape == (1, 4, 1024)

    # Compare to the ground-truth using run_with_cache
    # Re-enable reshaping for the forward pass only
    sae.turn_on_forward_pass_hook_z_reshaping()
    _, cache = sae.run_with_cache(input_act)
    sae.turn_off_forward_pass_hook_z_reshaping()
    assert torch.allclose(act_post, cache["hook_sae_acts_post"])


def test_get_sae_act_post_for_transcoder(transcoder: Transcoder):
    input_act = torch.randn(1, 4, 512)
    act_post = get_sae_act_post(transcoder, input_act)
    assert act_post.shape == (1, 4, 1024)

    # Compare to the ground-truth using run_with_cache
    _, cache = transcoder.run_with_cache(input_act)
    assert torch.allclose(act_post, cache["hook_hidden_post"])


def test_get_act_post_grad_metric_for_sae(sae: SAE):
    # Calculate the gradient using backpropagation
    cache_dict, fwd, bwd = sae.get_caching_hooks(incl_bwd=True)
    with sae.hooks(
        fwd_hooks=fwd,
        bwd_hooks=bwd,
    ):
        input = torch.randn(1, 4, 512)
        output = sae(input)
        metric = output.sum()
        metric.backward()

    expected_grad_metric_wrt_act = cache_dict["hook_sae_acts_post_grad"]

    # Calculate the analytic gradient
    grad_metric_wrt_output = torch.ones_like(output)
    actual_grad_metric_wrt_act = get_sae_act_post_grad_metric(
        sae, grad_metric_wrt_output
    )
    assert actual_grad_metric_wrt_act.shape == (1, 4, 1024)
    assert torch.allclose(expected_grad_metric_wrt_act, actual_grad_metric_wrt_act)


def test_get_act_post_grad_metric_for_transcoder(transcoder: Transcoder):
    # Calculate the gradient using backpropagation
    cache_dict, fwd, bwd = transcoder.get_caching_hooks(incl_bwd=True)
    with transcoder.hooks(
        fwd_hooks=fwd,
        bwd_hooks=bwd,
    ):
        input = torch.randn(1, 4, 512)
        output = transcoder(input)[0]
        metric = output.sum()
        metric.backward()

    expected_grad_metric_wrt_act = cache_dict["hook_hidden_post_grad"]

    # Calculate the analytic gradient
    grad_metric_wrt_output = torch.ones_like(output)
    actual_grad_metric_wrt_act = get_sae_act_post_grad_metric(
        transcoder, grad_metric_wrt_output
    )
    assert actual_grad_metric_wrt_act.shape == (1, 4, 1024)
    assert torch.allclose(expected_grad_metric_wrt_act, actual_grad_metric_wrt_act)
