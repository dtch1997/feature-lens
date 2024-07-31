import pytest
import torch

from projects.leap_sfc.leap.run_with_cache import (
    run_with_cache,
    get_sae_act_post,
    get_sae_act_post_grad_metric,
)


@pytest.mark.xfail(reason="Need to implement attention-SAE forward pass correctly")
def test_run_with_cache(model_handler, data_handler, metric_fn):
    run_with_cache(model_handler, data_handler, metric_fn)


def test_get_sae_act_post(sae):
    input_act = torch.randn(1, 4, 512)
    act_post = get_sae_act_post(sae, input_act)
    assert act_post.shape == (1, 4, 1024)

    # Compare to the ground-truth using run_with_cache
    _, cache = sae.run_with_cache(input_act)
    assert torch.allclose(act_post, cache["hook_sae_acts_post"])


def test_get_transcoder_act_post(transcoder):
    input_act = torch.randn(1, 4, 512)
    act_post = get_sae_act_post(transcoder, input_act)
    assert act_post.shape == (1, 4, 1024)

    # Compare to the ground-truth using run_with_cache
    _, cache = transcoder.run_with_cache(input_act)
    assert torch.allclose(act_post, cache["hook_hidden_post"])


def test_get_act_post_grad_metric_for_sae(sae):
    grad_metric_wrt_output = torch.randn(1, 4, 512)
    grad_metric_wrt_act = get_sae_act_post_grad_metric(sae, grad_metric_wrt_output)
    assert grad_metric_wrt_act.shape == (1, 4, 1024)

    # TODO: test against ground-truth using run_with_cache


def test_get_act_post_grad_metric_for_transcoder(transcoder):
    grad_metric_wrt_output = torch.randn(1, 4, 512)
    grad_metric_wrt_act = get_sae_act_post_grad_metric(
        transcoder, grad_metric_wrt_output
    )
    assert grad_metric_wrt_act.shape == (1, 4, 1024)

    # TODO: test against ground-truth using run_with_cache
