import torch
import pytest

from projects.leap_sfc.leap.leap import (
    construct_sparse_grad_input_tensor_for_mlp,
    construct_sparse_grad_input_tensor_for_att,
    construct_sparse_grad_output_tensor_for_mlp,
    construct_sparse_grad_output_tensor_for_att,
)


@pytest.fixture
def mlp_input_data():
    batch, seq, d_sae, d_model = 2, 3, 4, 5
    W_enc = torch.randn(d_model, d_sae)
    input_act = torch.zeros(batch, seq, d_sae)

    # Set some activations
    input_act[0, 0, 1] = 1
    input_act[0, 2, 3] = 1
    input_act[1, 1, 2] = 1

    return W_enc, input_act


def test_construct_sparse_grad_input_tensor_for_mlp(mlp_input_data):
    W_enc, input_act = mlp_input_data
    sparse_tensor = construct_sparse_grad_input_tensor_for_mlp(W_enc, input_act)

    assert sparse_tensor.shape == (2, 3, 4, 3, 5)
    assert sparse_tensor._nnz() == 3
    # Check that input data is correctly mapped to the sparse tensor
    assert torch.allclose(sparse_tensor[0, 0, 1, 0], W_enc[:, 1])
    assert torch.allclose(sparse_tensor[0, 2, 3, 2], W_enc[:, 3])
    assert torch.allclose(sparse_tensor[1, 1, 2, 1], W_enc[:, 2])


@pytest.fixture
def mlp_output_data():
    batch, seq, d_sae, d_model = 2, 3, 4, 5
    W_dec = torch.randn(d_sae, d_model)
    input_act = torch.zeros(batch, seq, d_sae)

    # Set some activations
    input_act[0, 0, 1] = 1
    input_act[0, 2, 3] = 1
    input_act[1, 1, 2] = 1

    return W_dec, input_act


def test_construct_sparse_grad_output_tensor_for_mlp(mlp_output_data):
    W_dec, input_act = mlp_output_data
    sparse_tensor = construct_sparse_grad_output_tensor_for_mlp(W_dec, input_act)

    assert sparse_tensor.shape == (2, 3, 4, 5)
    assert sparse_tensor._nnz() == 3
    # Check that input data is correctly mapped to the sparse tensor
    assert torch.allclose(sparse_tensor[0, 0, 1], W_dec[1, :])
    assert torch.allclose(sparse_tensor[0, 2, 3], W_dec[3, :])
    assert torch.allclose(sparse_tensor[1, 1, 2], W_dec[2, :])


@pytest.fixture
def att_input_data():
    batch_size = 2
    n_head = 4
    d_head = 16
    d_sae = 32
    d_model = 64
    query_seq = 10
    key_seq = 15

    W_enc = torch.randn(n_head, d_head, d_sae)
    W_V = torch.randn(n_head, d_head, d_model)
    attn_pattern = torch.randn(batch_size, n_head, query_seq, key_seq)
    input_act = torch.randn(batch_size, query_seq, d_sae)

    return W_enc, W_V, attn_pattern, input_act


# @pytest.mark.xfail(reason="Not implemented yet")
def test_construct_sparse_grad_input_tensor_for_att(att_input_data):
    W_enc, W_V, attn_pattern, input_act = att_input_data

    # Call the function
    result = construct_sparse_grad_input_tensor_for_att(
        W_enc, W_V, attn_pattern, input_act
    )

    # Check the shape of the result
    expected_shape = (
        input_act.shape[0],  # batch
        input_act.shape[1],  # query_seq
        input_act.shape[2],  # d_sae
        attn_pattern.shape[3],  # key_seq
        W_V.shape[2],  # d_model
    )
    assert (
        result.size() == expected_shape
    ), f"Expected shape {expected_shape}, but got {result.size()}"

    # Check if the d_model dimension is dense
    assert (
        result.sparse_dim() >= 3
    ), "The sparse dimensions should be the first 4 dimensions"

    # Check if the values are non-zero (assuming the function doesn't produce all zeros)
    assert result.values().abs().sum() > 0, "The result should contain non-zero values"

    # TODO: Check that the values are correct


@pytest.fixture
def att_output_data():
    batch_size = 2
    n_head = 4
    d_head = 16
    d_sae = 32
    d_model = 64
    query_seq = 10
    key_seq = 15

    W_dec = torch.randn(d_sae, n_head, d_head)
    W_O = torch.randn(n_head, d_head, d_model)
    input_act = torch.randn(batch_size, key_seq, d_sae)

    return W_dec, W_O, input_act


def test_construct_sparse_grad_output_tensor_for_att(att_output_data):
    W_dec, W_O, input_act = att_output_data

    # Call the function
    result = construct_sparse_grad_output_tensor_for_att(W_dec, W_O, input_act)

    # Check the shape of the result
    expected_shape = (
        input_act.shape[0],  # batch
        input_act.shape[1],  # key_seq
        W_dec.shape[0],  # d_sae
        W_O.shape[2],  # d_model
    )
    assert (
        result.size() == expected_shape
    ), f"Expected shape {expected_shape}, but got {result.size()}"

    # Check if the d_model dimension is dense
    assert (
        result.sparse_dim() == 3
    ), "The sparse dimensions should be the first 3 dimensions"

    # Check if the values are non-zero (assuming the function doesn't produce all zeros)
    assert result.values().abs().sum() > 0, "The result should contain non-zero values"

    # TODO: Check that the values are correct
