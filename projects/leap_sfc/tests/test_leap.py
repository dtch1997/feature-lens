import torch
import pytest

from projects.leap_sfc.leap.leap import construct_sparse_grad_input_tensor_for_mlp


@pytest.fixture
def input_data():
    batch, seq, d_sae, d_model = 2, 3, 4, 5
    W_enc = torch.randn(d_model, d_sae)
    input_act = torch.zeros(batch, seq, d_sae)

    # Set some activations
    input_act[0, 0, 1] = 1
    input_act[0, 2, 3] = 1
    input_act[1, 1, 2] = 1

    return W_enc, input_act


def test_construct_sparse_grad_input_tensor_for_mlp(input_data):
    W_enc, input_act = input_data
    sparse_tensor = construct_sparse_grad_input_tensor_for_mlp(W_enc, input_act)

    assert sparse_tensor.shape == (2, 3, 4, 5)
    assert sparse_tensor._nnz() == 3
    # Check that input data is correctly mapped to the sparse tensor
    assert torch.allclose(sparse_tensor[0, 0, 1], W_enc[:, 1])
    assert torch.allclose(sparse_tensor[0, 2, 3], W_enc[:, 3])
    assert torch.allclose(sparse_tensor[1, 1, 2], W_enc[:, 2])
