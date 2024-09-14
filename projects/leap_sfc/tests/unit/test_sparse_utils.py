import pytest
import torch

from projects.leap_sfc.leap.sparse_utils import (
    sparse_rearrange,
    sparse_repeat,
    sparse_dot_product,
)


@pytest.fixture
def sample_sparse_tensor():
    indices = torch.tensor([[0, 1, 2], [2, 0, 1]])
    values = torch.tensor([1.0, 2.0, 3.0])
    return torch.sparse_coo_tensor(indices, values, size=(3, 3))


@pytest.fixture
def sample_sparse_hybrid_tensor():
    indices = torch.tensor([[0, 1, 2], [2, 0, 1]])
    values = torch.tensor([[1.0], [2.0], [3.0]])
    return torch.sparse_coo_tensor(indices, values, size=(3, 3, 1))


def test_sparse_rearrange(sample_sparse_tensor):
    x = sample_sparse_tensor
    perm = (1, 0)
    result = sparse_rearrange(x, perm)
    assert result.shape == (3, 3)


def test_sparse_rearrange_with_hybrid_tensor(sample_sparse_hybrid_tensor):
    x = sample_sparse_hybrid_tensor
    perm = (1, 0)
    result = sparse_rearrange(x, perm)
    assert result.shape == (3, 3, 1)


@pytest.mark.parametrize("dim", [1])
def test_sparse_repeat_with_sparse_tensor(sample_sparse_tensor, dim):
    x = sample_sparse_tensor
    n_repeat = 3
    result = sparse_repeat(x, n_repeat, dim)

    expected_shape = list(x.shape)
    expected_shape.insert(dim, n_repeat)
    assert result.shape == tuple(expected_shape)


def test_sparse_repeat_with_hybrid_tensor(sample_sparse_hybrid_tensor):
    x = sample_sparse_hybrid_tensor
    n_repeat = 3
    dim = 1
    result = sparse_repeat(x, n_repeat, dim)

    expected_shape = list(x.shape)
    expected_shape.insert(dim, n_repeat)
    assert result.shape == tuple(expected_shape)


def test_sparse_dot_product():
    # Create two sparse tensors with the same sparsity pattern
    indices = torch.tensor([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
    values1 = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(1)
    values2 = torch.tensor([4.0, 5.0, 6.0]).unsqueeze(1)
    x = torch.sparse_coo_tensor(indices, values1, (3, 3, 3, 1))
    y = torch.sparse_coo_tensor(indices, values2, (3, 3, 3, 1))

    result = sparse_dot_product(x, y)

    assert result.shape == (3, 3, 3)


def test_sparse_rearrange_invalid_perm(sample_sparse_tensor):
    x = sample_sparse_tensor
    invalid_perm = (0, 2)
    with pytest.raises(ValueError):
        sparse_rearrange(x, invalid_perm)


def test_sparse_dot_product_incompatible_shapes():
    x = torch.sparse_coo_tensor([[0, 1], [0, 1]], [1.0, 2.0], (2, 2))
    y = torch.sparse_coo_tensor([[0, 1], [0, 1]], [1.0, 2.0], (2, 3))

    with pytest.raises(RuntimeError):
        sparse_dot_product(x, y)
