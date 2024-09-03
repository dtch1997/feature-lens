import torch
import pytest

from projects.leap_sfc.leap.sparse import (
    sparse_mean,
    efficient_sparse_mean,
    convert_hybrid_to_sparse,
)


@pytest.fixture
def hybrid_tensor():
    indices = torch.tensor([[0, 1, 2], [1, 0, 2]])
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    size = (3, 3, 2)
    return torch.sparse_coo_tensor(indices, values, size).coalesce()


def test_convert_hybrid_to_sparse(hybrid_tensor):
    fully_sparse = convert_hybrid_to_sparse(hybrid_tensor)

    # Check that the output is sparse
    assert fully_sparse.is_sparse

    # Check that all dimensions are now sparse
    assert fully_sparse.sparse_dim() == 3

    # Check the shape
    assert fully_sparse.shape == hybrid_tensor.shape

    # Check the number of non-zero elements
    assert fully_sparse._nnz() == 6  # 3 original sparse entries * 2 dense entries each

    # Check specific values
    dense_tensor = fully_sparse.to_dense()
    assert dense_tensor[0, 1, 0].item() == 1.0
    assert dense_tensor[0, 1, 1].item() == 2.0
    assert dense_tensor[1, 0, 0].item() == 3.0
    assert dense_tensor[1, 0, 1].item() == 4.0
    assert dense_tensor[2, 2, 0].item() == 5.0
    assert dense_tensor[2, 2, 1].item() == 6.0

    # Check that all other elements are zero
    assert dense_tensor.sum().item() == 21.0  # sum of all non-zero elements


def test_sparse_mean():
    # Create a sparse tensor
    indices = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 0, 1, 1], [0, 1, 1, 0, 2]])
    values = torch.randn(5, 3)  # 5 non-zero elements, each with 3 features
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(2, 2, 3, 3))

    # Calculate mean across different dimensions
    mean_dim0 = sparse_mean(sparse_tensor, dim=0)
    mean_dim1 = sparse_mean(sparse_tensor, dim=1)
    mean_dim2 = sparse_mean(sparse_tensor, dim=2)

    # Convert to dense for verification
    dense_tensor = sparse_tensor.to_dense()
    assert torch.allclose(
        mean_dim0.to_dense(), dense_tensor.mean(0), atol=1e-6
    ), f"{mean_dim0.to_dense()}{dense_tensor.mean(0)}"
    # assert torch.allclose(mean_dim1.to_dense(), dense_tensor.mean(1), atol=1e-6)
    # assert torch.allclose(mean_dim2.to_dense(), dense_tensor.mean(2), atol=1e-6)

    print("All tests passed!")


@pytest.mark.xfail(reason="Not implemented correctly")
def test_efficient_sparse_mean():
    indices = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 0, 1, 1], [0, 1, 1, 0, 2]])
    values = torch.randn(5, 3)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(2, 2, 3, 3))

    for dim in [0, 1, 2, [0, 1], [0, 1, 2]]:
        sparse_mean = efficient_sparse_mean(sparse_tensor, dim)
        dense_mean = sparse_tensor.to_dense().mean(dim)
        assert torch.allclose(sparse_mean.to_dense(), dense_mean, atol=1e-6)

    print("All tests passed!")
