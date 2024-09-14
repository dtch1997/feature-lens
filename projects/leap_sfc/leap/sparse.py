import torch
from typing import NewType

SparseTensor = NewType("SparseTensor", torch.Tensor)


def convert_hybrid_to_sparse(hybrid_tensor):
    """
    Convert a hybrid tensor where the last dimension is dense to a fully sparse tensor.
    """
    # Get the indices and values of the hybrid tensor
    indices = hybrid_tensor.indices()  # (sparse_dim, nnz)
    values = hybrid_tensor.values()  # (nnz, dense_dim)

    assert values.dim() == 2
    dense_dim = values.size(1)

    # Create the new indices and values
    # TODO: change because it's very slow
    new_indices = []
    new_values = []

    for sparse_idx, val in zip(indices.t(), values):
        for dense_idx in range(dense_dim):
            new_idx = torch.cat(
                (
                    sparse_idx,
                    torch.tensor(
                        [dense_idx],
                        device=hybrid_tensor.device,
                        dtype=hybrid_tensor.dtype,
                    ),
                )
            )
            new_val = val[dense_idx]
            new_indices.append(new_idx)
            new_values.append(new_val)

    new_indices = torch.stack(new_indices, dim=0)
    new_values = torch.stack(new_values, dim=0)
    return torch.sparse_coo_tensor(
        new_indices.t(),
        new_values,
        hybrid_tensor.size(),
        device=hybrid_tensor.device,
        dtype=hybrid_tensor.dtype,
    )


@torch.jit.script
def efficient_convert_hybrid_to_sparse(hybrid_tensor: torch.Tensor) -> torch.Tensor:
    """
    Efficiently convert a hybrid tensor where the last dimension is dense to a fully sparse tensor.
    This function is torch compilable.
    """
    assert hybrid_tensor.is_sparse, "Input tensor must be a sparse tensor"
    assert (
        hybrid_tensor.dense_dim() == 1
    ), "Input tensor must have exactly one dense dimension"

    indices = hybrid_tensor.indices()  # (sparse_dim, nnz)
    values = hybrid_tensor.values()  # (nnz, dense_dim)
    nnz, dense_dim = values.shape

    # Create new indices
    dense_indices = torch.arange(dense_dim, device=hybrid_tensor.device)
    old_sparse_indices = indices.repeat_interleave(dense_dim, dim=1)
    append_sparse_indices = dense_indices.repeat(nnz).unsqueeze(0)
    new_indices = torch.cat([old_sparse_indices, append_sparse_indices], dim=0)

    # Flatten values
    new_values = values.reshape(-1)

    return torch.sparse_coo_tensor(
        new_indices,
        new_values,
        hybrid_tensor.size(),
        device=hybrid_tensor.device,
        dtype=hybrid_tensor.dtype,
    )


def sparse_mean(sparse_tensor, dim):
    if not isinstance(dim, (list, tuple)):
        dim = [dim]

    # Sort dimensions in descending order
    dim = sorted(dim, reverse=True)

    # Get indices and values
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()

    # Calculate the size of the resulting tensor
    new_size = list(sparse_tensor.size())
    for d in dim:
        new_size.pop(d)

    # Create a new index tensor, removing the dimensions we're averaging over
    new_indices = []
    for i in range(indices.size(0)):
        if i not in dim:
            new_indices.append(indices[i])
    new_indices = (
        torch.stack(new_indices) if new_indices else torch.empty(0, indices.size(1))
    )

    # Group the values by the new indices
    if new_indices.numel() > 0:
        unique_indices, inverse_indices = torch.unique(
            new_indices, dim=1, return_inverse=True
        )
        summed_values = torch.zeros(
            unique_indices.size(1),
            values.size(1),
            dtype=values.dtype,
            device=values.device,
        )
        summed_values.index_add_(0, inverse_indices, values)
        counts = torch.zeros(
            unique_indices.size(1), dtype=torch.long, device=values.device
        )
        counts.index_add_(
            0,
            inverse_indices,
            torch.ones(inverse_indices.shape, device=values.device, dtype=torch.long),
        )
    else:
        raise ValueError("Cannot average over all dimensions")
        summed_values = values.sum(0, keepdim=True)
        counts = torch.tensor([indices.size(1)], dtype=torch.long, device=values.device)

    # Calculate the total number of elements in the averaged dimensions
    total_elements = 1
    for d in dim:
        total_elements *= sparse_tensor.size(d)

    # Adjust the mean values by the total number of elements
    mean_values = summed_values / (counts.unsqueeze(1) * total_elements)

    # Create the result sparse tensor
    result = torch.sparse_coo_tensor(unique_indices, mean_values, size=new_size)

    return result


def efficient_sparse_mean(sparse_tensor, dim):
    raise NotImplementedError("efficient_sparse_mean is not implemented yet")


def sparse_rearrange(x: SparseTensor, perm: list[int]):
    """
    x : a sparse COO tensor
    perm (list): a permutation of the dimensions of x
    return x with the dimensions permuted
    """
    assert set(perm) == set(range(x.dim()))
    x = x.coalesce()
    inds = x.indices()
    new_inds = torch.stack([inds[i] for i in perm], dim=0)

    new_shape = [x.shape[i] for i in perm]
    return torch.sparse_coo_tensor(new_inds, x.values(), new_shape)


def sparse_repeat(x: SparseTensor, n_repeat: int):
    """
    x : a sparse COO tensor, shape [s1, s2, ...]
    n_repeat : an int
    return x repeated n_repeat times on the 0th dimension, shape [n_repeat, s1, s2, ...]
    """
    x = x.coalesce()
    inds = x.indices()
    vals = x.values()

    # repeat the indices
    first_row = torch.arange(n_repeat, device=inds.device, dtype=inds.dtype).repeat(
        inds.shape[1]
    )
    repeated_inds = inds.repeat_interleave(n_repeat, dim=1)
    rep_inds = torch.cat([first_row.unsqueeze(0), repeated_inds], dim=0)

    # repeat the values
    rep_vals = (
        torch.tensor([val for val in vals for _ in range(n_repeat)])
        .to(vals.device)
        .to(vals.dtype)
    )

    return torch.sparse_coo_tensor(rep_inds, rep_vals, (n_repeat,) + x.shape)
