import torch

SparseCOOTensor = torch.Tensor  # Alias for Sparse COO tensor


@torch.no_grad()
def sparse_rearrange(x: SparseCOOTensor, perm: tuple[int]) -> SparseCOOTensor:
    """
    Rearrange the sparse dimensions of a sparse COO tensor.

    Args:
        x: SparseCOOTensor of shape [s1, ..., sm, d1, ..., dn]
        perm: Permutation of the dimensions of x

    Returns:
        SparseCOOTensor of shape [s_perm[0], ..., s_perm[m], d1, ..., dn]
    """
    n_sparse = len(get_sparse_dim(x))
    if not set(perm) == set(range(n_sparse)):
        raise ValueError(f"Expected permutation of {n_sparse} elements; got {perm}")
    x = x.coalesce()
    inds = x.indices()
    new_inds = torch.stack([inds[i] for i in perm], dim=0)

    # new_shape = [x.shape[i] for i in perm]
    return torch.sparse_coo_tensor(new_inds, x.values()).coalesce()


@torch.no_grad()
def sparse_repeat(
    x: SparseCOOTensor, n_repeat: torch.Tensor, dim: int
) -> SparseCOOTensor:
    """
    Repeat a sparse COO tensor along the specified dimension.

    Args:
        x: SparseCOOTensor of shape [s1, ..., sm, d1, ..., dn]
        n_repeat: Number of times to repeat x

    Return:
        SparseCOOTensor of shape [s1, ..., sk-1, n_repeat, sk, ..., sm, d1, ..., dn]
    """

    x = x.coalesce()
    idx = x.indices()  # (n_sparse, nnz)
    vals = x.values()  # (nnz, d1, ..., dn)

    # repeat the indices
    # The new index should consist of torch.arange(n_repeat), interleaved
    new_idx = (
        torch.arange(n_repeat, device=idx.device, dtype=idx.dtype)
        .repeat(idx.shape[1])
        .unsqueeze(0)
    )
    orig_idx = idx.repeat_interleave(n_repeat, dim=1)

    # Insert the new index at the appropriate position
    rep_idx = torch.cat([orig_idx[:dim], new_idx, orig_idx[dim:]], dim=0)

    # repeat the values
    rep_vals = vals.repeat_interleave(n_repeat, dim=0)
    return torch.sparse_coo_tensor(rep_idx, rep_vals).coalesce()


@torch.no_grad()
def sparse_dot_product(x: SparseCOOTensor, y: SparseCOOTensor) -> SparseCOOTensor:
    """
    Dot product for hybrid sparse COO tensors in torch.

    Args:
        x: SparseCOOTensor of shape [s1, ..., sm, d1, ..., dn]
        y: SparseCOOTensor of shape [s1, ..., sm, d1, ..., dn]

    Returns:
        z: SparseCOOTensor of shape [s1, ..., sm]

    Einsum is calculated as follows.
    1. At each index (i1, ..., im, j1, ..., jn), take x[i1, ..., im] and y[i1, ..., im].
    2. Compute the dot product of x[i1, ..., im] and y[i1, ..., im].
    """

    # First, use elementwise multiplication to find the nonzero indices.
    # These indices are guaranteed to be nonzero.
    xy = (x * y).coalesce()
    xy_idx, xy_val = xy.indices(), xy.values()
    nnz = xy_val.shape[0]

    # Next, sum the values at each index.
    # This is equivalent to a dot product.
    z_val = xy_val.view(nnz, -1).sum(dim=-1)
    z_idx = xy_idx

    # Finally, create the output tensor.
    z = torch.sparse_coo_tensor(z_idx, z_val, device=x.device, dtype=x.dtype)
    return z.coalesce()


# Get the sparse dimensions of a hybrid COO tensor


def get_sparse_dim(x: SparseCOOTensor) -> tuple[int]:
    """
    Get the sparse dimensions of a hybrid COO tensor.

    Args:
        x: SparseCOOTensor of shape [s1, ..., sm, d1, ..., dn]

    Returns:
        (s1, ..., sm)
    """
    return x.shape[: x.sparse_dim()]


def get_dense_dim(x: SparseCOOTensor) -> tuple[int]:
    """
    Get the dense dimensions of a hybrid COO tensor.

    Args:
        x: SparseCOOTensor of shape [s1, ..., sm, d1, ..., dn]

    Returns:
        (d1, ..., dn)
    """
    return x.values().shape[1:]


def get_nnz(x: SparseCOOTensor) -> int:
    """
    Get the number of nonzero elements in a hybrid COO tensor.

    Args:
        x: SparseCOOTensor of shape [s1, ..., sm, d1, ..., dn]
    """
    return x.values().shape[0]
