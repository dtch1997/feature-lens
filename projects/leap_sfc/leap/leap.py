# type: ignore

from __future__ import annotations

import networkx
import torch

from einops import einsum
from jaxtyping import Float
from typing import Iterable

from .types import Node, Head
from .cache_handler import CacheHandler
from .model_handler import ModelHandler
from .utils import iter_upstream_heads
from .sparse import sparse_mean

# Define a sparse tensor type annotation
SparseTensor = torch.Tensor


def construct_sparse_grad_input_tensor_for_mlp(
    W_enc: Float[torch.Tensor, "d_model d_sae"],
    input_act: Float[torch.Tensor, "batch query_seq d_sae"],
) -> Float[SparseTensor, "batch query_seq down_d_sae key_seq d_model"]:
    # NOTE: Include key seq dimension in return tensor t
    # - for MLP transcoders, t[:,q,:,k,:] = 0 iff k < q
    """Compute d(node_act)/d(head_input) for MLP transcoders

    Inputs:
    - W_enc: [d_model, d_sae] - the encoder weights
    - input_act: [batch, seq, d_sae] - the activations of the SAE

    Outputs:
    - sparse_tensor: [batch, query_seq, d_sae, key_seq, d_model] - the sparse tensor of gradients
    """
    batch, seq, d_sae = input_act.shape
    d_model = W_enc.shape[0]

    # Find nonzero activations
    nonzero_mask = input_act != 0
    nonzero_indices = nonzero_mask.nonzero()

    # Add key seq dimension
    # [batch, query_seq, d_sae] -> [batch, query_seq, d_sae, key_seq]
    # In practice, just do this by repeating the query_seq dimension
    nonzero_indices = torch.cat(
        [nonzero_indices, nonzero_indices[:, 1].unsqueeze(1)], dim=1
    )  # [n_nonzero, 4]

    # Get the corresponding rows from W_enc
    Wenc_values = W_enc[:, [idx[2] for idx in nonzero_indices]]  # [d_model, n_nonzero]

    # Create sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(
        indices=nonzero_indices.t(),
        values=Wenc_values.t(),
        size=(batch, seq, d_sae, seq, d_model),
    )

    return sparse_tensor


def construct_sparse_grad_input_tensor_for_att(
    W_enc: Float[torch.Tensor, "n_head d_head d_sae"],
    W_V: Float[torch.Tensor, "n_head d_head d_model"],
    attn_pattern: Float[torch.Tensor, "batch n_head query_seq key_seq"],
    input_act: Float[torch.Tensor, "batch query_seq down_d_sae"],
) -> Float[SparseTensor, "batch query_seq down_d_sae key_seq d_model"]:
    """Compute d(node_act)/d(head_input) for attention-out SAEs

    Inputs:
    - W_enc: [n_head, d_head, d_sae] - the encoder weights
    - W_V: [n_head, d_head, d_model] - the attention weights
    - attn_pattern: [batch, n_head, query_seq, key_seq] - the attention pattern
    - input_act: [batch, seq, d_sae] - the activations of the SAE

    Outputs:
    - sparse_tensor: [batch, query_seq, d_sae, key_seq, d_model] - the sparse tensor of gradients

    In the returned sparse tensor, the d_model dimension is dense (i.e. not sparse)
    """
    batch, seq, d_sae = input_act.shape
    n_head, d_head, d_sae = W_enc.shape
    n_head, d_head, d_model = W_V.shape
    batch, n_head, query_seq, key_seq = attn_pattern.shape

    nonzero_indices = input_act != 0
    nonzero_indices = nonzero_indices.nonzero()  # [n_nonzero, 3]

    example_indices = nonzero_indices[:, 0]  # [n_nonzero]
    feature_indices = nonzero_indices[:, 2]  # [n_nonzero]
    token_indices = nonzero_indices[:, 1]  # [n_nonzero]

    Wenc_values = W_enc[:, :, feature_indices]  # [n_head, d_head, n_nonzero]
    pattern_values = attn_pattern[
        example_indices, :, token_indices, :
    ]  # [n_nonzero, n_head, key_seq]

    grad = einsum(
        Wenc_values,
        pattern_values,
        W_V,
        "n_head d_head n_nonzero, n_nonzero n_head key_seq, n_head d_head d_model -> n_nonzero key_seq d_model",
    )

    # Create sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(
        indices=nonzero_indices.t(),
        values=grad,
        size=(batch, query_seq, d_sae, key_seq, d_model),
    )

    return sparse_tensor.coalesce()


def get_sae_act_post_grad_head_input(
    model_handler: ModelHandler,
    cache_handler: CacheHandler,
    downstream_head: Head,
    downstream_important_nodes: list[Node],
) -> Float[SparseTensor, "batch query_seq down_d_sae d_model"]:
    """Get the gradient of the post-ReLU activations of the SAE w.r.t the input to that head

    Remarks
    - We cannot cache these gradients due to memory concerns
    - To avoid blowup in memory, the features under consideration should be filtered
        - e.g. gradients for non-active features will be zero
    """
    # TODO: Implement filtering by downstream_important_nodes

    if downstream_head.head_type == "att":
        # Attention-out SAE
        # NOTE: This will be encoder weights, multiplied by W_V
        sae = model_handler.get_sae_for_head(downstream_head)
        W_enc = sae.W_enc
        W_V = model_handler.blocks[downstream_head.layer].attn.W_V
        # Hackily get the attention pattern
        attn_pattern = cache_handler.get_cache(downstream_head, "clean")[
            "blocks.0.attn.hook_pattern"
        ]
        input_act = cache_handler.get_act(downstream_head)
        return construct_sparse_grad_input_tensor_for_att(
            W_enc, W_V, attn_pattern, input_act
        )

    elif downstream_head.head_type == "mlp":
        W_enc = model_handler.get_sae_for_head(
            downstream_head
        ).W_enc  # [d_model, d_sae]
        # Get only the active features
        input_act = cache_handler.get_act(downstream_head)  # [batch, seq, d_sae]
        return construct_sparse_grad_input_tensor_for_mlp(W_enc, input_act)

    else:
        raise ValueError(f"Unknown head type: {downstream_head.head_type}")


def construct_sparse_grad_output_tensor_for_mlp(
    W_dec: Float[torch.Tensor, "up_d_sae d_model"],
    input_act: Float[torch.Tensor, "batch key_seq up_d_sae"],
) -> Float[SparseTensor, "batch key_seq up_d_sae d_model"]:
    """Construct a sparse tensor of d(head_output)/d(node_act) for MLP transcoders"""

    batch, seq, d_sae = input_act.shape
    d_model = W_dec.shape[1]

    # Find nonzero activations
    nonzero_mask = input_act != 0
    nonzero_indices = nonzero_mask.nonzero()  # [n_nonzero, 3]

    # Get the corresponding rows from W_dec
    values = W_dec[[idx[2] for idx in nonzero_indices], :]  # [n_nonzero, d_model]

    # Create sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(
        indices=nonzero_indices.t(),
        values=values,
        size=(batch, seq, d_sae, d_model),
    )

    return sparse_tensor


def construct_sparse_grad_output_tensor_for_att(
    W_dec: Float[torch.Tensor, "d_sae n_head d_head"],
    W_O: Float[torch.Tensor, "n_head d_head d_model"],
    input_act: Float[SparseTensor, "batch seq d_sae"],
) -> Float[SparseTensor, "batch key_seq up_d_sae d_model"]:
    """Construct a sparse tensor of d(head_output)/d(node_act) for attention-out SAEs"""
    W = einsum(
        W_dec, W_O, "d_sae n_head d_head, n_head d_head d_model -> d_sae d_model"
    )

    batch, seq, d_sae = input_act.shape
    d_model = W.shape[1]

    # Find nonzero activations
    nonzero_mask = input_act != 0
    nonzero_indices = nonzero_mask.nonzero()  # [n_nonzero, 3]

    # Get the corresponding rows from W_dec
    values = W[[idx[2] for idx in nonzero_indices], :]  # [n_nonzero, d_model]

    # Create sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(
        indices=nonzero_indices.t(),
        values=values,
        size=(batch, seq, d_sae, d_model),
    )

    return sparse_tensor.coalesce()


def get_sae_act_post_grad_head_output(
    model_handler: ModelHandler,
    cache_handler: CacheHandler,
    upstream_head: Head,
) -> Float[SparseTensor, "batch key_seq up_d_sae d_model"]:
    """Get the gradient of the output of that head w.r.t the post ReLU activations of the SAE

    Remarks
    - We cannot cache these gradients due to memory concerns
    - To avoid blowup in memory, the features under consideration should be filtered
        - e.g. gradients for non-active features will be zero
    """
    if upstream_head.head_type == "att":
        # Attention-out SAE
        # NOTE: This will be decoder weights, multiplied by W_V
        # NOTE: We need to multiply the W_V matrix here manually
        sae = model_handler.get_sae_for_head(upstream_head)
        W_dec = sae.W_dec
        W_O = model_handler.model.blocks[upstream_head.layer].attn.W_O
        input_act = cache_handler.get_act(upstream_head)
        return construct_sparse_grad_output_tensor_for_att(W_dec, W_O, input_act)

    elif upstream_head.head_type == "mlp":
        W_dec = model_handler.get_sae_for_head(upstream_head).W_dec  # [d_sae, d_model]
        # Get only the active features
        input_act = cache_handler.get_act(upstream_head)  # [batch, seq, d_sae]
        sparse_tensor = construct_sparse_grad_output_tensor_for_mlp(W_dec, input_act)
        return sparse_tensor

    else:
        raise ValueError(f"Unknown head type: {upstream_head.head_type}")


def get_grad_metric_wrt_act(
    cache_handler: CacheHandler,
    downstream_head: Head,
    downstream_important_nodes: list[Node],
):
    grad = cache_handler.get_grad_metric_wrt_act(
        downstream_head
    )  # [batch, query_seq, down_d_sae]
    grad_sparse = grad.to_sparse()

    # Filter grad_sparse by downstream_important_nodes
    for idx in grad_sparse.indices():
        _, query_pos, down_feature = idx
        if (
            Node(
                downstream_head.layer,
                downstream_head.head_type,
                query_pos,
                down_feature,
            )
            not in downstream_important_nodes
        ):
            grad_sparse.values()[idx] = 0

    return grad_sparse


class LeapAlgo:
    graph: networkx.DiGraph
    model_handler: ModelHandler
    cache_handler: CacheHandler
    threshold: float

    def __init__(
        self, model_handler: ModelHandler, cache_handler: CacheHandler, threshold: float
    ):
        self.graph = networkx.DiGraph()
        self.model_handler = model_handler
        self.cache_handler = cache_handler
        self.threshold = threshold

    def iter_important_nodes_at_head(self, head: Head) -> Iterable[Node]:
        # So we can just return the nodes at the head
        def is_node_at_head(node: Node) -> bool:
            return node.head_type == head.head_type and node.layer == head.layer

        for node in self.graph.nodes:
            if is_node_at_head(node):
                yield node

    def leap_step(self, head: Head):
        downstream_head = head
        # TODO: Handle filtering downstream head by important nodes
        downstream_important_nodes = list(
            self.iter_important_nodes_at_head(downstream_head)
        )

        grad_down_act_head_input = get_sae_act_post_grad_head_input(
            self.model_handler,
            self.cache_handler,
            downstream_head,
            downstream_important_nodes,
        )
        # TODO: Implement this correctly

        # grad_down_act_head_input /= self.cache_handler.get_layernorm_scale(
        #     downstream_head
        # )

        for upstream_head in iter_upstream_heads(downstream_head):
            # TODO: can we implement this in a batched way instead of looping over each head?
            grad_head_output_up_act = get_sae_act_post_grad_head_output(
                self.model_handler, self.cache_handler, upstream_head
            )
            up_act = self.cache_handler.get_act(upstream_head)

            dpa = einsum(
                up_act,
                grad_head_output_up_act,
                grad_down_act_head_input,
                "batch key_seq up_d_sae, batch key_seq up_d_sae d_model, batch query_seq down_d_sae d_model, -> batch query_seq down_d_sae key_seq up_d_sae",
            )
            metric_grad = get_grad_metric_wrt_act(
                self.cache_handler, downstream_head, downstream_important_nodes
            )

            # NOTE: This has to be done using sparse tensors due to memory concerns
            ma = einsum(
                dpa,
                metric_grad,
                "batch query_seq down_d_sae key_seq up_d_sae, batch query_seq down_d_sae -> batch query_seq down_d_sae key_seq up_d_sae",
            )

            # Batch mean
            ma_batchmean = sparse_mean(ma, dim=0)

            # Add the edge if the score is above the threshold
            for idx, val in zip(ma_batchmean.indices(), ma_batchmean.values()):
                query_pos, down_feature, key_pos, up_feature = idx
                if val > self.threshold:
                    upstream_node = Node(
                        upstream_head.layer,
                        upstream_head.head_type,
                        key_pos,
                        up_feature,
                    )
                    downstream_node = Node(
                        downstream_head.layer,
                        downstream_head.head_type,
                        query_pos,
                        down_feature,
                    )
                    assert downstream_node in self.graph
                    self.graph.add_edge(upstream_node, downstream_node, score=val)
                    # NOTE: the above also adds upstream_node to the graph

            # TODO: implement other pruning strategies
            # - Prune by absolute threshold
            # - Prune by top k

    def run_leap(self):
        """Computes edge attribution scores for every pair of nodes in the model

        At the end, the graph will contain selected edges
        """
        for layer in reversed(list(range(self.model_handler.n_layers))):
            for head_type in ["mlp", "att"]:
                print(f"Layer {layer}, head type {head_type}")
                # NOTE: "att" currently handles 'OV' circuit only
                # TODO: implement "q", "k" circuits
                head = Head(layer, head_type)
                self.leap_step(head)

        return self
