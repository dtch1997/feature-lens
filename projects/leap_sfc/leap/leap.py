# type: ignore

from __future__ import annotations

import networkx
import torch

from einops import einsum, rearrange
from jaxtyping import Float
from typing import Iterable

from .types import Node, Head
from .cache_handler import CacheHandler
from .model_handler import ModelHandler
from .utils import iter_upstream_heads
from .sparse import sparse_mean


def compute_direct_path_attribution(
    model_handler: ModelHandler,
    cache_handler: CacheHandler,
    upstream_head: Head,
    downstream_head: Head,
) -> Float[torch.Tensor, "batch query_seq down_d_sae key_seq up_d_sae"]:
    """Compute a score that measures how much an upstream node directly affects a downstream node

    The Direct Path Attribution (DPA) from an upstream feature F to a downstream feature G
    is the amount by which the activation of G decreases when we ablate the direct-path contribution of F.

    Concretely:
    - Let a(F) and a(G) be the clean acts of F and G.
    - Let a’(G) be act of G when we subtract a(F)*decoder_col(F) from the input to the head where G lives.
    - DPA(F -> G) = a(G) - a’(G)
    """

    # Get the activations of the downstream node
    downstream_acts = cache_handler.get_act(upstream_head)

    # Get the activations of the upstream node
    upstream_acts = cache_handler.get_act(downstream_head)

    # Get the activations of the downstream node when we ablate the direct-path contribution of the upstream node
    ablated_acts = 0
    raise NotImplementedError("Need to calculate ablated_acts using encoder and relu")

    return downstream_acts - ablated_acts


def construct_sparse_grad_input_tensor_for_mlp(
    W_enc: Float[torch.Tensor, "d_model d_sae"],
    input_act: Float[torch.Tensor, "batch seq d_sae"],
) -> Float[torch.Tensor, "batch seq d_sae d_model"]:
    """Construct a sparse tensor of d(head_output)/d(node_act) for MLP transcoders

    Inputs:
    - W_enc: [d_model, d_sae] - the encoder weights
    - input_act: [batch, seq, d_sae] - the activations of the SAE

    Outputs:
    - sparse_tensor: [batch, seq, d_sae, d_model] - the sparse tensor of gradients
    """
    batch, seq, d_sae = input_act.shape
    d_model = W_enc.shape[0]

    # Find nonzero activations
    nonzero_mask = input_act != 0
    nonzero_indices = nonzero_mask.nonzero()

    # Get the corresponding rows from W_enc
    values = W_enc[:, [idx[2] for idx in nonzero_indices]]

    # Create sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(
        indices=nonzero_indices.t(),
        values=values.t(),
        size=(batch, seq, d_sae, d_model),
    )

    return sparse_tensor


def construct_sparse_grad_input_tensor_for_att(
    W_enc: Float[torch.Tensor, "n_head d_head d_sae"],
    W_V: Float[torch.Tensor, "n_head d_head d_model"],
    attn_pattern: Float[torch.Tensor, "batch n_head query_seq key_seq"],
    input_act: Float[torch.Tensor, "batch seq d_sae"],
) -> Float[torch.Tensor, "batch query_seq key_seq d_sae d_model"]:
    """Construct a sparse tensor of d(head_output)/d(node_act) for attention-out SAEs

    Calculates d(node_act)/d(head_input) for each node

    Inputs:
    - W_enc: [n_head, d_head, d_sae] - the encoder weights
    - W_V: [n_head, d_head, d_model] - the attention weights
    - attn_pattern: [batch, n_head, query_seq, key_seq] - the attention pattern
    - input_act: [batch, seq, d_sae] - the activations of the SAE

    Outputs:
    - sparse_tensor: [batch, query_seq, key_seq, d_sae, d_model] - the sparse tensor of gradients
    """
    batch, seq, d_sae = input_act.shape
    n_head, d_head, d_sae = W_enc.shape
    n_head, d_head, d_model = W_V.shape
    batch, n_head, query_seq, key_seq = attn_pattern.shape

    # Find nonzero activations
    nonzero_mask = input_act != 0
    nonzero_indices = nonzero_mask.nonzero()

    raise NotImplementedError("Need to implement construct_sparse_grad_tensor_for_att")


def get_sae_act_post_grad_head_input(
    model_handler: ModelHandler,
    cache_handler: CacheHandler,
    downstream_node: Node,
) -> Float[torch.Tensor, "batch down_layer query_seq down_d_sae d_model"]:
    """Get the gradient of the post-ReLU activations of the SAE w.r.t the input to that head

    Remarks
    - We cannot cache these gradients due to memory concerns
    - To avoid blowup in memory, the features under consideration should be filtered
        - e.g. gradients for non-active features will be zero
    """

    if downstream_node.head_type == "att":
        # Attention-out SAE

        # NOTE: currently implement only the OV step
        # Conceptually, this is just (W_enc row) @ W_V @ attention_pattern
        # - attention pattern from the clean cache
        # - W_enc from the SAE
        # - W_V from the attention head

        # TODO: Q, K steps
        raise NotImplementedError("Need to implement OV step")

    elif downstream_node.head_type == "mlp":
        W_enc = model_handler.get_sae_for_head(
            downstream_node
        ).W_enc  # [d_model, d_sae]
        # Get only the active features
        input_act = cache_handler.get_act(downstream_node)  # [batch, seq, d_sae]
        sparse_tensor = construct_sparse_grad_input_tensor_for_mlp(W_enc, input_act)
        return sparse_tensor
    else:
        raise ValueError(f"Unknown head type: {downstream_node.head_type}")


def construct_sparse_grad_output_tensor_for_mlp(
    W_dec: Float[torch.Tensor, "d_model d_sae"],
    input_act: Float[torch.Tensor, "batch seq d_sae"],
) -> Float[torch.Tensor, "batch seq d_sae d_model"]:
    """Construct a sparse tensor of d(head_output)/d(node_act) for MLP transcoders"""
    raise NotImplementedError(
        "Need to implement construct_sparse_grad_output_tensor_for_mlp"
    )


def construct_sparse_grad_output_tensor_for_att(
    W_dec: Float[torch.Tensor, "d_sae n_head d_head"],
    W_O: Float[torch.Tensor, "n_head d_head d_model"],
    input_act: Float[torch.Tensor, "batch seq d_sae"],
):
    """Construct a sparse tensor of d(head_output)/d(node_act) for attention-out SAEs"""
    raise NotImplementedError(
        "Need to implement construct_sparse_grad_output_tensor_for_att"
    )


def get_sae_act_post_grad_head_output(
    model_handler: ModelHandler,
    cache_handler: CacheHandler,
    upstream_head: Head,
) -> Float[torch.Tensor, "batch key_seq up_d_sae d_model"]:
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
        W_dec = sae.W_dec  # [d_sae, d_model]
        W_dec_reshape = rearrange(
            W_dec,
            "d_sae d_model -> d_sae n_head d_head",
            n_head=model_handler.model.cfg.n_heads,
        )
        W_V = model_handler.blocks[
            upstream_head.layer
        ].attn.W_V  # [n_head, d_head, d_model]
        input_act = cache_handler.get_act(upstream_head)
        attn_pattern = cache_handler.get_cache("clean")[
            f"blocks.{upstream_head.layer}.attn.hook_pattern"
        ]
        return construct_sparse_grad_output_tensor_for_att(
            W_dec_reshape, W_V, attn_pattern, input_act
        )

    elif upstream_head.head_type == "mlp":
        W_dec = model_handler.get_sae_for_head(upstream_head).W_dec  # [d_sae, d_model]
        # Get only the active features
        input_act = cache_handler.get_act(upstream_head)  # [batch, seq, d_sae]
        sparse_tensor = construct_sparse_grad_output_tensor_for_mlp(W_dec, input_act)
        return sparse_tensor

    else:
        raise ValueError(f"Unknown head type: {upstream_head.head_type}")


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
        # TODO: Handle filtering downstream head by important features...
        grad = get_sae_act_post_grad_head_input(
            self.model_handler, self.cache_handler, downstream_head
        )
        grad /= self.cache_handler.get_layernorm_scale(downstream_head)

        for upstream_head in iter_upstream_heads(downstream_head):
            dpa = compute_direct_path_attribution(
                self.model_handler,
                self.cache_handler,
                upstream_head,
                downstream_head,
            )
            metric_grad = self.cache_handler.get_grad_metric_wrt_act(downstream_head)
            # NOTE: This has to be done using sparse tensors due to memory concerns
            ma = einsum(
                dpa,
                metric_grad,
                "batch query_seq down_d_sae key_seq up_d_sae, batch query_seq down_d_sae -> batch query_seq down_d_sae key_seq up_d_sae",
            )

            # Batch mean
            ma_batchmean = sparse_mean(ma, dim=0)

            # Add the edge if the score is above the threshold
            for idx, val in zip(ma.indices(), ma.values()):
                batch, query_pos, down_feature, key_pos, up_feature = idx
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
                    self.graph.add_edge(upstream_node, downstream_node)

            # TODO: implement other pruning strategies
            # - Prune by absolute threshold
            # - Prune by top k

    def run_leap(self):
        """Computes edge attribution scores for every pair of nodes in the model

        At the end, the graph will contain selected edges
        """
        for layer in reversed(self.model_handler.n_layers):
            for head_type in ["mlp", "att"]:
                # NOTE: "att" currently handles 'OV' circuit only
                # TODO: implement "q", "k" circuits
                head = Head(layer, head_type)
                self.leap_step(head)
