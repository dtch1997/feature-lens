# type: ignore

from __future__ import annotations

import networkx
import torch

from jaxtyping import Float
from typing import Iterable
from transformer_lens import ActivationCache

from .types import Node, Head, SAE, Transcoder
from .cache_handler import CacheHandler
from .model_handler import ModelHandler
from .utils import iter_all_nodes_at_head, iter_upstream_heads


def compute_direct_path_attribution(
    cache_handler: CacheHandler,
    upstream_node: Node,
    downstream_node: Node,
):
    """Compute a score that measures how much an upstream node directly affects a downstream node

    The Direct Path Attribution (DPA) from an upstream feature F to a downstream feature G
    is the amount by which the activation of G decreases when we ablate the direct-path contribution of F.

    Concretely:
    - Let a(F) and a(G) be the clean acts of F and G.
    - Let a’(G) be act of G when we subtract a(F)*decoder_col(F) from the input to the head where G lives.
    - DPA(F -> G) = a(G) - a’(G)
    """

    # Get the activations of the downstream node
    downstream_acts = cache_handler.get_act(downstream_node)

    # Get the activations of the upstream node
    upstream_acts = cache_handler.get_act(upstream_node)

    # Get the activations of the downstream node when we ablate the direct-path contribution of the upstream node
    # TODO: calculate using encoder and relu
    ablated_acts = 0

    raise NotImplementedError("Need to calculate ablated_acts using encoder and relu")

    return downstream_acts - ablated_acts


def compute_metric_attribution(
    cache_handler: CacheHandler,
    upstream_node: Node,
    downstream_node: Node,
):
    """Compute a score that measures how important an edge is for the metric

    Concretely: MA(F -> G) = DPA(F -> G) * d(metric) / da(G)
    """
    # Get the metric of the downstream node
    dpa = compute_direct_path_attribution(cache_handler, upstream_node, downstream_node)
    metric_grad = cache_handler.get_grad_metric_wrt_act(downstream_node)
    return dpa * metric_grad


def get_sae_act_post_grad_input(
    sae: SAE | Transcoder,
    cache: ActivationCache,
) -> Float[torch.Tensor, "batch seq d_sae"]:
    """Get the gradient of the post-ReLU activations of the SAE w.r.t the input to that head

    Remarks
    - We cannot cache these gradients due to memory concerns
    - To avoid blowup in memory, the features under consideration should be filtered
        - e.g. gradients for non-active features will be zero
    """

    if isinstance(sae, SAE):
        # Attention-out SAE

        # NOTE: currently implement only the OV step
        # Conceptually, this is just (W_enc row) @ W_V @ attention_pattern
        # - attention pattern from the clean cache
        # - W_enc from the SAE
        # - W_V from the attention head

        # TODO: Q, K steps
        raise NotImplementedError("Need to implement OV step")

    elif isinstance(sae, Transcoder):
        return sae.W_enc


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
        for downstream_node in self.iter_important_nodes_at_head(head):
            # Compute the gradient d(node_act)/d(head_input)
            grad = get_sae_act_post_grad_input(
                self.model_handler, self.cache_handler, downstream_node
            )

            grad /= self.cache_handler.get_layernorm_scale(head.layer)

            # TODO: implement batched computation of MA
            for upstream_head in iter_upstream_heads(head):
                for upstream_node in iter_all_nodes_at_head(
                    head=upstream_head,
                    n_features=self.model_handler.get_n_features_at_head(upstream_head),
                    n_tokens=self.cache_handler.seq_len,
                ):
                    ma = compute_metric_attribution(
                        self.cache_handler, upstream_node, downstream_node
                    )
                    # Add to graph any edges with large MA
                    if ma > self.threshold:
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
