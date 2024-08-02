from .types import Head, Node
from typing import Iterable


def iter_all_heads(n_layers: int) -> Iterable[Head]:
    """Iterate over all heads in the model, in order"""
    for layer in range(n_layers):
        for head_type in ["att", "mlp"]:
            yield Head(layer, head_type)


def iter_upstream_heads(head: Head) -> Iterable[Head]:
    """Iterate over all heads that are upstream of the given head, in order"""
    for layer in range(head.layer):
        for head_type in ["att", "mlp"]:
            yield Head(layer, head_type)

    # The MLP head in a layer is downstream of the ATT head
    if head.head_type == "mlp":
        yield Head(head.layer, "att")


def iter_all_nodes_at_head(
    head: Head, n_features: int, n_tokens: int
) -> Iterable[Node]:
    """Get all the nodes at a head"""
    for feature_id in range(n_features):
        for token_pos in range(n_tokens):
            yield Node(head.layer, head.head_type, feature_id, token_pos)
