from __future__ import annotations

import torch
import networkx
from feature_lens.core.types import HookName
from feature_lens.utils.data_handler import DataHandler

from jaxtyping import Float
from sae_lens import SAE, HookedSAETransformer
from feature_lens.nn.transcoder import Transcoder
from transformer_lens import ActivationCache

from dataclasses import dataclass
from typing import Literal, Iterable

HeadType = Literal["mlp", "att", "metric"]
Vector = Float[torch.Tensor, " d"]

@dataclass
class Head:
    layer: int
    head_type: HeadType

    def to_string(self) -> str:
        return f"{self.layer}.{self.head_type}"
    
    @staticmethod
    def from_string(s: str) -> "Head":
        layer, head_type = s.split(".")
        return Head(int(layer), head_type)

@dataclass(frozen=True)
class Node:
    """
    A node is a tuple (feature, token_position)

    The feature itself is specified by:
        1. Layer
        2. Head type
        3. Feature_id (the d_sae index)

    So a node might look like f“{layer}.{head}.{id}.{pos}”
        - e.g. “11.attn.23421.15”

    By convention, we count the metric as a (terminal) node.
        - e.g. if n_layers = 12, we have nodes f“12.metric.0.{pos}” for each token position

    """
    layer: int
    head_type: HeadType
    feature_id: int # The position of the feature in the feature vector
    token_pos: int # The position of the token in the sequences

    def to_string(self) -> str:
        return f"{self.layer}.{self.head_type}.{self.feature_id}.{self.token_pos}"
    
    @staticmethod
    def from_string(s: str) -> "Node":
        layer, head_type, feature_id, token_pos = s.split(".")
        return Node(int(layer), head_type, int(feature_id), int(token_pos))

@dataclass(frozen=True)
class Edge:
    upstream: Node
    downstream: Node

    def to_string(self) -> str:
        return f"{self.upstream.to_string()}->{self.downstream.to_string()}"
    
    @staticmethod
    def from_string(s: str) -> "Edge":
        upstream, downstream = s.split("->")
        return Edge(Node.from_string(upstream), Node.from_string(downstream))
    
class ModelHandler:
    """ Handles a model and SAEs """
    model: HookedSAETransformer
    saes: dict[HookName, SAE]
    transcoders: dict[HookName, Transcoder]

    def __init__(self, model: HookedSAETransformer, saes: dict[HookName, SAE], transcoders: dict[HookName, Transcoder]):
        self.model = model
        self.saes = saes
        self.transcoders = transcoders

    @property 
    def n_layers(self) -> int:
        return self.model.cfg.n_layers

    def run_with_cache(data_handler: DataHandler) -> CacheHandler:
        """ Runs the model on some data, populating a cache """
        # TODO: implement hooks to get cached activations
        # TODO: implement running the model on the data
        raise NotImplementedError("Need to implement run_with_cache")
    
    def get_upstream_heads(self, head: Head) -> list[Head]:
        """ Get the upstream heads of a head """
        raise NotImplementedError("Need to implement get_upstream_heads")

    def get_n_features_at_head(self, head: Head) -> int:
        """ Get the number of features at a head """
        if head.head_type == "mlp":
            return self.transcoders[head.layer].W_enc.shape[0]
        elif head.head_type == "att":
            return self.saes[head.layer].W_enc.shape[0]
        else:
            raise ValueError(f"Invalid head type: {head.head_type}")
 
    def get_encoder_weight(self, node: Node) -> Float[torch.Tensor, " d_model"]:
        if node.head_type == "mlp":
            return self.transcoders[node.layer].W_enc[node.feature_id]
        elif node.head_type == "att":
            return self.saes[node.layer].W_enc[node.feature_id]
        else:
            raise ValueError(f"Invalid head type: {node.head_type}")
    
    def get_decoder_weight(self, node: Node) -> Float[torch.Tensor, " d_model"]:
        """ Returns the decoder weight for a given node """
        if node.head_type == "mlp":
            return self.transcoders[node.layer].W_dec[node.feature_id]
        elif node.head_type == "att":
            return self.saes[node.layer].W_dec[node.feature_id]
        else:
            raise ValueError(f"Invalid head type: {node.head_type}")        
    
def iter_all_nodes_at_head(head: Head, n_features: int, n_tokens: int) -> Iterable[Node]:
    """ Get all the nodes at a head """
    for feature_id in range(n_features):
        for token_pos in range(n_tokens):
            yield Node(head.layer, head.head_type, feature_id, token_pos)

class CacheHandler:
    """ Handles a clean and corrupt cache """
    clean_cache: ActivationCache
    corrupt_cache: ActivationCache

    def __init__(self, clean_cache: ActivationCache, corrupt_cache: ActivationCache):
        self.clean_cache = clean_cache
        self.corrupt_cache = corrupt_cache

    @property 
    def n_token(self) -> int:
        """ Returns the number of token positions in the cached activations """
        raise NotImplementedError("Need to implement n_token")

    def get_layernorm_scale(layer: int) -> Float[torch.Tensor, " d_model"]:
        """ Returns the layernorm scale for a layer """
        raise NotImplementedError("Need to implement get_layernorm_scale")

    def get_act(self, node: Node, corrupt: bool = False) -> Float[torch.Tensor, " d_model"]:
        """ Returns the activation of a node """
        raise NotImplementedError("Need to implement get_activation")
    
    def get_grad_metric_wrt_act(self, node: Node, corrupt: bool = False) -> Float[torch.Tensor, " d_model"]:
        """ Returns the gradient of the metric with respect to the node """
        raise NotImplementedError("Need to implement get_metric_grad")
    
    def get_grad_act_wrt_input(self, node: Node, corrupt: bool = False) -> Float[torch.Tensor, " d_model"]:
        """ Returns the gradient of the node with respect to its head input """
        raise NotImplementedError("Need to implement get_grad_wrt_input")

def compute_direct_path_attribution(
    cache_handler: CacheHandler,
    upstream_node: Node,
    downstream_node: Node,
):
    """ Compute a score that measures how much an upstream node directly affects a downstream node

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
    """ Compute a score that measures how important an edge is for the metric
    
    Concretely: MA(F -> G) = DPA(F -> G) * d(metric) / da(G)
    """
    # Get the metric of the downstream node
    dpa = compute_direct_path_attribution(cache_handler, upstream_node, downstream_node)
    metric_grad = cache_handler.get_grad_metric_wrt_act(downstream_node)
    return dpa * metric_grad

def compute_grad_wrt_input(
    model_handler: ModelHandler,
    cache_handler: CacheHandler,
    node: Node,
):
    raise NotImplementedError("Need to implement compute_grad_wrt_input")

class LeapAlgo:
    graph: networkx.DiGraph
    model_handler: ModelHandler
    cache_handler: CacheHandler
    threshold: float

    def __init__(self, model_handler: ModelHandler, cache_handler: CacheHandler, threshold: float):
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
            grad = compute_grad_wrt_input(
                self.model_handler,
                self.cache_handler,
                downstream_node
            )

            grad /= self.cache_handler.get_layernorm_scale(head.layer)

            # TODO: implement batched computation of MA
            for upstream_head in self.model_handler.get_upstream_heads(head):
                for upstream_node in iter_all_nodes_at_head(
                    head = upstream_head, 
                    n_features = self.model_handler.get_n_features_at_head(upstream_head), 
                    n_tokens = self.cache_handler.n_token
                ):

                    ma = compute_metric_attribution(self.cache_handler, upstream_node, downstream_node)
                    # Add to graph any edges with large MA
                    if ma > self.threshold:
                        self.graph.add_edge(upstream_node, downstream_node)

            # TODO: implement other pruning strategies
            # - Prune by absolute threshold
            # - Prune by top k

    def run_leap(self):
        for layer in reversed(self.model_handler.n_layers):
            for head_type in ["mlp", "att"]:
                # NOTE: "att" currently handles 'OV' circuit only
                # TODO: implement "q", "k" circuits
                head = Head(layer, head_type)
                self.leap_step(head)