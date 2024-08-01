from __future__ import annotations

import torch
from jaxtyping import Float
from dataclasses import dataclass
from typing import Literal

from feature_lens.core.types import HookName
from sae_lens import HookedSAETransformer, SAE  # noqa: F401
from feature_lens.nn.transcoder import Transcoder  # noqa: F401
from transformer_lens import ActivationCache  # noqa: F401

Model = HookedSAETransformer
HeadType = Literal["mlp", "att", "metric"]
Vector = Float[torch.Tensor, " d"]


@dataclass
class Head:
    """
    The basic component in the models' computational graph

    """

    layer: int
    head_type: HeadType

    def to_string(self) -> str:
        return f"{self.layer}.{self.head_type}"

    @staticmethod
    def from_string(s: str) -> "Head":
        layer, head_type = s.split(".")
        return Head(int(layer), head_type)  # type: ignore

    @property
    def hook_name_in(self) -> HookName:
        """Return the (input) hook name associated with this head"""
        if self.head_type == "mlp":
            return f"blocks.{self.layer}.ln2.hook_normalized"
        elif self.head_type == "att":
            return f"blocks.{self.layer}.attn.hook_z"
        else:  # metric
            raise ValueError("No hook name for metric head")

    @property
    def hook_name_out(self) -> HookName:
        """Return the (output) hook name associated with this head"""
        if self.head_type == "mlp":
            return f"blocks.{self.layer}.hook_mlp_out"
        elif self.head_type == "att":
            return f"blocks.{self.layer}.attn.hook_z"
        else:
            raise ValueError("No hook name for metric head")


@dataclass(frozen=True)
class Feature:
    """
    A tuple (layer, head_type, feature_id)

    """

    layer: int
    head_type: HeadType
    feature_id: int

    def to_string(self) -> str:
        return f"{self.layer}.{self.head_type}.{self.feature_id}"

    @staticmethod
    def from_string(s: str) -> "Feature":
        layer, head_type, feature_id = s.split(".")
        return Feature(int(layer), head_type, int(feature_id))  # type: ignore

    @property
    def head(self) -> Head:
        return Head(self.layer, self.head_type)

    @property
    def hook_name_in(self) -> HookName:
        return self.head.hook_name_in

    @property
    def hook_name_out(self) -> HookName:
        return self.head.hook_name_out


@dataclass(frozen=True)
class Node:
    """
    A tuple (feature, token_position)

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
    feature_id: int  # The position of the feature in the feature vector
    token_pos: int  # The position of the token in the sequences

    def to_string(self) -> str:
        return f"{self.layer}.{self.head_type}.{self.feature_id}.{self.token_pos}"

    @staticmethod
    def from_string(s: str) -> "Node":
        layer, head_type, feature_id, token_pos = s.split(".")
        return Node(int(layer), head_type, int(feature_id), int(token_pos))  # type: ignore

    @property
    def feature(self) -> Feature:
        return Feature(self.layer, self.head_type, self.feature_id)

    @property
    def hook_name_in(self) -> HookName:
        return self.feature.hook_name_in

    @property
    def hook_name_out(self) -> HookName:
        return self.feature.hook_name_out


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
