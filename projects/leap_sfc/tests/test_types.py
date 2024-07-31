import pytest
import torch
from typing import Literal

from projects.leap_sfc.leap.types import Head, Node, Edge

def test_head():
    # Test Head creation
    head = Head(1, "mlp")
    assert head.layer == 1
    assert head.head_type == "mlp"

    # Test Head to_string method
    assert head.to_string() == "1.mlp"

    # Test Head from_string method
    head_from_string = Head.from_string("2.att")
    assert head_from_string.layer == 2
    assert head_from_string.head_type == "att"

def test_node():
    # Test Node creation
    node = Node(11, "attn", 23421, 15)
    assert node.layer == 11
    assert node.head_type == "attn"
    assert node.feature_id == 23421
    assert node.token_pos == 15

    # Test Node to_string method
    assert node.to_string() == "11.attn.23421.15"

    # Test Node from_string method
    node_from_string = Node.from_string("12.metric.0.5")
    assert node_from_string.layer == 12
    assert node_from_string.head_type == "metric"
    assert node_from_string.feature_id == 0
    assert node_from_string.token_pos == 5

def test_edge():
    # Test Edge creation
    upstream = Node(1, "mlp", 100, 0)
    downstream = Node(2, "att", 200, 1)
    edge = Edge(upstream, downstream)
    assert edge.upstream == upstream
    assert edge.downstream == downstream

    # Test Edge to_string method
    assert edge.to_string() == "1.mlp.100.0->2.att.200.1"

    # Test Edge from_string method
    edge_from_string = Edge.from_string("3.att.300.2->4.mlp.400.3")
    assert edge_from_string.upstream.to_string() == "3.att.300.2"
    assert edge_from_string.downstream.to_string() == "4.mlp.400.3"