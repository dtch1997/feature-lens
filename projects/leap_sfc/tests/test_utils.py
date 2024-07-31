from projects.leap_sfc.leap.utils import Head, Node, iter_all_heads, iter_upstream_heads, iter_all_nodes_at_head

def test_iter_all_heads():
    n_layers = 2
    heads = list(iter_all_heads(n_layers))
    assert len(heads) == 4
    assert heads == [
        Head(0, "att"),
        Head(0, "mlp"),
        Head(1, "att"),
        Head(1, "mlp"),
    ]

def test_iter_upstream_heads():
    head = Head(1, "mlp")
    upstream_heads = list(iter_upstream_heads(head))
    assert len(upstream_heads) == 3
    assert upstream_heads == [
        Head(0, "att"),
        Head(0, "mlp"),
        Head(1, "att"),
    ]

def test_iter_all_nodes_at_head():
    head = Head(0, "att")
    n_features = 2
    n_tokens = 3
    nodes = list(iter_all_nodes_at_head(head, n_features, n_tokens))
    assert len(nodes) == 6
    assert nodes == [
        Node(0, "att", 0, 0),
        Node(0, "att", 0, 1),
        Node(0, "att", 0, 2),
        Node(0, "att", 1, 0),
        Node(0, "att", 1, 1),
        Node(0, "att", 1, 2),
    ]