from projects.leap_sfc.leap.types import ActivationCache, Vector, Node


class CacheHandler:
    """Handles a clean and corrupt cache

    Assumes that the relevant SAE activations have already been added to the respective caches
    """

    clean_cache: ActivationCache
    corrupt_cache: ActivationCache

    def __init__(self, clean_cache: ActivationCache, corrupt_cache: ActivationCache):
        self.clean_cache = clean_cache
        self.corrupt_cache = corrupt_cache

    @property
    def n_token(self) -> int:
        """Returns the number of token positions in the cached activations"""
        raise NotImplementedError("Need to implement n_token")

    def get_layernorm_scale(layer: int) -> Vector:
        """Returns the layernorm scale for a layer"""
        raise NotImplementedError("Need to implement get_layernorm_scale")

    def get_act(self, node: Node, corrupt: bool = False) -> Vector:
        """Returns the activation of a node"""
        raise NotImplementedError("Need to implement get_activation")

    def get_grad_metric_wrt_act(self, node: Node, corrupt: bool = False) -> Vector:
        """Returns the gradient of the metric with respect to the node"""
        raise NotImplementedError("Need to implement get_metric_grad")

    def get_grad_act_wrt_input(self, node: Node, corrupt: bool = False) -> Vector:
        """Returns the gradient of the node with respect to its head input"""
        raise NotImplementedError("Need to implement get_grad_wrt_input")
