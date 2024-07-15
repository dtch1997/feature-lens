from transformer_lens import ActivationCache
from feature_lens.core.types import HookName, ForwardHook


def make_ablate_hook(
    ablate_cache: ActivationCache, hook_name: HookName, feature_id: int
) -> ForwardHook:
    def ablate_hook(act, hook):
        assert hook.name == hook_name
        ablate_act = ablate_cache[hook.name]
        assert act.shape == ablate_act.shape

        act[:, :, feature_id] = ablate_act[:, :, feature_id]
        return act

    return ForwardHook(hook_name, ablate_hook)
