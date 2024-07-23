
from feature_lens.core.types import *
from feature_lens.utils.device import get_device, set_device
from feature_lens.utils.viz import *
from feature_lens.utils.data_utils import *
from feature_lens.utils.load_pretrained import load_model, load_sae
from feature_lens.data.make_ioi import make_ioi
from feature_lens.data.string_handler import as_dataframe
from feature_lens.core.experiment import (
    ExperimentArtifact, 
    ArtifactSpec, 
    ExperimentStage, 
    ExperimentSpec,
    ExperimentRunner, 
    ExperimentState,
    InMemoryExperimentState
)
from feature_lens.cache import get_sae_cache

from typing import Any

import pathlib
import matplotlib.pyplot as plt

class LoadDataAndModels(ExperimentStage):
    """ Load the dataset, model, and SAEs. """

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        del inputs

        model = load_model("gpt2-small")
        ioi_data = make_ioi(model)
        sae_up = load_sae(release = "gpt2-small-res-jb", sae_id = "blocks.8.hook_resid_pre")
        sae_target = load_sae(release = "gpt2-small-hook-z-kk", sae_id="blocks.8.hook_z")
        sae_down = load_sae(release = "gpt2-small-res-jb", sae_id = "blocks.9.hook_resid_pre")
        sae_up_hook_point = "blocks.8.hook_resid_pre.hook_sae_acts_post"
        sae_target_hook_point = "blocks.8.attn.hook_z.hook_sae_acts_post"
        sae_down_hook_point = "blocks.9.hook_resid_pre.hook_sae_acts_post"

        saes = [sae_up, sae_target, sae_down]
        SAE_TARGET_FEATURE = 16513
        SAE_RANDOM_FEATURE = 4987

        # Make dict of local variables to return
        return locals()
    

class ComputeCache(ExperimentStage):
    """ Compute the cache for the SAEs. """

    def run(self, inputs: dict[str, Any]):

        model = inputs["model"]
        ioi_data = inputs["ioi_data"]
        saes = inputs["saes"]

        with model.saes(saes):
            clean_cache = get_sae_cache(model, ioi_data, input="clean")
            corrupt_cache = get_sae_cache(model, ioi_data, input="corrupt")

        return {
            "clean_cache": clean_cache,
            "corrupt_cache": corrupt_cache,
        }
    
class VisualizeFeatureActivation(ExperimentStage):
    """ Visualize the target feature activations. """
    
    def run(self, inputs: dict[str, Any]):

        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme()

        clean_cache = inputs["clean_cache"]
        SAE_TARGET_FEATURE = inputs["SAE_TARGET_FEATURE"]
        SAE_RANDOM_FEATURE = inputs["SAE_RANDOM_FEATURE"]
        sae_target_hook_point = inputs["sae_target_hook_point"]

        def plot_acts_summed(feature):
            clean_acts_summed = clean_cache[sae_target_hook_point][:, :, feature].sum(dim=1)
            clean_acts_np = clean_acts_summed.detach().cpu().numpy()

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_title("Sum of activations per example")
            ax.set_xlabel("Example index")
            ax.set_ylabel("Sum of activations")
            sns.barplot(clean_acts_np, orient = "h", ax=ax)
            return fig, ax

        fig_target_act, _ = plot_acts_summed(SAE_TARGET_FEATURE)
        fig_random_act, _ = plot_acts_summed(SAE_RANDOM_FEATURE)        

        return {
            "fig_sae_target_feature_activation": fig_target_act,
            "fig_sae_random_feature_activation": fig_random_act,
        }
    
def save_all_plots(state: ExperimentState, save_dir: pathlib.Path):

    save_dir.mkdir(exist_ok=True)

    for name, spec in state.all_artifacts().items():
        if spec.name.startswith("fig"):
            fig = state.read_artifact(spec)
            assert isinstance(fig, plt.Figure)
            fig.savefig(save_dir / f"{spec.name}.pdf")

if __name__ == "__main__":

    spec = ExperimentSpec(
        "gpt2-small_att8.16513",
        [
            LoadDataAndModels(),
            ComputeCache(),
            VisualizeFeatureActivation()
        ],
    )
    state = InMemoryExperimentState()
    runner = ExperimentRunner(spec, state)
    runner.run()

    save_all_plots(state, pathlib.Path("plots"))