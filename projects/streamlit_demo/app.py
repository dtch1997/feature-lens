# type: ignore

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import altair as alt
import plotly.express as px
import torch

from dataclasses import dataclass
from feature_lens.utils import load_pretrained, memory
from feature_lens.utils.neuronpedia import get_feature_info
from feature_lens.data.text_handler import TextHandler
from feature_lens.attribute import (
    get_sae_cache_for_target_feature_as_metric,
    compute_attribution,
)


@dataclass(frozen=True)
class SAEMetadata:
    model_id: str
    release: str
    layer: int
    id: str
    hook_point: str

    @property
    def neuronpedia_layer(self):
        if self.release == "gpt2-small-res-jb":
            return f"{self.layer}-res-jb"
        elif self.release == "gpt2-small-hook-z-kk":
            return f"{self.layer}-att-kk"
        else:
            raise ValueError(f"Invalid release: {self.release}")


def get_resid_sae_metadata(layer: int) -> SAEMetadata:
    model_id = "gpt2-small"
    release = "gpt2-small-res-jb"
    id = f"blocks.{layer}.hook_resid_pre"
    act_hook_point = f"blocks.{layer}.hook_resid_pre.hook_sae_acts_post"
    return SAEMetadata(model_id, release, layer, id, act_hook_point)


def get_att_sae_metadata(layer: int) -> SAEMetadata:
    model_id = "gpt2-small"
    release = "gpt2-small-hook-z-kk"
    id = f"blocks.{layer}.hook_z"
    act_hook_point = f"blocks.{layer}.attn.hook_z.hook_sae_acts_post"
    return SAEMetadata(model_id, release, layer, id, act_hook_point)


def get_sae_metadata(release: str, layer: int) -> SAEMetadata:
    if release == "gpt2-small-res-jb":
        return get_resid_sae_metadata(layer)
    elif release == "gpt2-small-hook-z-kk":
        return get_att_sae_metadata(layer)
    else:
        raise ValueError(f"Invalid release: {release}")


def get_feature_iframe_src_html(metadata: SAEMetadata, feature: int):
    return f"https://neuronpedia.org/{metadata.model_id}/{metadata.neuronpedia_layer}/{feature}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"


def load_sae_from_metadata(metadata: SAEMetadata):
    return load_pretrained.load_sae(metadata.release, metadata.id)


ALL_RELEASES = ["gpt2-small-res-jb", "gpt2-small-hook-z-kk"]
ALL_LAYERS = [i for i in range(12)]

# Streamlit app


def query_user_for_sae_metadata():
    st.subheader("Upstream SAE")
    upstream_release = st.selectbox(
        "Select upstream SAE release",
        ALL_RELEASES,
        index=ALL_RELEASES.index("gpt2-small-res-jb"),
    )
    upstream_layer = st.selectbox(
        "Select upstream SAE layer", ALL_LAYERS, index=ALL_LAYERS.index(8)
    )
    upstream_metadata = get_sae_metadata(upstream_release, upstream_layer)

    st.subheader("Downstream SAE")
    downstream_release = st.selectbox(
        "Select downstream SAE release",
        ALL_RELEASES,
        index=ALL_RELEASES.index("gpt2-small-hook-z-kk"),
    )
    downstream_layer = st.selectbox(
        "Select downstream SAE layer", ALL_LAYERS, index=ALL_LAYERS.index(8)
    )
    downstream_metadata = get_sae_metadata(downstream_release, downstream_layer)

    return upstream_metadata, downstream_metadata


if __name__ == "__main__":
    # Set up the page
    st.set_page_config(
        page_title="Feature-Lens Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    alt.themes.enable("dark")

    with st.sidebar:
        st.title("🏂 Feature Lens")
        st.write("Welcome to the Feature Lens dashboard!")
        st.write(
            "This dashboard allows you to explore the attribution of a downstream feature in a model, "
            "using a pair of SAEs (Sparse Attributional Explainer)."
        )
        st.write(
            "To get started, please select the SAEs you would like to use for attribution."
        )

        upstream_metadata, downstream_metadata = query_user_for_sae_metadata()
        st.session_state["upstream_metadata"] = upstream_metadata
        st.session_state["downstream_metadata"] = downstream_metadata
        # Define the target feature
        n_features = 49512
        downstream_feature = st.number_input(
            "Select downstream feature",
            min_value=0,
            max_value=n_features - 1,
            value=16513,
        )
        downstream_feature = int(downstream_feature)

        custom_prompt = st.text_area(
            "Enter a custom prompt (leave blank to use max-activating example)",
            value="",
        )
        load_saes_button = st.button("Load SAEs")

        if not load_saes_button:
            st.stop()
        upstream_metadata = st.session_state["upstream_metadata"]
        downstream_metadata = st.session_state["downstream_metadata"]

        # Load model, SAEs
        model = load_pretrained.load_model("gpt2-small")
        upstream_sae = load_sae_from_metadata(upstream_metadata)
        downstream_sae = load_sae_from_metadata(downstream_metadata)

    # Get the Neuronpedia dashboard for target feature.
    st.subheader("Neuronpedia dashboard for downstream feature")
    feature_iframe = get_feature_iframe_src_html(
        downstream_metadata, downstream_feature
    )
    components.iframe(feature_iframe, height=500, width=1080)

    feature_info = get_feature_info(
        "gpt2-small", downstream_metadata.neuronpedia_layer, downstream_feature
    )
    max_activating_example = "".join(feature_info["activations"][0]["tokens"])
    if custom_prompt == "":
        custom_prompt = max_activating_example

    # Make data handler
    handler = TextHandler(model, custom_prompt)

    # Get the SAE cache for target feature
    with model.saes([upstream_sae, downstream_sae]):
        upstream_cache = get_sae_cache_for_target_feature_as_metric(
            model, handler, downstream_metadata.hook_point, downstream_feature
        )
    # Compute attribution
    attrib = compute_attribution(
        upstream_metadata.hook_point, clean_cache=upstream_cache, corrupt_cache=None
    )

    # Free memory
    del upstream_cache
    del upstream_sae
    del downstream_sae
    del model
    memory.clear_memory()

    # Display attribution histogram
    st.subheader("Upstream feature attribution histogram")
    attrib_summed = attrib.sum(dim=1).mean(dim=0)
    attrib_summed_np = attrib_summed.detach().cpu().numpy()
    attrib_df = pd.DataFrame(attrib_summed_np, columns=["attribution"])
    attrib_df["feature"] = attrib_df.index
    attrib_df["feature"] = attrib_df["feature"].astype(str)
    fig = px.histogram(attrib_df, x="attribution", title="Attribution scores")
    st.plotly_chart(fig)

    # Top 10 attribution barplot
    st.subheader("Top 10 upstream attribution scores")
    topk_scores, topk_features = torch.topk(attrib_summed, 10)
    topk_scores = topk_scores.detach().cpu().numpy()
    topk_features = topk_features.detach().cpu().numpy()
    topk_df = pd.DataFrame({"feature": topk_features, "attribution": topk_scores})
    topk_df["feature"] = topk_df["feature"].astype(str)
    fig = px.bar(
        topk_df,
        x="feature",
        y="attribution",
        title="Top 10 attribution scores",
        category_orders={"feature": topk_df["feature"].tolist()},
    )
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Attribution Score",
        xaxis={"type": "category"},
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)

    # Top 10 feature dashboards
    st.subheader("Neuronpedia dashboards for top 10 upstream features")
    for i, (upstream_feature, score) in enumerate(zip(topk_features, topk_scores)):
        feature_iframe = get_feature_iframe_src_html(
            upstream_metadata, upstream_feature
        )
        st.write(f"Upstream feature {upstream_feature} with score {score}")
        components.iframe(feature_iframe, height=300, width=540)
