# feature-lens
![Github Actions](https://github.com/dtch1997/feature-lens/actions/workflows/tests.yaml/badge.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)

A research engineering toolkit for understanding how SAE features relate to each other, and to upstream / downstream components. 

For projects supported by `feature-lens`, see `projects`

# Quickstart

```bash
git clone https://github.com/dtch1997/feature-lens
cd feature-lens
pip install -e .
```

# Development

Refer to [Setup](docs/setup.md) for how to set up development environment.

# Implementation Details

Techniques for finding relevant feature associations:
- Activation patching (employed in [Causal Graphs](https://www.lesswrong.com/posts/uNGAjA8wCNDZHJxu8/causal-graphs-of-gpt-2-small-s-residual-stream))
- (Total) attribution patching (employed in [Sparse Feature Circuits](https://arxiv.org/abs/2403.19647))
- Direct attribution patching (employed in [MLP Transcoders](https://arxiv.org/abs/2406.11944), [Attention-out SAEs](https://www.lesswrong.com/posts/FSTRedtjuHa4Gfdbr/attention-saes-scale-to-gpt-2-small))

Note: "Total" attribution patching calculates the full effect of one feature on another via gradient backpropagation. "Direct" attribution patching estimates only the direct effect, which can be calculated analytically using matrix multiplication. 

Tools which will be implemented. 
- [ ] **SAE features.** We will use SAEs from SAE-Lens, which are annotated and have Neuronpedia visualizations.
- [ ] **Feature dashboards.**  By visualizing the "functional connectome" of SAE features, we may obtain novel insight about what an SAE feature is doing. For each target SAE feature, we can create a dashboard of all relevant upstream / downstream features.

Ideas under consideration.
- [ ] **Feature clustering**. Features with similar upstream and downstream features could be hypothesized to be performing a similar role. Clustering features based on their connections may reveal novel insight about the general types of "functional role" played by SAE features. 
- [ ] **Linear direction features.** Steering vectors are like an SAE feature because they are added to the residual activations. 
