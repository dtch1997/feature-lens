import torch
import uuid
from jaxtyping import Int
from feature_lens.cache import ActivationCache
from feature_lens.utils.viz import colored_tokens_html

def make_div(id: str):
    return f"<div id='{id}' style='margin: 15px 0;'></div>\n"

def make_activation_pattern_viz(
    cache: ActivationCache,
    tokens: list[list[str]],
    sae_hook_point: str,
    sae_features: list[int],
):
    """Make a visualization of the activation pattern for a given SAE hook point and features. """
    dashboard_html_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Visualization Data</title>
    </head>
    <body>
    """
    ids = {}

    # Make the HTML skeleton 
    for feature in sae_features:
        # Add a div for the feature
        dashboard_html_str += f"<div>Feature {feature}</div>\n"
        for example_idx in range(len(tokens)):
            id = str(uuid.uuid4())[:13]
            ids[(example_idx, feature)] = id
            dashboard_html_str += make_div(id)

    # Make the Javascript section
    dashboard_html_str += "<script crossorigin type='module'>\n"
    dashboard_html_str += f"import {{ render, ColoredTokens }} from 'https://unpkg.com/circuitsvis@1.43.2/dist/cdn/esm.js';\n"
    dashboard_html_str += "const visualizationData = ["

    for feature in sae_features:
        for example_idx in range(len(tokens)):
            ts = tokens[example_idx]
            acts = cache[sae_hook_point][example_idx, :, feature]
            id = ids[(example_idx, feature)]
            dashboard_html_str += "{id: " + f'"{id}"' + ", tokens: " + str(ts) + ", values: " + str(acts.tolist()) + "},\n"
    dashboard_html_str += "];\n"

    dashboard_html_str += """function renderAllVisualizations() {
        visualizationData.forEach(data => {
            render(
                data.id,
                ColoredTokens,
                { tokens: data.tokens, values: data.values }
            );
        });
    }
    
    renderAllVisualizations();
    </script>

    </body>
    </html>
    """

    return dashboard_html_str