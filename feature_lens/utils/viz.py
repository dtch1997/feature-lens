"""Visualization utilities"""

import numpy as np
import torch
import uuid
import pandas as pd
import json

from typing import Union


PythonProperty = Union[
    dict,
    list,
    None,
    np.ndarray,
    torch.Tensor,
    bool,
    float,
    int,
    str,
]

JavaScriptProperty = Union[
    bool,  # Boolean
    dict,  # Object
    float,  # Number
    int,  # Number
    list,  # Array
    str,  # String
    None,  # Undefined
]

def make_pretty_print_html(df: pd.DataFrame) -> str:
    """Make the HTML to pretty print a dataframe in an IPython notebook"""
    return (
        df.to_html().replace("\\n", "<br>")  # Visualize newlines nicely
        # TODO: Figure out how to align text left
    )

def convert_prop_type(prop: PythonProperty) -> JavaScriptProperty:
    """Convert property to a JavaScript supported type

    For example, JavaScript doesn't support numpy arrays or torch tensors, so we
    convert them to lists (which JavaScript will recognize as an array).

    Args:
        prop: The property to convert

    Returns:
        Union[str, int, float, bool]: JavaScript safe property
    """
    if isinstance(prop, torch.Tensor):
        return prop.tolist()
    if isinstance(prop, np.ndarray):
        return prop.tolist()

    return prop

def convert_props(props: dict[str, PythonProperty]) -> str:
    """Convert a set of properties to a JavaScript safe string

    Args:
        props: The properties to convert

    Returns:
        str: JavaScript safe properties
    """
    props_with_values = {k: v for k, v in props.items() if v is not None}

    return json.dumps({k: convert_prop_type(v) for k, v in props_with_values.items()})

def colored_tokens_html(
    tokens: list[str], 
    values: list[float] | np.ndarray | torch.Tensor,
) -> str:
    """Create a HTML string for a list of tokens"""

    id = str(uuid.uuid4())[:13]
    react_element_name = "ColoredTokens"
    circuitsvis_version = "1.43.2"
    props = convert_props({"tokens": tokens, "values": values})
    html = (
        f"<div id='{id}' style='margin: 15px 0;'></div>\n"
        + "<script crossorigin type='module'>\n"
        + f"import {{ render, {react_element_name} }} from 'https://unpkg.com/circuitsvis@{circuitsvis_version}/dist/cdn/esm.js';\n"
        + f"""render(
            "{id}",
            {react_element_name},
            {props}
        )\n"""
        + "</script>"
    )
    return html