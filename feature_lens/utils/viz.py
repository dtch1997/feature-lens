""" Utilities for working with dataframes """
import pandas as pd
from IPython.display import display, HTML

def flatten_dict(nested_dict, prefix="", delimiter="."):
    """ Flattens a nested dictionary """
    flattened = {}
    for key, value in nested_dict.items():
        new_key = f"{prefix}{key}"
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, f"{new_key}{delimiter}"))
        else:
            flattened[new_key] = value
    return flattened

def pretty_print(df):
    """ Pretty print a dataframe in an IPython notebook """
    return display( HTML( df.to_html().replace("\\n","<br>") ) )