"""Utilities for working with various data types."""


def flatten_dict(nested_dict, prefix="", delimiter="."):
    """Flattens a nested dictionary"""
    flattened = {}
    for key, value in nested_dict.items():
        new_key = f"{prefix}{key}"
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, f"{new_key}{delimiter}"))
        else:
            flattened[new_key] = value
    return flattened
