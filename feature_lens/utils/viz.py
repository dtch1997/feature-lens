"""Visualization utilities"""


def make_pretty_print_html(df) -> str:
    """Make the HTML to pretty print a dataframe in an IPython notebook"""
    return (
        df.to_html().replace("\\n", "<br>")  # Visualize newlines nicely
        # TODO: Figure out how to align text left
    )
