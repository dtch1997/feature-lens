from .ioi import make_ioi

registry = {
    "ioi": make_ioi,
}


def make_dataset(name: str):
    return registry[name]()
