import pytest

from feature_lens.utils.load_pretrained import load_model

@pytest.fixture
def solu1l_model():
    return load_model("solu-1l")

