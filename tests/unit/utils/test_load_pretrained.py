from feature_lens.utils.load_pretrained import load_model, load_sae, load_transcoder


def test_load_model():
    model = load_model()
    assert model is not None


def test_load_sae():
    sae = load_sae()
    assert sae is not None


def test_load_transcoder():
    transcoder = load_transcoder()
    assert transcoder is not None
