from feature_lens.data.text_handler import TextHandler


def test_text_handler(solu1l_model):
    text = "The quick brown fox jumps over the lazy dog."
    handler = TextHandler(solu1l_model, text)
    logits = handler.get_logits(solu1l_model)
    assert len(logits.shape) == 3
    metric = handler.get_metric(logits)
    assert metric.shape == ()


def test_text_handler_with_sae(solu1l_model, solu1l_sae):
    text = "The quick brown fox jumps over the lazy dog."
    handler = TextHandler(solu1l_model, text)
    with solu1l_model.saes([solu1l_sae]):
        logits = handler.get_logits(solu1l_model)
        assert len(logits.shape) == 3
        metric = handler.get_metric(logits)
        assert metric.shape == ()
