import pytest
import torch

from projects.leap_sfc.leap.types import Model, SAE, Transcoder, Head, Feature, Node
from projects.leap_sfc.leap.model_handler import ModelHandler

@pytest.fixture()
def model_handler(model: Model, sae: SAE, transcoder: Transcoder):
    return ModelHandler(
        model, 
        {sae.cfg.hook_name: sae},
        {transcoder.cfg.hook_name: transcoder},
    )

def test_model_handler_get_n_features_at_head(model_handler: ModelHandler, expansion_factor: int):
    assert model_handler.n_layers == 1
    assert model_handler.get_n_features_at_head(Head(0, "mlp")) == expansion_factor * model_handler.model.cfg.d_model
    assert model_handler.get_n_features_at_head(Head(0, "att")) == expansion_factor * model_handler.model.cfg.d_model

def test_model_handler_get_encoder_weight(model_handler: ModelHandler):
    assert model_handler.get_encoder_weight(Feature(0, "mlp", 0)).shape == (model_handler.model.cfg.d_model,)
    assert model_handler.get_encoder_weight(Feature(0, "att", 0)).shape == (model_handler.model.cfg.d_model,)
    # test the weight matches
    sae = next(iter(model_handler.saes.values()))
    assert torch.allclose(model_handler.get_encoder_weight(Feature(0, "att", 0)), sae.W_enc[:, 0])
    transcoder = next(iter(model_handler.transcoders.values()))
    assert torch.allclose(model_handler.get_encoder_weight(Feature(0, "mlp", 0)), transcoder.W_enc[:, 0])

def test_model_handler_get_decoder_weight(model_handler: ModelHandler):
    assert model_handler.get_decoder_weight(Feature(0, "mlp", 0)).shape == (model_handler.model.cfg.d_model,)
    assert model_handler.get_decoder_weight(Feature(0, "att", 0)).shape == (model_handler.model.cfg.d_model,)

    # test the weight matches
    sae = next(iter(model_handler.saes.values()))
    assert torch.allclose(model_handler.get_decoder_weight(Feature(0, "att", 0)), sae.W_dec[0])
    transcoder = next(iter(model_handler.transcoders.values()))
    assert torch.allclose(model_handler.get_decoder_weight(Feature(0, "mlp", 0)), transcoder.W_dec[0]) 