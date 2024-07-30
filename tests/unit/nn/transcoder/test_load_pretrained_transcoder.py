import torch

# import pytest
from feature_lens.nn.transcoder import Transcoder, load_pretrained


def test_load_pretrained():
    transcoders = load_pretrained()
    assert len(transcoders) == 12

    # test the transcoder
    transcoder: Transcoder = list(transcoders.values())[0]
    d_in = transcoder.d_in
    d_tr = transcoder.d_sae
    seq_len = 32
    tr_in = torch.zeros(1, seq_len, d_in).to(transcoder.device)
    tr_out, tr_hid = transcoder(tr_in)[:2]

    assert tr_out.shape == tr_in.shape
    assert tr_hid.shape[-1] == d_tr
