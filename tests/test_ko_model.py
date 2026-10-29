import os
import sys

import pytest
import torch
from detectorscope.model.sae import JumpReLUSAE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from ko_model import encode, slice_sae  # noqa: E402


@pytest.mark.parametrize("pre_enc_bias", [False, True])
def test_sliced_encoder_matches_full_sae_on_kept_features(pre_enc_bias):
    torch.manual_seed(0)
    sae = JumpReLUSAE(16, 64)
    sae.use_pre_enc_bias = pre_enc_bias
    sae.b_enc.data.normal_()
    sae.b_dec.data.normal_()
    sae.log_threshold.data.uniform_(-2, 0.5)
    h = torch.randn(10, 16)
    local = torch.tensor([3, 17, 40, 63])
    expected = sae.encode(h.clone())[:, local]
    got = encode(h, slice_sae(sae, local))
    assert torch.allclose(got, expected, atol=1e-5)
    assert (got > 0).any() and (got == 0).any()  # the threshold actually bites
