"""200 Hz model wiring: padding-mask scale and pre-train -> fine-tune transfer.

Needs torch, so it runs on the training box only:

    python -m pytest data/tests/test_200hz_model_wiring.py -q
"""

import os
import sys

import pytest

torch = pytest.importorskip('torch')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from models.conformer import ConformerEncoder                    # noqa: E402
from models.multiview import RegionalConformerEncoder             # noqa: E402
from models.pretrain_model import ConformerPreTrainModel          # noqa: E402

# The SNN best cell (d512 / kernel_count 256, 1T1S) with a 4-layer local transformer.
GEOM = dict(d_model=64, n_filters=16, n_spatial_filters=8, temporal_kernel=10,
            pool_stride=40, n_heads=4, n_transformer_layers=2)


def test_full_length_200hz_sentence_keeps_every_token():
    enc = RegionalConformerEncoder(n_channels=6, tokens_per_view=32, n_cls_tokens=4,
                                   **GEOM).eval()
    x = torch.randn(2, 6, 2000)
    eeg_len = torch.tensor([2000, 1000])
    _, _, valid = enc(x, eeg_len=eeg_len, raw_max_len=2000)
    assert valid[0].sum().item() == 32
    assert valid[1].sum().item() == 16
    # The old call (no raw_max_len) divided by 5000 and dropped most of the sentence.
    _, _, old = enc(x, eeg_len=eeg_len)
    assert old[0].sum().item() == 13


def test_translator_passes_raw_eeg_max_len_to_the_region_encoders(monkeypatch):
    from models import multiview
    seen = []
    orig = multiview.RegionalConformerEncoder.forward

    def spy(self, x, eeg_len=None, raw_max_len=5000):
        seen.append(raw_max_len)
        return orig(self, x, eeg_len=eeg_len, raw_max_len=raw_max_len)

    monkeypatch.setattr(multiview.RegionalConformerEncoder, 'forward', spy)

    class _Bart(torch.nn.Module):
        pass

    model = multiview.MultiViewConformerTranslator(
        _Bart(), tokens_per_view=32, n_cls_per_view=4, n_encoder_layers=2,
        n_global_layers=1, decoder_embedding_size=32, raw_eeg_max_len=2000,
        d_model=64, n_filters=16, n_spatial_filters=8, temporal_kernel=10,
        pool_stride=40, n_heads=4).eval()
    views = {r: torch.randn(1, ch, 2000)
             for r, ch in multiview.BRAIN_REGION_CHANNEL_COUNT.items()}
    model.encode(views, raw_eeg_lens=torch.tensor([2000]))
    assert seen and set(seen) == {2000}


def test_default_raw_eeg_max_len_is_unchanged():
    from models import multiview
    import inspect
    sig = inspect.signature(multiview.MultiViewConformerTranslator.__init__)
    assert sig.parameters['raw_eeg_max_len'].default == 5000


def test_pretrain_spatial_filters_default_keeps_old_shapes():
    enc = ConformerEncoder(n_channels=6, d_model=64, n_filters=16, temporal_kernel=10,
                           pool_stride=40, n_heads=4, n_transformer_layers=1)
    assert enc.projection.in_features == 16
    assert enc.spatial_conv[0].out_channels == 16


def test_pretrained_encoder_transfers_projection_with_narrowing_stem(tmp_path):
    from models import multiview

    pre = ConformerPreTrainModel(n_channels=105, target_T=2000, **GEOM)
    x = torch.randn(2, 105, 2000)
    assert pre(x).shape == (2, 105, 2000)
    path = tmp_path / 'encoder_best.pt'
    torch.save(pre.encoder.state_dict(), path)

    class _Bart(torch.nn.Module):
        pass

    ft = multiview.MultiViewConformerTranslator(
        _Bart(), tokens_per_view=32, n_cls_per_view=4,
        n_encoder_layers=GEOM['n_transformer_layers'], n_global_layers=1,
        decoder_embedding_size=32, raw_eeg_max_len=2000,
        **{k: v for k, v in GEOM.items() if k != 'n_transformer_layers'})
    ft.load_pretrained_encoder(str(path))
    region = next(iter(ft.view_encoders.values()))
    assert torch.equal(region.projection.weight, pre.encoder.projection.weight)
    assert torch.equal(region.temporal_conv[0].weight, pre.encoder.temporal_conv[0].weight)
    assert torch.equal(region.transformer.layers[-1].linear2.weight,
                       pre.encoder.transformer.layers[-1].linear2.weight)
