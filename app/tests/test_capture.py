"""Tests for the opt-in diagnostic capture of below-threshold compares.

The engine is mocked (conftest.fake_embeddings), so we drive run_comparison
directly with controlled embeddings to force a low/high confidence and assert
that both images are captured (as non-empty JPEG base64) only when enabled AND
below the configured threshold.
"""
import base64

import numpy as np

import app.compare as compare
from app.config import get_settings
from app.image_loader import ImageSlot


def _slots(jpeg_bytes):
    return (
        ImageSlot("image1", image_file=jpeg_bytes),
        ImageSlot("image2", image_file=jpeg_bytes),
    )


def test_captures_pair_when_below_threshold(jpeg_bytes, fake_embeddings, monkeypatch):
    # Orthogonal vectors -> cosine similarity 0 -> confidence 0 (< threshold).
    v1 = np.zeros(512); v1[0] = 1.0
    v2 = np.zeros(512); v2[1] = 1.0
    fake_embeddings(v1, v2)

    settings = get_settings()
    monkeypatch.setattr(settings, "save_failed_below", 50.0)
    s1, s2 = _slots(jpeg_bytes)

    confidence, captures = compare.run_comparison(s1, s2, settings)

    assert confidence < 50.0
    assert captures is not None
    # Both entries are non-empty and valid base64 that decodes to JPEG bytes.
    for blob in captures:
        assert blob
        assert base64.b64decode(blob)[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_no_capture_when_above_threshold(jpeg_bytes, fake_embeddings, monkeypatch):
    # Identical vectors -> similarity 1 -> confidence 100 (>= threshold).
    fake_embeddings(np.ones(512), np.ones(512))

    settings = get_settings()
    monkeypatch.setattr(settings, "save_failed_below", 50.0)
    s1, s2 = _slots(jpeg_bytes)

    confidence, captures = compare.run_comparison(s1, s2, settings)

    assert confidence >= 50.0
    assert captures is None


def test_no_capture_when_feature_disabled(jpeg_bytes, fake_embeddings, monkeypatch):
    # Below any real threshold, but feature off (0) -> nothing captured.
    v1 = np.zeros(512); v1[0] = 1.0
    v2 = np.zeros(512); v2[1] = 1.0
    fake_embeddings(v1, v2)

    settings = get_settings()
    monkeypatch.setattr(settings, "save_failed_below", 0.0)
    s1, s2 = _slots(jpeg_bytes)

    confidence, captures = compare.run_comparison(s1, s2, settings)

    assert captures is None
