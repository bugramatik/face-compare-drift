"""Pipeline orchestration for a single comparison (PRD §4 steps 3-5).

This is the blocking, CPU-bound core. The FastAPI route offloads it to a
threadpool so a worker's event loop isn't blocked during inference.
"""
from __future__ import annotations

import base64
from typing import Optional

import cv2

from .config import Settings
from .face_engine import (
    cosine_similarity,
    represent,
    similarity_to_confidence,
)
from .image_loader import ImageSlot, resolve_slot


def _encode_jpeg_b64(img) -> str:
    """JPEG-encode a decoded BGR image and return base64 text (for diagnostic
    capture of below-threshold attempts). Returns '' if encoding fails."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def thresholds(settings: Settings) -> dict[str, float]:
    """Face++-shaped threshold tiers (structurally compatible, recalibrated)."""
    return {
        "1e-3": settings.threshold_1e_3,
        "1e-4": settings.threshold_1e_4,
        "1e-5": settings.threshold_1e_5,
    }


def run_comparison(
    slot1: ImageSlot, slot2: ImageSlot, settings: Settings
) -> tuple[float, Optional[tuple[str, str]]]:
    """Resolve both slots, embed each face, return (confidence, captures).

    `captures` is None unless diagnostic capture is enabled
    (settings.save_failed_below > 0) AND the confidence falls below it, in which
    case it is (image1_jpeg_b64, image2_jpeg_b64) so the caller can persist the
    failing pair. Otherwise the decoded images and embeddings stay in memory and
    are discarded when this function returns (PRD §2 Data Handling).

    Raises CompareError (NO_FACE_DETECTED / INVALID_IMAGE / MISSING_ARGUMENTS /
    IMAGE_*) which the caller maps to a Face++-shaped error response.
    """
    img1 = resolve_slot(slot1, settings)
    emb1 = represent(img1, settings, slot1.name)

    img2 = resolve_slot(slot2, settings)
    emb2 = represent(img2, settings, slot2.name)

    similarity = cosine_similarity(emb1, emb2)
    confidence = similarity_to_confidence(similarity, settings)

    captures: Optional[tuple[str, str]] = None
    if settings.save_failed_below and confidence < settings.save_failed_below:
        captures = (_encode_jpeg_b64(img1), _encode_jpeg_b64(img2))
    return confidence, captures
