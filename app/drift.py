"""Confidence drift detection for the compare service.

A compare service can keep answering 200s while its confidence scores quietly
slide: a model swap, a detector backend change or a new image pipeline moves the
whole score distribution, and callers that used to pass start failing. Latency
and error dashboards never show that. This module compares the recent score
distribution against a frozen baseline and reports the population stability
index (PSI) plus a per-bin shift, so the move is visible before callers notice.

Scores are bucketed into ten fixed bins over the 0-100 confidence range. The
baseline is a snapshot of successful scores stored in SQLite; it is created on
first use and only replaced when someone explicitly re-baselines.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import aiosqlite

from . import db

logger = logging.getLogger("facecompare.drift")

# Ten fixed bins over the 0-100 confidence range. Fixed edges (not quantiles) so
a baseline and a later window are always compared on the same scale.
BIN_EDGES: tuple[float, ...] = tuple(float(i * 10) for i in range(11))
N_BINS: int = len(BIN_EDGES) - 1

# PSI above this is treated as a real shift. 0.1 is the usual rule of thumb for
# "noticeable", 0.25 for "large"; the service default sits at the first one.
DEFAULT_PSI_THRESHOLD: float = 0.1
DEFAULT_WINDOW: int = 1000

# PSI is undefined when a bin is empty on one side; a small floor keeps the
# logarithm finite without hiding a shift (an empty bin still contributes).
_EPS: float = 1e-4

_BASELINE_SCHEMA = """
CREATE TABLE IF NOT EXISTS drift_baseline (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    sample_size INTEGER NOT NULL,
    bins_json   TEXT    NOT NULL
);
"""


# --------------------------------------------------------------------------- #
# Binning and PSI                                                             #
# --------------------------------------------------------------------------- #
def bin_index(score: float) -> int:
    """Return the 0-based bin a confidence score falls in.

    Scores are clamped to [0, 100] so a calibration change that pushes a value
    slightly outside the range still lands in the first or last bin instead of
    raising.
    """
    value = min(max(float(score), BIN_EDGES[0]), BIN_EDGES[-1])
    for i in range(N_BINS):
        if value < BIN_EDGES[i + 1]:
            return i
    return N_BINS - 1


def histogram(scores: list[float]) -> list[float]:
    """Return the share of `scores` in each of the ten bins (sums to 1.0).

    An empty list returns all zeros; callers check the sample size first.
    """
    counts = [0] * N_BINS
    for score in scores:
        counts[bin_index(score)] += 1
    total = len(scores)
    if total == 0:
        return [0.0] * N_BINS
    return [c / total for c in counts]


def population_stability_index(
    baseline: list[float], current: list[float]
) -> float:
    """PSI between two ten-bin distributions.

    PSI = sum over bins of (current - baseline) * ln(current / baseline).
    Both inputs are shares, so they should each sum to about 1.0.
    """
    if len(baseline) != N_BINS or len(current) != N_BINS:
        raise ValueError(f"expected {N_BINS} bins on each side")
    psi = 0.0
    for b, c in zip(baseline, current):
        b_adj = max(b, _EPS)
        c_adj = max(c, _EPS)
        psi += (c_adj - b_adj) * math.log(c_adj / b_adj)
    return psi


def per_bin_shift(
    baseline: list[float], current: list[float]
) -> list[dict[str, float]]:
    """Per-bin baseline share, current share and the difference in points."""
    rows: list[dict[str, float]] = []
    for i in range(N_BINS):
        rows.append(
            {
                "bin": i,
                "lower": BIN_EDGES[i],
                "upper": BIN_EDGES[i + 1],
                "baseline": round(baseline[i], 4),
                "current": round(current[i], 4),
                "delta": round(current[i] - baseline[i], 4),
            }
        )
    return rows


def verdict(psi: float, threshold: float) -> str:
    """Plain-language reading of a PSI value against the configured threshold."""
    if psi >= threshold * 2.5:
        return "large shift"
    if psi >= threshold:
        return "drift"
    return "stable"


# --------------------------------------------------------------------------- #
# Baseline storage                                                            #
# --------------------------------------------------------------------------- #
async def init_baseline_table() -> None:
    """Create the baseline table if it does not exist yet."""
    async with aiosqlite.connect(db.DB_PATH) as conn:
        await conn.executescript(_BASELINE_SCHEMA)
        await conn.commit()


async def load_baseline() -> Optional[dict]:
    """Return the newest stored baseline, or None when none was frozen yet."""
    import json

    async with aiosqlite.connect(db.DB_PATH) as conn:
        cur = await conn.execute(
            "SELECT created_at, sample_size, bins_json FROM drift_baseline "
            "ORDER BY id DESC LIMIT 1"
        )
        row = await cur.fetchone()
    if row is None:
        return None
    created_at, sample_size, bins_json = row
    return {
        "created_at": created_at,
        "sample_size": int(sample_size),
        "bins": json.loads(bins_json),
    }


async def freeze_baseline(window: int = DEFAULT_WINDOW) -> dict:
    """Snapshot the most recent `window` successful scores as the new baseline.

    Called on first start (when no baseline exists) and whenever the operator
    decides the current behaviour is the one to hold the service to.
    """
    import json

    scores = await db.recent_success_scores(window)
    if not scores:
        raise ValueError("no successful compares logged yet; nothing to freeze")
    bins = histogram(scores)
    async with aiosqlite.connect(db.DB_PATH) as conn:
        await conn.execute(
            "INSERT INTO drift_baseline (sample_size, bins_json) VALUES (?, ?)",
            (len(scores), json.dumps(bins)),
        )
        await conn.commit()
    logger.info("Froze drift baseline from %d scores.", len(scores))
    return {"sample_size": len(scores), "bins": bins}


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #
async def build_report(
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_PSI_THRESHOLD,
) -> dict:
    """Compare the last `window` successful scores with the frozen baseline.

    Returns a dict the route serialises as-is. When no baseline exists yet the
    current window is frozen as the baseline and the report says so, so the
    first call after a fresh install is not a false alarm.
    """
    baseline = await load_baseline()
    if baseline is None:
        frozen = await freeze_baseline(window)
        return {
            "status": "baseline_created",
            "psi": 0.0,
            "threshold": threshold,
            "verdict": "stable",
            "window": window,
            "samples": frozen["sample_size"],
            "baseline_samples": frozen["sample_size"],
            "baseline_created_at": None,
            "bins": per_bin_shift(frozen["bins"], frozen["bins"]),
        }

    scores = await db.recent_success_scores(window)
    current = histogram(scores)
    psi = population_stability_index(baseline["bins"], current)
    return {
        "status": "ok",
        "psi": round(psi, 4),
        "threshold": threshold,
        "verdict": verdict(psi, threshold),
        "window": window,
        "samples": len(scores),
        "baseline_samples": baseline["sample_size"],
        "baseline_created_at": baseline["created_at"],
        "bins": per_bin_shift(baseline["bins"], current),
    }
