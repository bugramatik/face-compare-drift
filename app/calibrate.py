"""Offline calibration + threshold analysis for face-compare-drift.

Runs REAL historical attendance pairs through the SAME pipeline production uses
(detector, downscale, Facenet512, alignment, cosine similarity) so the numbers
transfer directly. It answers two questions the single-point CONFIDENCE_SCALE
guess cannot:

  A. THRESHOLD (FAR / FRR) -- given genuine pairs (a person's registered face vs
     a face that really attended) plus auto-generated impostor pairs (that
     person's attended face vs a DIFFERENT employee's registered face), where
     should the pass threshold sit to let genuine people through while keeping
     impostors out? This is what directly reduces failed attendance.

  B. FACE++ ALIGNMENT (optional) -- if you recorded Face++'s own confidence for a
     pair, fit similarity -> Face++ confidence so face-compare-drift's numbers line up with
     the vendor you're replacing. Produces CONFIDENCE_SCALE / CONFIDENCE_OFFSET.

INPUT: a manifest CSV, format-agnostic to however your 2 years of data is stored:

    employee_id,reference,attended,facepp_confidence
    E001,refs/E001.jpg,attended/E001_2024-01-15.jpg,91.7
    E001,refs/E001.jpg,attended/E001_2024-03-02.jpg,
    E002,refs/E002.jpg,attended/E002_2024-01-15.jpg,88.2

  - reference / attended: image paths, relative to --images ROOT (or absolute).
  - facepp_confidence: optional; leave blank if you don't have it. Section B is
    skipped when no rows carry it.

Recognition settings (MODEL_NAME, DETECTOR_BACKEND, MAX_IMAGE_DIM, ALIGN) are
read from the SAME environment as the service, so calibration matches live config.

Run it inside the api image (it has DeepFace + the model-weights volume):

    docker compose run --rm \
      -v "/host/path/to/calib_data:/calib" \
      -v "$(pwd)/app/calibrate.py:/app/app/calibrate.py:ro" \
      api python -m app.calibrate --manifest /calib/manifest.csv --images /calib

Nothing is written unless you pass --out; input images are never modified.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
from pathlib import Path, PurePosixPath
from typing import Optional

import cv2
import numpy as np

from .config import get_settings
from .face_engine import cosine_similarity, represent, warmup
from .image_loader import _maybe_downscale  # same downscale as production

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("facecompare.calibrate")

# Similarity floor for "these could plausibly be the same face". Purely for the
# printed FAR/FRR sweep; not a decision the tool makes.
_SWEEP_POINTS = 41


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
class Row:
    __slots__ = ("employee_id", "reference", "attended", "facepp")

    def __init__(self, employee_id: str, reference: str, attended: str,
                 facepp: Optional[float]) -> None:
        self.employee_id = employee_id
        self.reference = reference
        self.attended = attended
        self.facepp = facepp


def _strip_drive(rel: str) -> str:
    rel = rel.lstrip("/")
    first = rel.split("/", 1)[0]
    if ":" in first:  # strip a leading drive like "F:"
        rel = rel[len(first) + 1:].lstrip("/")
    return rel


def _resolve_path(raw: str, images_root: Path, special_dir: Optional[Path],
                  is_reference: bool, strip_prefix: Optional[str]) -> str:
    """Turn a raw DB path (Windows, backslashes, maybe absolute) into a path that
    resolves INSIDE the container.

    Three modes, in priority order:

    - --strip-prefix: ONE rule for both columns. Remove a shared Windows prefix
      (e.g. "F:\\DCT_DOCS") and keep the rest of the subtree under --images. Use
      this when reference and attended share a common root and you mount that
      root once. Backslashes are normalised.
    - --reference-dir / --attended-dir: references keep only their filename
      (flat folder); attended keep their relative subtree (drive stripped).
    - neither: the path is simply joined under --images.
    """
    norm = raw.replace("\\", "/").strip()
    if strip_prefix:
        sp = strip_prefix.replace("\\", "/").rstrip("/")
        if norm.lower().startswith(sp.lower()):
            rel = norm[len(sp):].lstrip("/")
        else:  # prefix didn't match this path -> just drop any drive letter
            rel = _strip_drive(norm)
        return str(images_root / rel)
    if special_dir is None:
        return str(images_root / norm)
    if is_reference:
        return str(special_dir / PurePosixPath(norm).name)
    return str(special_dir / _strip_drive(norm))


def _read_manifest(manifest: Path, images_root: Path,
                   reference_dir: Optional[Path], attended_dir: Optional[Path],
                   strip_prefix: Optional[str], no_header: bool) -> list[Row]:
    """Read the manifest. With --no-header the columns are positional:
    employee_id, reference, attended, [facepp_confidence]. utf-8-sig strips any
    BOM that SSMS/Excel prepends."""
    rows: list[Row] = []

    def _emit(i: int, emp: str, ref: str, att: str, fpp_raw: str) -> None:
        emp, ref, att = emp.strip(), ref.strip(), att.strip()
        if not emp or not ref or not att:
            logger.warning("row %d: blank employee_id/reference/attended, skipped", i)
            return
        fpp_raw = (fpp_raw or "").strip()
        try:
            fpp = float(fpp_raw) if fpp_raw else None
        except ValueError:
            logger.warning("row %d: bad facepp_confidence %r, treated as blank", i, fpp_raw)
            fpp = None
        rows.append(Row(
            emp,
            _resolve_path(ref, images_root, reference_dir, True, strip_prefix),
            _resolve_path(att, images_root, attended_dir, False, strip_prefix),
            fpp,
        ))

    with manifest.open(newline="", encoding="utf-8-sig") as fh:
        if no_header:
            for i, cols in enumerate(csv.reader(fh), start=1):
                if not cols or all(not c.strip() for c in cols):
                    continue
                if len(cols) < 3:
                    logger.warning("row %d: fewer than 3 columns, skipped", i)
                    continue
                _emit(i, cols[0], cols[1], cols[2], cols[3] if len(cols) > 3 else "")
        else:
            reader = csv.DictReader(fh)
            required = {"employee_id", "reference", "attended"}
            missing = required - set(h.strip() for h in (reader.fieldnames or []))
            if missing:
                sys.exit(f"manifest is missing required column(s): {sorted(missing)}. "
                         f"If the export has no header row, pass --no-header.")
            for i, raw in enumerate(reader, start=2):  # header is line 1
                _emit(i, raw.get("employee_id") or "", raw.get("reference") or "",
                      raw.get("attended") or "", raw.get("facepp_confidence") or "")
    return rows


# --------------------------------------------------------------------------- #
# Embedding (production-identical preprocessing)
# --------------------------------------------------------------------------- #
def _embed_file(path: str, settings) -> Optional[np.ndarray]:
    """Decode -> downscale -> represent, exactly like the live request path.

    Returns None (and logs) if the file can't be read or no face is detected,
    so one bad image never aborts a multi-thousand-pair run.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        logger.warning("cannot read image: %s", path)
        return None
    img = _maybe_downscale(img, settings)
    try:
        return represent(img, settings, "calib")
    except Exception as exc:  # NO_FACE_DETECTED / INVALID_IMAGE etc.
        logger.warning("no usable face in %s (%s)", path, type(exc).__name__)
        return None


def _build_embedding_cache(rows: list[Row], settings) -> dict[str, np.ndarray]:
    """Embed every unique image path once. References are reused across many
    rows, so caching avoids re-running the detector on the same file."""
    unique = sorted({p for r in rows for p in (r.reference, r.attended)})
    logger.info("Embedding %d unique images (detector=%s, model=%s)...",
                len(unique), settings.detector_backend, settings.model_name)
    cache: dict[str, np.ndarray] = {}
    for n, path in enumerate(unique, start=1):
        emb = _embed_file(path, settings)
        if emb is not None:
            cache[path] = emb
        if n % 25 == 0 or n == len(unique):
            logger.info("  %d/%d embedded (%d with a detected face)",
                        n, len(unique), len(cache))
    return cache


# --------------------------------------------------------------------------- #
# Pair scoring
# --------------------------------------------------------------------------- #
def _genuine_scores(rows: list[Row], cache: dict[str, np.ndarray]) -> list[tuple[Row, float]]:
    out: list[tuple[Row, float]] = []
    for r in rows:
        a, b = cache.get(r.reference), cache.get(r.attended)
        if a is not None and b is not None:
            out.append((r, cosine_similarity(a, b)))
    return out


def _impostor_scores(rows: list[Row], cache: dict[str, np.ndarray],
                     per_attended: int, rng: random.Random) -> list[float]:
    """Pair each attended face against `per_attended` OTHER employees' references."""
    refs_by_emp: dict[str, str] = {}
    for r in rows:
        refs_by_emp.setdefault(r.employee_id, r.reference)
    all_emps = list(refs_by_emp)
    if len(all_emps) < 2:
        return []

    scores: list[float] = []
    for r in rows:
        att = cache.get(r.attended)
        if att is None:
            continue
        others = [e for e in all_emps if e != r.employee_id]
        rng.shuffle(others)
        picked = 0
        for e in others:
            ref = cache.get(refs_by_emp[e])
            if ref is None:
                continue
            scores.append(cosine_similarity(att, ref))
            picked += 1
            if picked >= per_attended:
                break
    return scores


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else float("nan")


def _describe(name: str, a: np.ndarray) -> None:
    if not a.size:
        logger.info("  %-9s (none)", name)
        return
    logger.info("  %-9s n=%-5d  min=%.4f  p5=%.4f  p50=%.4f  mean=%.4f  p95=%.4f  max=%.4f",
                name, a.size, a.min(), _pct(a, 5), _pct(a, 50), a.mean(), _pct(a, 95), a.max())


def _far_frr_table(genuine: np.ndarray, impostor: np.ndarray, settings) -> None:
    """Print FRR (genuine rejected) and FAR (impostor accepted) across a sweep of
    similarity thresholds, plus recommended operating points."""
    if not genuine.size:
        logger.info("No genuine pairs scored -- cannot recommend a threshold.")
        return

    lo = min(genuine.min(), impostor.min() if impostor.size else genuine.min())
    hi = max(genuine.max(), impostor.max() if impostor.size else genuine.max())
    ts = np.linspace(lo, hi, _SWEEP_POINTS)

    scale, offset = settings.confidence_scale, settings.confidence_offset

    logger.info("")
    logger.info("FAR / FRR sweep  (confidence = similarity*%.3f + %.3f, current .env scale)", scale, offset)
    logger.info("  sim     conf    FRR(genuine rejected)   FAR(impostor accepted)")
    for t in ts:
        frr = float(np.mean(genuine < t))
        far = float(np.mean(impostor >= t)) if impostor.size else float("nan")
        conf = t * scale + offset
        logger.info("  %.3f  %6.2f        %6.1f%%                %6.1f%%",
                    t, conf, 100 * frr, 100 * far)

    # Recommended operating points.
    logger.info("")
    logger.info("Recommended operating points (similarity | confidence@current-scale):")

    # 1) Youden's J: maximize (1-FRR) - FAR over observed thresholds.
    if impostor.size:
        cand = np.unique(np.concatenate([genuine, impostor]))
        j = [(t, (1 - np.mean(genuine < t)) - np.mean(impostor >= t)) for t in cand]
        t_best = max(j, key=lambda x: x[1])[0]
        logger.info("  balanced (Youden): sim>=%.4f  conf>=%.2f  ->  FRR=%.1f%%  FAR=%.1f%%",
                    t_best, t_best * scale + offset,
                    100 * np.mean(genuine < t_best), 100 * np.mean(impostor >= t_best))

        # 2) Strict: just above the highest impostor -> ~0 observed false accepts.
        t_strict = float(np.nextafter(impostor.max(), np.inf))
        logger.info("  strict (0 observed FAR): sim>=%.4f  conf>=%.2f  ->  FRR=%.1f%%",
                    t_strict, t_strict * scale + offset, 100 * np.mean(genuine < t_strict))

    # 3) Lenient: pass 95% of genuine (5% FRR) -- shows the FAR cost of being lax.
    t_p5 = _pct(genuine, 5)
    far_at = 100 * np.mean(impostor >= t_p5) if impostor.size else float("nan")
    logger.info("  lenient (5%% FRR): sim>=%.4f  conf>=%.2f  ->  FAR=%.1f%%",
                t_p5, t_p5 * scale + offset, far_at)


def _facepp_fit(genuine_pairs: list[tuple[Row, float]]) -> None:
    """Fit similarity -> Face++ confidence (linear least squares) for rows that
    carry a facepp_confidence. Prints CONFIDENCE_SCALE / CONFIDENCE_OFFSET."""
    pts = [(sim, r.facepp) for r, sim in genuine_pairs if r.facepp is not None]
    if len(pts) < 3:
        logger.info("")
        logger.info("Section B (Face++ alignment): skipped -- need >=3 rows with "
                    "facepp_confidence (found %d).", len(pts))
        return
    sims = np.array([p[0] for p in pts], dtype=np.float64)
    fpp = np.array([p[1] for p in pts], dtype=np.float64)

    slope, intercept = np.polyfit(sims, fpp, 1)
    pred = slope * sims + intercept
    ss_res = float(np.sum((fpp - pred) ** 2))
    ss_tot = float(np.sum((fpp - fpp.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

    logger.info("")
    logger.info("Section B -- Face++ alignment  (fit on %d labelled pairs)", len(pts))
    logger.info("  best-fit: Face++ confidence  ~=  similarity * %.3f + %.3f   (R^2=%.3f)",
                slope, intercept, r2)
    logger.info("  => set in .env:")
    logger.info("       CONFIDENCE_SCALE=%.3f", slope)
    logger.info("       CONFIDENCE_OFFSET=%.3f", intercept)
    if r2 < 0.5:
        logger.info("  NOTE: low R^2 -- similarity and Face++ don't track linearly on this")
        logger.info("        set; the affine fit is rough. A logistic fit would be better.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="face-compare-drift calibration / threshold analysis")
    ap.add_argument("--manifest", required=True, type=Path, help="CSV manifest path")
    ap.add_argument("--images", default=Path("."), type=Path,
                    help="root dir that manifest paths are relative to (default: .)")
    ap.add_argument("--reference-dir", type=Path, default=None,
                    help="container dir where reference images are mounted; when set, "
                         "the manifest's reference column is treated as a raw DB path and "
                         "only its filename is joined here (handles F:\\... absolute paths)")
    ap.add_argument("--attended-dir", type=Path, default=None,
                    help="container dir where the attended image tree is mounted; when set, "
                         "the manifest's attended column keeps its relative subtree "
                         "(2026/7/23/...) under this dir, backslashes normalised")
    ap.add_argument("--strip-prefix", default=None,
                    help="shared Windows prefix to remove from BOTH path columns "
                         r"(e.g. 'F:\DCT_DOCS'); the remaining subtree is joined under "
                         "--images. Use when reference & attended share one root.")
    ap.add_argument("--no-header", action="store_true",
                    help="manifest has no header row; columns are positional: "
                         "employee_id, reference, attended, [facepp_confidence]")
    ap.add_argument("--impostors-per-attended", type=int, default=5,
                    help="how many other-employee references to pair each attended "
                         "face against (default: 5)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only use the first N manifest rows (0 = all; use for a smoke test)")
    ap.add_argument("--seed", type=int, default=1234, help="RNG seed for impostor sampling")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional CSV to write every genuine pair score to")
    args = ap.parse_args(argv)

    if not args.manifest.is_file():
        sys.exit(f"manifest not found: {args.manifest}")

    settings = get_settings()
    rows = _read_manifest(args.manifest, args.images, args.reference_dir,
                          args.attended_dir, args.strip_prefix, args.no_header)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        sys.exit("no usable rows in manifest")
    logger.info("Loaded %d rows across %d employees.",
                len(rows), len({r.employee_id for r in rows}))

    warmup(settings)
    cache = _build_embedding_cache(rows, settings)

    genuine_pairs = _genuine_scores(rows, cache)
    genuine = np.array([s for _, s in genuine_pairs], dtype=np.float64)
    rng = random.Random(args.seed)
    impostor = np.array(
        _impostor_scores(rows, cache, args.impostors_per_attended, rng), dtype=np.float64
    )

    logger.info("")
    logger.info("Scored %d genuine pairs and %d impostor pairs.", genuine.size, impostor.size)
    logger.info("Cosine-similarity distributions:")
    _describe("genuine", genuine)
    _describe("impostor", impostor)

    if genuine.size and impostor.size:
        overlap = _pct(genuine, 5) <= _pct(impostor, 95)
        logger.info("  separation: genuine p5=%.4f vs impostor p95=%.4f  (%s)",
                    _pct(genuine, 5), _pct(impostor, 95),
                    "OVERLAP - some impostors score like genuine" if overlap
                    else "clean gap")

    _far_frr_table(genuine, impostor, settings)   # Section A
    _facepp_fit(genuine_pairs)                     # Section B

    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["employee_id", "reference", "attended", "similarity", "facepp_confidence"])
            for r, sim in genuine_pairs:
                w.writerow([r.employee_id, r.reference, r.attended, f"{sim:.5f}",
                            "" if r.facepp is None else r.facepp])
        logger.info("")
        logger.info("Wrote per-pair genuine scores to %s", args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
