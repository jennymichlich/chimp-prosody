"""
chimp_prosody.core
===================

Core, dependency-light logic for the pitch-intensity coordination pipeline:
voiced-segment detection, ToBI-style contour classification, and the
pitch-intensity coordination metrics (covariation, peak-timing offset).

This module intentionally has NO file I/O and NO dependency on parselmouth
directly in its function signatures (it operates on plain numpy arrays of
times/F0/intensity). That makes every function here trivially unit-testable
with synthetic data, and reusable regardless of how the audio was loaded or
chunked (see io_utils.py for chunked audio reading).

Previously, this logic was copy-pasted (via sed substitution) into ~15
per-subject scripts. That approach caused a real bug during this project:
one substitution silently failed to update a file path, and the resulting
script processed the wrong subject's data for one run. Consolidating into a
single, imported module removes that entire class of error and means a
bugfix here automatically applies everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Voiced-segment detection
# ---------------------------------------------------------------------------

def find_voiced_segments(
    times: np.ndarray,
    f0: np.ndarray,
    gap_thresh: float = 0.02,
    min_dur: float = 0.03,
) -> list[np.ndarray]:
    """Partition a pitch track into contiguous voiced segments.

    Parameters
    ----------
    times : array of frame timestamps (seconds), strictly increasing.
    f0 : array of F0 values in Hz, with unvoiced frames as NaN. Must be the
        same length as `times`.
    gap_thresh : maximum allowed gap (seconds) between consecutive voiced
        frames for them to be considered part of the same segment. Frames
        separated by more than this are split into separate segments.
    min_dur : minimum segment duration (seconds) to be retained. Shorter
        candidate segments are discarded as unreliable for contour
        classification.

    Returns
    -------
    List of integer index arrays into `times`/`f0`, one per retained voiced
    segment, in chronological order.
    """
    if len(times) != len(f0):
        raise ValueError(f"times and f0 must be the same length ({len(times)} vs {len(f0)})")

    valid = ~np.isnan(f0)
    segments: list[tuple[int, int]] = []
    start_idx: Optional[int] = None
    last_t: Optional[float] = None

    for i, v in enumerate(valid):
        if v:
            if start_idx is None:
                start_idx = i
            elif last_t is not None and times[i] - last_t > gap_thresh:
                segments.append((start_idx, i - 1))
                start_idx = i
            last_t = times[i]
    if start_idx is not None:
        tail_idxs = [j for j in range(start_idx, len(valid)) if valid[j]]
        if tail_idxs:
            segments.append((start_idx, tail_idxs[-1]))

    out: list[np.ndarray] = []
    for s, e in segments:
        idxs = np.array([i for i in range(s, e + 1) if valid[i]])
        if len(idxs) < 3:
            continue
        if times[idxs[-1]] - times[idxs[0]] < min_dur:
            continue
        out.append(idxs)
    return out


# ---------------------------------------------------------------------------
# Contour classification (ToBI-style pitch-accent x boundary-tone)
# ---------------------------------------------------------------------------

PITCH_ACCENTS = ("H*", "L*", "L*+H", "L+H*", "H+!H*", "H* !H*")
BOUNDARY_TONES = ("L-L%", "L-H%", "H-L%", "H-H%")


def three_point(values: np.ndarray) -> tuple[float, float, float]:
    """Mean of the initial, medial, and final thirds of a 1D array."""
    values = np.asarray(values)
    n = len(values)
    i1, i2 = n // 3, 2 * n // 3
    first = values[:i1] if i1 > 0 else values[:1]
    mid = values[i1:i2] if i2 > i1 else values[i1:i1 + 1]
    last = values[i2:] if i2 < n else values[-1:]
    return float(np.mean(first)), float(np.mean(mid)), float(np.mean(last))


MIN_FRAMES_FOR_BITONAL_DETECTION = 15
"""Below this many voiced frames, a segment does not carry enough temporal
resolution to reliably distinguish one local maximum from two (most real
segments in this dataset are short: median duration ~70ms). Below this
threshold, bitonal detection is skipped entirely and the segment is
classified using level/slope alone."""

MIN_PEAK_PROMINENCE_HZ = 15.0
"""Absolute floor on peak prominence (Hz) for bitonal-accent detection, in
addition to the relative prominence_frac-of-range threshold. A purely
relative threshold (e.g. 15% of the segment's own pitch range) is easily
satisfied by ordinary pitch-tracker jitter in narrow-range, low-amplitude
segments -- found during validation against real recordings: a 30-frame
segment spanning only a 45Hz range (147-192Hz) with a few Hz of frame-to-
frame jitter was misclassified as bitonal (H* !H*) under a 1.0Hz absolute
floor, purely from noise, not a genuine second accent peak. 15Hz was chosen
as comfortably above typical cross-correlation pitch-tracker jitter (a few
Hz) while remaining well below real multi-peaked calls seen during manual
spectrogram review."""


def classify_contour(seg_f0: np.ndarray, prominence_frac: float = 0.15) -> tuple[Optional[str], Optional[str], str]:
    """Classify a segment's pitch contour into a ToBI-style pitch-accent x
    boundary-tone label.

    Parameters
    ----------
    seg_f0 : F0 values (Hz) for one voiced segment, no NaNs.
    prominence_frac : minimum peak prominence, as a fraction of the segment's
        pitch range, used for local-maximum detection (floor of 1 Hz).

    Returns
    -------
    (pitch_accent, boundary_tone, detail_string). Returns (None, None, msg)
    if the segment has fewer than 4 voiced frames (too short to classify).

    Note on peak detection near segment edges: peak detection uses
    scipy.signal.find_peaks with a prominence threshold. Prominence is
    computed relative to the lowest point between a peak and either a
    higher peak or the edge of the array being searched. A true local
    maximum whose preceding or following trough falls very close to the
    segment boundary therefore has an artificially low computed prominence
    and can fail to be detected as a second peak -- misclassifying a
    genuinely bitonal contour (e.g. H+!H*) as a simple accent. Segments are
    cut wherever voicing was detected to start/stop, not at any
    phonetically principled point, so this can't be resolved without data
    from outside the segment that isn't available. This is a known,
    accepted limitation, documented and pinned by
    tests/test_core.py::test_edge_peak_limitation_is_documented. See
    MIN_PEAK_PROMINENCE_HZ and MIN_FRAMES_FOR_BITONAL_DETECTION above for
    two related, narrower issues that were found and fixed.
    """
    f0v = np.asarray(seg_f0)
    n = len(f0v)
    if n < 4:
        return None, None, "insufficient voiced pitch to classify"

    med = np.median(f0v)
    pitch_range = f0v.max() - f0v.min()
    i1, i2 = n // 3, 2 * n // 3
    first = f0v[:i1] if i1 > 0 else f0v[:1]
    mid = f0v[i1:i2] if i2 > i1 else f0v[i1:i1 + 1]
    last = f0v[i2:] if i2 < n else f0v[-1:]

    start_level = np.mean(first)
    end_level = np.mean(last[-max(1, len(last) // 2):])
    prelast_level = np.mean(mid) if len(mid) else start_level

    win = max(5, n // 6)
    if win % 2 == 0:
        win += 1
    if n >= MIN_FRAMES_FOR_BITONAL_DETECTION:
        # A true peak very close to the segment boundary may still be
        # missed here (see docstring above); nothing is fabricated to
        # compensate for it.
        smooth = np.convolve(f0v, np.ones(win) / win, mode="valid") if n > win else f0v
        min_prom = max(pitch_range * prominence_frac, MIN_PEAK_PROMINENCE_HZ)
        peak_idx, _ = find_peaks(smooth, prominence=min_prom)
        peaks = len(peak_idx)
    else:
        peaks = 0

    slope_first_half = f0v[i2 - 1] - f0v[0] if i2 > 0 else 0
    slope_last_20pct = last[-1] - last[0] if len(last) > 1 else 0

    if peaks >= 2:
        pitch_accent = "H* !H*" if start_level >= med else "H+!H*"
    elif start_level >= med and slope_first_half <= 0:
        pitch_accent = "H*"
    elif start_level < med and slope_first_half > 0:
        pitch_accent = "L*"
    elif start_level < med and (mid.max() if len(mid) else f0v.max()) > med and slope_last_20pct <= 0:
        pitch_accent = "L*+H"
    elif start_level < med and slope_first_half > 0:
        pitch_accent = "L+H*"
    else:
        pitch_accent = "L+H*" if slope_first_half > 0 else "H*"

    phrase_level = "H" if prelast_level >= med else "L"
    boundary_level = "H" if slope_last_20pct > 0 or end_level >= med else "L"
    boundary_tone = f"{phrase_level}-{boundary_level}%"

    detail = (
        f"start={start_level:.0f}Hz, median={med:.0f}Hz, end={end_level:.0f}Hz, "
        f"peaks={peaks}, first-half slope={slope_first_half:+.0f}Hz, "
        f"final slope={slope_last_20pct:+.0f}Hz"
    )
    return pitch_accent, boundary_tone, detail


def flag_octave_error(seg_f0: np.ndarray, lo: float = 0.6, hi: float = 1.7) -> bool:
    """Flag a segment if any frame-to-frame pitch ratio falls outside
    [lo, hi], indicating the cross-correlation tracker may have locked onto
    half or double the true fundamental for part of the segment.

    Validated by manual inspection against spectrograms (see paper Methods,
    "Quality control"): reliably catches implausible intra-segment frequency
    jumps while not flagging segments with unusually low but internally
    consistent pitch throughout.
    """
    seg_f0 = np.asarray(seg_f0)
    if len(seg_f0) < 2:
        return False
    ratios = seg_f0[1:] / seg_f0[:-1]
    return bool(np.any((ratios > hi) | (ratios < lo)))


# ---------------------------------------------------------------------------
# Pitch-intensity coordination metrics
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    peak_pitch_t: float
    peak_pitch_hz: float
    peak_intensity_t: float
    peak_intensity_db: float
    peak_offset_ms: float
    pitch_initial: float
    pitch_mid: float
    pitch_final: float
    intensity_initial: float
    intensity_mid: float
    intensity_final: float
    covary_initial_to_mid: bool
    covary_mid_to_final: bool

    def as_dict(self) -> dict:
        return {
            "peak_pitch_t": round(self.peak_pitch_t, 3),
            "peak_pitch_hz": round(self.peak_pitch_hz, 1),
            "peak_intensity_t": round(self.peak_intensity_t, 3),
            "peak_intensity_db": round(self.peak_intensity_db, 1),
            "peak_offset_ms": round(self.peak_offset_ms, 1),
            "pitch_initial": round(self.pitch_initial, 1),
            "pitch_mid": round(self.pitch_mid, 1),
            "pitch_final": round(self.pitch_final, 1),
            "intensity_initial": round(self.intensity_initial, 1),
            "intensity_mid": round(self.intensity_mid, 1),
            "intensity_final": round(self.intensity_final, 1),
            "covary_initial_to_mid": self.covary_initial_to_mid,
            "covary_mid_to_final": self.covary_mid_to_final,
        }


def intensity_alignment(
    seg_t: np.ndarray,
    seg_f0: np.ndarray,
    int_times: np.ndarray,
    int_values: np.ndarray,
) -> Optional[AlignmentResult]:
    """Compute peak-timing offset and initial/mid/final covariation between
    a segment's pitch contour and the intensity contour over the same span.

    Returns None if no intensity samples fall within (or near) the segment.
    """
    t0, t1 = seg_t[0], seg_t[-1]
    mask = (int_times >= t0) & (int_times <= t1)
    if mask.sum() < 2:
        mask = (int_times >= t0 - 0.01) & (int_times <= t1 + 0.01)
    seg_int_t = int_times[mask]
    seg_int_v = int_values[mask]
    if len(seg_int_t) == 0:
        return None

    peak_pitch_idx = int(np.argmax(seg_f0))
    peak_pitch_t, peak_pitch_v = float(seg_t[peak_pitch_idx]), float(seg_f0[peak_pitch_idx])
    peak_int_idx = int(np.argmax(seg_int_v))
    peak_int_t, peak_int_v = float(seg_int_t[peak_int_idx]), float(seg_int_v[peak_int_idx])
    offset_ms = (peak_int_t - peak_pitch_t) * 1000

    p_init, p_mid, p_final = three_point(seg_f0)
    int_on_pitch_grid = np.interp(seg_t, int_times, int_values)
    i_init, i_mid, i_final = three_point(int_on_pitch_grid)

    covary_im = bool(np.sign(p_mid - p_init) == np.sign(i_mid - i_init))
    covary_mf = bool(np.sign(p_final - p_mid) == np.sign(i_final - i_mid))

    return AlignmentResult(
        peak_pitch_t=peak_pitch_t, peak_pitch_hz=peak_pitch_v,
        peak_intensity_t=peak_int_t, peak_intensity_db=peak_int_v,
        peak_offset_ms=offset_ms,
        pitch_initial=p_init, pitch_mid=p_mid, pitch_final=p_final,
        intensity_initial=i_init, intensity_mid=i_mid, intensity_final=i_final,
        covary_initial_to_mid=covary_im, covary_mid_to_final=covary_mf,
    )
