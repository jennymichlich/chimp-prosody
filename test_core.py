"""
Unit tests for chimp_prosody.core, using synthetic (not recorded) pitch and
intensity arrays with known properties. These exist specifically because,
prior to this refactor, the classification logic had never been tested
against anything with a known-correct answer -- only validated indirectly by
visual inspection of real recordings (see the paper's pitch-tracking
validation subsection). Unit tests here catch regressions in the
segmentation/classification thresholds that visual spot-checks would not
reliably catch.
"""

import numpy as np
import pytest

from chimp_prosody.core import (
    find_voiced_segments,
    classify_contour,
    flag_octave_error,
    intensity_alignment,
    three_point,
)


# ---------------------------------------------------------------------------
# find_voiced_segments
# ---------------------------------------------------------------------------

def test_two_segments_separated_by_gap():
    """Two voiced regions separated by an unvoiced gap should yield two
    segments, not one."""
    times = np.arange(0, 2.0, 0.01)  # 200 frames, 10ms apart
    f0 = np.full_like(times, np.nan)
    f0[0:50] = 200.0     # voiced 0.00-0.49s
    f0[80:150] = 250.0   # voiced 0.80-1.49s (gap of 0.30s >> 0.02s threshold)
    segments = find_voiced_segments(times, f0, gap_thresh=0.02, min_dur=0.03)
    assert len(segments) == 2
    assert times[segments[0][0]] < 0.5
    assert times[segments[1][0]] > 0.7


def test_short_segment_is_dropped():
    """A voiced region shorter than min_dur should not be returned."""
    times = np.arange(0, 1.0, 0.01)
    f0 = np.full_like(times, np.nan)
    f0[0:2] = 200.0  # only 2 frames = ~10ms, well under a 30ms minimum
    segments = find_voiced_segments(times, f0, gap_thresh=0.02, min_dur=0.03)
    assert len(segments) == 0


def test_small_gap_does_not_split():
    """A gap smaller than gap_thresh should NOT split one segment into two."""
    times = np.arange(0, 1.0, 0.01)
    f0 = np.full_like(times, np.nan)
    f0[0:20] = 200.0
    f0[21:60] = 200.0  # 1-frame (0.01s) gap, under the 0.02s threshold
    segments = find_voiced_segments(times, f0, gap_thresh=0.02, min_dur=0.03)
    assert len(segments) == 1


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        find_voiced_segments(np.arange(10), np.arange(5, dtype=float))


# ---------------------------------------------------------------------------
# classify_contour
# ---------------------------------------------------------------------------

def test_falling_contour_from_high_is_H_star():
    """A segment that starts above its own median and falls monotonically
    should be classified as a simple H* accent."""
    seg_f0 = np.linspace(400, 200, 30)  # starts high, falls
    accent, boundary, _ = classify_contour(seg_f0)
    assert accent == "H*"


def test_rising_contour_from_low_is_L_star():
    """A segment that starts below its own median and rises monotonically
    is classified as L* under this scheme's rules."""
    seg_f0 = np.linspace(200, 400, 30)  # starts low, rises
    accent, boundary, _ = classify_contour(seg_f0)
    assert accent == "L*"


def test_two_peaked_contour_is_bitonal():
    """A segment with two clear local maxima, well inside the segment (not
    near the boundary) and each with a visible trough on both sides, should
    be classified as one of the bitonal (!H) categories."""
    x = np.linspace(0, 1, 90)
    seg_f0 = (
        250
        + 120 * np.exp(-((x - 0.30) ** 2) / (2 * 0.06 ** 2))
        + 120 * np.exp(-((x - 0.70) ** 2) / (2 * 0.06 ** 2))
    )
    accent, boundary, _ = classify_contour(seg_f0)
    assert accent in ("H* !H*", "H+!H*")


def test_edge_peak_limitation_is_documented():
    """A genuine local maximum sitting very close to the segment's boundary
    can be missed, because its prominence can't be reliably computed
    without data on the far side of the boundary that we simply don't
    have. This is an accepted, documented limitation (see classify_contour
    docstring). This test pins the current, honest behavior so any future
    change to this trade-off is a visible, deliberate decision rather than
    an accidental regression.
    """
    x = np.linspace(0, 1, 60)
    seg_f0 = 300 + 100 * np.sin(2 * np.pi * 2 * x)  # first peak near t=0.125
    accent, _, _ = classify_contour(seg_f0)
    assert accent == "L+H*"  # the near-edge peak is missed; documented, not silently wrong


def test_short_segment_never_gets_spurious_bitonal_label():
    """A short segment (below MIN_FRAMES_FOR_BITONAL_DETECTION) should
    never be labeled bitonal, regardless of shape -- there isn't enough
    temporal resolution to support that distinction reliably."""
    from chimp_prosody.core import MIN_FRAMES_FOR_BITONAL_DETECTION
    n = MIN_FRAMES_FOR_BITONAL_DETECTION - 5
    assert n >= 4, "test assumption: still long enough to classify at all"
    seg_f0 = np.linspace(200, 260, n)  # short, simple, monotonic rise
    accent, _, _ = classify_contour(seg_f0)
    assert accent not in ("H* !H*", "H+!H*")


def test_narrow_range_jitter_is_not_misread_as_bitonal():
    """Regression test for a real false positive found during validation
    against actual recordings: a segment with a small overall pitch range
    (45Hz) and a few Hz of ordinary frame-to-frame jitter was, under a
    purely relative (15%-of-range) prominence threshold, misclassified as
    bitonal from noise alone. MIN_PEAK_PROMINENCE_HZ adds an absolute floor
    to prevent this.
    """
    seg_f0 = np.array([
        192.5, 190.0, 178.7, 178.3, 176.5, 175.7, 172.4, 172.2, 172.3, 173.5,
        173.4, 171.0, 170.7, 167.5, 169.0, 168.7, 154.8, 155.7, 158.6, 165.6,
        166.1, 155.6, 147.1, 147.0, 155.0, 158.9, 160.1, 170.1, 173.5, 178.7,
    ])
    accent, _, _ = classify_contour(seg_f0)
    assert accent not in ("H* !H*", "H+!H*")


def test_too_short_segment_returns_none():
    accent, boundary, detail = classify_contour(np.array([200.0, 210.0]))
    assert accent is None and boundary is None
    assert "insufficient" in detail


def test_boundary_tone_is_one_of_four_categories():
    seg_f0 = np.linspace(300, 150, 20)
    _, boundary, _ = classify_contour(seg_f0)
    assert boundary in ("L-L%", "L-H%", "H-L%", "H-H%")


# ---------------------------------------------------------------------------
# flag_octave_error
# ---------------------------------------------------------------------------

def test_normal_contour_not_flagged():
    seg_f0 = np.array([200.0, 205.0, 210.0, 208.0, 203.0])
    assert flag_octave_error(seg_f0) is False


def test_octave_jump_up_is_flagged():
    """A frame-to-frame ratio > 1.7 (e.g. a halving artifact resolving)
    should be flagged."""
    seg_f0 = np.array([110.0, 112.0, 108.0, 1180.0, 1190.0])  # ~10x jump
    assert flag_octave_error(seg_f0) is True


def test_octave_jump_down_is_flagged():
    seg_f0 = np.array([1200.0, 1190.0, 118.0, 115.0])  # ~10x drop
    assert flag_octave_error(seg_f0) is True


def test_gradual_change_not_flagged():
    """A large but gradual change (no single-frame ratio outside bounds)
    should not be flagged, even if the total change across the segment is
    large."""
    seg_f0 = np.linspace(150, 1400, 50)  # ~9x change but spread over 50 frames
    assert flag_octave_error(seg_f0) is False


def test_single_frame_never_flagged():
    assert flag_octave_error(np.array([200.0])) is False


# ---------------------------------------------------------------------------
# three_point
# ---------------------------------------------------------------------------

def test_three_point_split_on_simple_ramp():
    values = np.arange(1, 10)  # 1..9, thirds are [1,2,3],[4,5,6],[7,8,9]
    first, mid, last = three_point(values)
    assert first == pytest.approx(2.0)
    assert mid == pytest.approx(5.0)
    assert last == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# intensity_alignment
# ---------------------------------------------------------------------------

def test_peak_offset_matches_known_synthetic_delay():
    """Construct a segment where pitch peaks at t=0.50s and intensity peaks
    exactly 80ms later; verify the recovered offset matches."""
    seg_t = np.linspace(0.0, 1.0, 101)  # 10ms steps
    seg_f0 = 200 + 100 * np.exp(-((seg_t - 0.50) ** 2) / (2 * 0.05 ** 2))

    int_times = np.linspace(0.0, 1.0, 101)
    int_values = 50 + 20 * np.exp(-((int_times - 0.58) ** 2) / (2 * 0.05 ** 2))

    result = intensity_alignment(seg_t, seg_f0, int_times, int_values)
    assert result is not None
    assert result.peak_offset_ms == pytest.approx(80.0, abs=15.0)  # grid-limited precision


def test_covarying_segment_flags_true_both_halves():
    """Pitch and intensity both rise then both fall -> covary=True for both
    halves."""
    seg_t = np.linspace(0, 1, 30)
    shape = np.concatenate([np.linspace(0, 1, 15), np.linspace(1, 0, 15)])
    seg_f0 = 200 + 100 * shape
    int_values = 40 + 20 * shape
    result = intensity_alignment(seg_t, seg_f0, seg_t, int_values)
    assert result.covary_initial_to_mid is True
    assert result.covary_mid_to_final is True


def test_anticorrelated_segment_flags_false():
    """Pitch rises while intensity falls -> covary=False."""
    seg_t = np.linspace(0, 1, 30)
    seg_f0 = np.linspace(200, 400, 30)       # rising
    int_values = np.linspace(60, 30, 30)     # falling
    result = intensity_alignment(seg_t, seg_f0, seg_t, int_values)
    assert result.covary_initial_to_mid is False
    assert result.covary_mid_to_final is False


def test_no_overlapping_intensity_returns_none():
    seg_t = np.array([5.0, 5.1, 5.2])
    seg_f0 = np.array([200.0, 210.0, 205.0])
    int_times = np.array([100.0, 100.1, 100.2])  # far away in time
    int_values = np.array([40.0, 42.0, 41.0])
    result = intensity_alignment(seg_t, seg_f0, int_times, int_values)
    assert result is None
