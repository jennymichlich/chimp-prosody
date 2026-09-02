"""chimp_prosody: pitch-intensity coordination pipeline for the Gombe chimpanzee archive."""

from .core import (
    find_voiced_segments,
    classify_contour,
    flag_octave_error,
    intensity_alignment,
    three_point,
    AlignmentResult,
    PITCH_ACCENTS,
    BOUNDARY_TONES,
)

__all__ = [
    "find_voiced_segments",
    "classify_contour",
    "flag_octave_error",
    "intensity_alignment",
    "three_point",
    "AlignmentResult",
    "PITCH_ACCENTS",
    "BOUNDARY_TONES",
]
