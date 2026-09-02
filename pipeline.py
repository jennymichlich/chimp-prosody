"""
chimp_prosody.pipeline
========================

Single, manifest-driven entry point for the full pipeline: given a table of
(subject, file path(s), age class, age range), extract pitch and intensity,
segment, classify, and compute pitch-intensity coordination metrics for
every subject, writing one combined output CSV.

This replaces the previous approach of ~15 separate per-subject scripts
(each a `sed`-adapted copy of the last one), which was the source of at
least one real bug in this project: a substitution that silently failed to
update a file path, causing one script to process the wrong subject's
audio for a full run. With a manifest table and one code path, that
failure mode is not possible -- a wrong path is a wrong row in a CSV that
can be diffed and checked, not a wrong line buried in a copy of a script.

Usage
-----
    python -m chimp_prosody.pipeline manifest.csv output.csv

Manifest CSV columns
---------------------
    subject       : short subject identifier, e.g. "Sherry_raw"
    file_paths     : one file path, or multiple separated by ";" (a subject
                     may have more than one archived tape, e.g. Flint)
    age_class     : "infant" | "juvenile" | "adolescent"
    age_lo        : lower bound of age range (years)
    age_hi        : upper bound of age range (years)
    age_mid       : point estimate of age (years), typically (lo+hi)/2
    chunk_minutes : (optional) override the default chunk duration for this
                     file, in minutes. Leave blank to auto-select.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import (
    find_voiced_segments,
    classify_contour,
    flag_octave_error,
    intensity_alignment,
)
from .io_utils import iter_audio_chunks, choose_chunk_duration

PITCH_FLOOR_HZ = 100.0
PITCH_CEILING_HZ = 2000.0
INTENSITY_MIN_PITCH_HZ = 200.0
SEGMENT_GAP_THRESH_S = 0.02
SEGMENT_MIN_DUR_S = 0.03


@dataclass
class SubjectSpec:
    subject: str
    file_paths: list[str]
    age_class: str
    age_lo: float
    age_hi: float
    age_mid: float
    chunk_minutes: float | None = None


def read_manifest(path: str) -> list[SubjectSpec]:
    specs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            specs.append(SubjectSpec(
                subject=row["subject"],
                file_paths=[p.strip() for p in row["file_paths"].split(";") if p.strip()],
                age_class=row["age_class"],
                age_lo=float(row["age_lo"]),
                age_hi=float(row["age_hi"]),
                age_mid=float(row["age_mid"]),
                chunk_minutes=float(row["chunk_minutes"]) if row.get("chunk_minutes") else None,
            ))
    return specs


def process_file(file_path: str, chunk_minutes: float | None = None) -> list[dict]:
    """Process one audio file end to end: chunked pitch/intensity
    extraction, segmentation, classification, and alignment metrics.
    Returns a list of segment-level result dicts with times already
    corrected onto this file's own continuous timeline.
    """
    chunk_s = (chunk_minutes * 60.0) if chunk_minutes else choose_chunk_duration(file_path)
    results = []
    seg_counter = 0

    for chunk in iter_audio_chunks(file_path, chunk_duration_s=chunk_s):
        pitch = chunk.sound.to_pitch_cc(pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ)
        intensity = chunk.sound.to_intensity(minimum_pitch=INTENSITY_MIN_PITCH_HZ)

        pitch_times = np.array(pitch.xs())
        f0_values = pitch.selected_array["frequency"]
        f0_values = np.where(f0_values == 0, np.nan, f0_values)
        int_times = np.array(intensity.xs())
        int_values = np.array(intensity.values[0])

        segments = find_voiced_segments(
            pitch_times, f0_values,
            gap_thresh=SEGMENT_GAP_THRESH_S, min_dur=SEGMENT_MIN_DUR_S,
        )

        for idxs in segments:
            seg_t = pitch_times[idxs]
            seg_f0 = f0_values[idxs]
            pitch_accent, boundary_tone, detail = classify_contour(seg_f0)
            octave_flag = flag_octave_error(seg_f0)
            align = intensity_alignment(seg_t, seg_f0, int_times, int_values)

            seg_counter += 1
            row = {
                "file": file_path,
                "segment": seg_counter,
                "start_s": round(float(seg_t[0]) + chunk.offset_s, 3),
                "end_s": round(float(seg_t[-1]) + chunk.offset_s, 3),
                "duration_s": round(float(seg_t[-1] - seg_t[0]), 3),
                "pitch_accent": pitch_accent,
                "boundary_tone": boundary_tone,
                "detail": detail,
                "possible_octave_error": octave_flag,
            }
            if align is not None:
                align_dict = align.as_dict()
                align_dict["peak_pitch_t"] = round(align_dict["peak_pitch_t"] + chunk.offset_s, 3)
                align_dict["peak_intensity_t"] = round(align_dict["peak_intensity_t"] + chunk.offset_s, 3)
                row.update(align_dict)
            results.append(row)

        # Release chunk-level Praat objects before the next chunk loads.
        del pitch, intensity, pitch_times, f0_values, int_times, int_values

    return results


def process_subject(spec: SubjectSpec) -> pd.DataFrame:
    """Process all of a subject's recording(s). For subjects with multiple
    archived tapes (e.g. Flint, whose recording was originally split across
    two upload files), each subsequent file's timestamps are offset to
    continue from the end of the previous file, so the subject's segments
    sit on one continuous timeline rather than each file restarting at
    time 0. This matches how multi-tape subjects were handled in the
    original analysis."""
    all_rows = []
    cumulative_offset = 0.0
    for file_path in spec.file_paths:
        rows = process_file(file_path, chunk_minutes=spec.chunk_minutes)
        file_max_end = 0.0
        for r in rows:
            r["start_s"] = round(r["start_s"] + cumulative_offset, 3)
            r["end_s"] = round(r["end_s"] + cumulative_offset, 3)
            if "peak_pitch_t" in r:
                r["peak_pitch_t"] = round(r["peak_pitch_t"] + cumulative_offset, 3)
            if "peak_intensity_t" in r:
                r["peak_intensity_t"] = round(r["peak_intensity_t"] + cumulative_offset, 3)
            r["subject"] = spec.subject
            r["age_class"] = spec.age_class
            r["age_lo"] = spec.age_lo
            r["age_hi"] = spec.age_hi
            r["age_mid"] = spec.age_mid
            file_max_end = max(file_max_end, r["end_s"])
        all_rows.extend(rows)
        cumulative_offset = file_max_end

    df = pd.DataFrame(all_rows)
    if len(df):
        df["segment"] = range(1, len(df) + 1)
    return df


def run_pipeline(manifest_path: str, output_path: str) -> pd.DataFrame:
    specs = read_manifest(manifest_path)
    all_dfs = []
    for spec in specs:
        print(f"Processing {spec.subject} ({len(spec.file_paths)} file(s))...", flush=True)
        df = process_subject(spec)
        print(f"  -> {len(df)} segments", flush=True)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined)} total segments to {output_path}")
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", help="Path to manifest CSV")
    parser.add_argument("output", help="Path to write the combined output CSV")
    args = parser.parse_args()
    run_pipeline(args.manifest, args.output)


if __name__ == "__main__":
    sys.exit(main())
