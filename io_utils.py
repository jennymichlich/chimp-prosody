"""
chimp_prosody.io_utils
=======================

Memory-safe audio reading for long field recordings.

Background: earlier in this project, long recordings (>40 minutes) were
split into chunks by shelling out to ffmpeg and writing separate chunk
files to disk before processing each one. This had two real failure modes
we hit in practice:

1. `ffmpeg -c copy` (stream copy) silently ignores `-ss`/`-t` trim points
   for some codecs (notably FLAC), producing "chunk" files that were each
   silent full-length copies of the original recording rather than actual
   trimmed segments. This was only caught by manually checking chunk
   durations after the fact.
2. Even with correct (re-encoded) trimming, the extra ffmpeg step and
   on-disk intermediate files added a layer of file-path bookkeeping that
   was a source of subject-mixup bugs when scripts were adapted via `sed`
   substitution for each new subject.

This module avoids both problems: it reads only the requested sample range
directly from the source file using `soundfile` (which supports random
access into most audio containers, including FLAC and WAV, without
decoding the whole file), and hands back an in-memory numpy array. No
intermediate files are ever written.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import soundfile as sf
import parselmouth


@dataclass
class AudioChunk:
    """One chunk of a longer recording, with its offset into the full
    recording's timeline so per-chunk segment timestamps can be corrected
    back onto the original continuous timeline."""
    sound: "parselmouth.Sound"
    offset_s: float
    duration_s: float


def iter_audio_chunks(path: str, chunk_duration_s: float = 600.0) -> Iterator[AudioChunk]:
    """Yield successive chunks of a (possibly very long) audio file as
    in-memory parselmouth.Sound objects, without ever loading the full file
    or writing intermediate files to disk.

    Parameters
    ----------
    path : path to the source audio file (WAV, FLAC, etc.)
    chunk_duration_s : target chunk length in seconds. The final chunk may
        be shorter. 600s (10 minutes) keeps peak memory well under 1GB for
        96kHz mono recordings in our testing (see project notes: a full
        53-minute file loaded whole caused an OOM kill at ~3GB RSS in a
        3.9GB-RAM container; 10-minute chunks used well under 1GB).

    Yields
    ------
    AudioChunk objects in chronological order.
    """
    info = sf.info(path)
    sr = info.samplerate
    total_frames = info.frames
    chunk_frames = int(chunk_duration_s * sr)

    start_frame = 0
    while start_frame < total_frames:
        stop_frame = min(start_frame + chunk_frames, total_frames)
        data, read_sr = sf.read(path, start=start_frame, stop=stop_frame, dtype="float64", always_2d=False)
        assert read_sr == sr

        # parselmouth.Sound can be built directly from a numpy array plus
        # sampling frequency -- no temp file needed.
        if data.ndim > 1:
            data = data.mean(axis=1)  # downmix to mono if the source is multi-channel
        sound = parselmouth.Sound(data, sampling_frequency=sr)

        offset_s = start_frame / sr
        duration_s = (stop_frame - start_frame) / sr
        yield AudioChunk(sound=sound, offset_s=offset_s, duration_s=duration_s)

        start_frame = stop_frame


def choose_chunk_duration(path: str, target_peak_mb: float = 700.0) -> float:
    """Suggest a chunk duration (seconds) that should keep peak memory for
    pitch/intensity extraction under `target_peak_mb`, based on the
    empirical scaling observed in this project (roughly linear in duration
    for a fixed sample rate; see project notes for the 96kHz-mono case this
    was calibrated against). Conservative by design -- prefers smaller
    chunks over risking another OOM kill.
    """
    info = sf.info(path)
    duration_s = info.frames / info.samplerate
    # Empirically: a 37-minute 96kHz mono file used ~2GB peak RSS unchunked.
    # That's roughly 54 MB/min. Solve for chunk length hitting target_peak_mb.
    mb_per_minute = 54.0
    minutes = target_peak_mb / mb_per_minute
    suggested = max(300.0, minutes * 60.0)  # never go below 5 minutes
    return min(suggested, duration_s) if duration_s > 0 else suggested
