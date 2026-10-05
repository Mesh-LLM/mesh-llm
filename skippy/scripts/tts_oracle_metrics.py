"""The shared PCM acceptance contract for comparison, evidence writing and verification."""

from __future__ import annotations

import math

MAX_RELATIVE_RMS_ERROR = 0.02
MIN_WAVEFORM_COSINE = 0.9995


def validate_tts_metrics(metrics: object) -> None:
    """Reject incomplete, non-finite or out-of-policy PCM evidence."""
    if not isinstance(metrics, dict):
        raise ValueError("TTS oracle evidence lacks PCM metrics")
    for field in ("sample_rate_hz", "channels", "sample_count"):
        value = metrics.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"TTS PCM metric {field} must be a positive integer")
    bounds = {
        "relative_rms_error": (0.0, MAX_RELATIVE_RMS_ERROR),
        "waveform_cosine": (MIN_WAVEFORM_COSINE, 1.0),
    }
    for field, (minimum, maximum) in bounds.items():
        value = metrics.get(field)
        if (type(value) not in (int, float) or not minimum <= value <= maximum
                or not math.isfinite(value)):
            raise ValueError(f"TTS PCM metric {field} must be finite and within [{minimum}, {maximum}]")
