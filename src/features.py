from __future__ import annotations

import numpy as np
import librosa


def _validate_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    values = np.asarray(audio, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("audio must be a non-empty mono waveform")
    if not np.isfinite(values).all():
        raise ValueError("audio must contain only finite samples")
    if not isinstance(sr, (int, np.integer)) or sr <= 0:
        raise ValueError("sample rate must be a positive integer")
    return values


def extract_mfcc(audio: np.ndarray, sr: int = 22050, n_mfcc: int = 13) -> np.ndarray:
    """Extract frame-major MFCC features from a mono waveform."""
    values = _validate_audio(audio, sr)
    if not isinstance(n_mfcc, (int, np.integer)) or n_mfcc < 1:
        raise ValueError("n_mfcc must be a positive integer")
    return librosa.feature.mfcc(y=values, sr=sr, n_mfcc=n_mfcc).T


def extract_spectral_features(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract frame-major spectral centroid, rolloff, and bandwidth."""
    values = _validate_audio(audio, sr)
    features = np.column_stack([
        librosa.feature.spectral_centroid(y=values, sr=sr)[0],
        librosa.feature.spectral_rolloff(y=values, sr=sr)[0],
        librosa.feature.spectral_bandwidth(y=values, sr=sr)[0],
    ])
    if not np.isfinite(features).all():
        raise ValueError("audio produced non-finite spectral features")
    return features


def create_synthetic_audio(
    n_speakers: int = 3,
    duration: float = 10.0,
    sr: int = 22050,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate sequential synthetic tones for decoder demonstrations only."""
    if not isinstance(n_speakers, int) or n_speakers < 1:
        raise ValueError("n_speakers must be a positive integer")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if not isinstance(sr, int) or sr <= 0:
        raise ValueError("sample rate must be a positive integer")

    n_samples = int(duration * sr)
    if n_samples < n_speakers:
        raise ValueError("duration is too short for the requested segments")

    rng = np.random.default_rng(42)
    t = np.arange(n_samples, dtype=float) / sr
    audio = np.zeros(n_samples, dtype=float)
    labels = np.zeros(n_samples, dtype=int)
    boundaries = np.linspace(0, n_samples, n_speakers + 1, dtype=int)

    for speaker in range(n_speakers):
        start, end = boundaries[speaker], boundaries[speaker + 1]
        f0 = 100.0 + speaker * 50.0
        segment_time = t[start:end]
        audio[start:end] = (
            0.3 * np.sin(2 * np.pi * f0 * segment_time)
            + 0.1 * np.sin(2 * np.pi * 2 * f0 * segment_time)
            + 0.05 * rng.normal(0.0, 1.0, end - start)
        )
        labels[start:end] = speaker

    return audio, labels
