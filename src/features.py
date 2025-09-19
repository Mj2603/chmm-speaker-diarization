from __future__ import annotations
import numpy as np
import librosa


def extract_mfcc(audio: np.ndarray, sr: int = 22050, n_mfcc: int = 13) -> np.ndarray:
    """Extract MFCC features from audio"""
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T  # (time, n_mfcc)


def extract_spectral_features(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract spectral features (centroid, rolloff, bandwidth)"""
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    
    return np.column_stack([spectral_centroids, spectral_rolloff, spectral_bandwidth])


def create_synthetic_audio(n_speakers: int = 3, duration: float = 10.0, sr: int = 22050) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic multi-speaker audio for demonstration"""
    rng = np.random.default_rng(42)
    n_samples = int(duration * sr)
    
    # Generate different frequency components for each speaker
    t = np.linspace(0, duration, n_samples)
    audio = np.zeros(n_samples)
    true_labels = np.zeros(n_samples, dtype=int)
    
    # Create segments with different speakers
    segment_length = n_samples // n_speakers
    for i in range(n_speakers):
        start_idx = i * segment_length
        end_idx = min((i + 1) * segment_length, n_samples)
        
        # Different fundamental frequencies for each speaker
        f0 = 100 + i * 50  # 100, 150, 200 Hz
        speaker_audio = 0.3 * np.sin(2 * np.pi * f0 * t[start_idx:end_idx])
        speaker_audio += 0.1 * np.sin(2 * np.pi * f0 * 2 * t[start_idx:end_idx])  # harmonic
        speaker_audio += 0.05 * rng.normal(0, 1, end_idx - start_idx)  # noise
        
        audio[start_idx:end_idx] = speaker_audio
        true_labels[start_idx:end_idx] = i
    
    return audio, true_labels
