from __future__ import annotations

import numpy as np

from .chmm import CHMM, create_synthetic_chmm
from .features import extract_mfcc


def diarize_speakers(audio: np.ndarray, sr: int = 22050, n_speakers: int = 3) -> np.ndarray:
    """Decode MFCC frames with synthetic parameters for demonstration only.

    The repository does not yet include a training routine or a trained model;
    output from this function is not meaningful speaker diarization.
    """
    if n_speakers < 1:
        raise ValueError("n_speakers must be positive")
    mfccs = extract_mfcc(audio, sr)
    chmm = create_synthetic_chmm(n_states=n_speakers, n_features=mfccs.shape[1])
    return chmm.viterbi(mfccs)


def evaluate_diarization(true_labels: np.ndarray, predicted_labels: np.ndarray) -> dict:
    """Compute clustering agreement metrics for aligned label sequences."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    if true_labels.ndim != 1 or predicted_labels.ndim != 1:
        raise ValueError("label sequences must be one-dimensional")
    if true_labels.size == 0 or true_labels.shape != predicted_labels.shape:
        raise ValueError("label sequences must be non-empty and have equal length")

    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    return {
        "adjusted_rand_index": float(ari),
        "normalized_mutual_info": float(nmi),
        "n_speakers_true": int(np.unique(true_labels).size),
        "n_speakers_pred": int(np.unique(predicted_labels).size),
    }
