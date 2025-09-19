from __future__ import annotations
import numpy as np
from .chmm import CHMM, create_synthetic_chmm
from .features import extract_mfcc, create_synthetic_audio


def diarize_speakers(audio: np.ndarray, sr: int = 22050, n_speakers: int = 3) -> np.ndarray:
    """Perform speaker diarization using CHMM"""
    # Extract features
    mfccs = extract_mfcc(audio, sr)
    
    # Create and train CHMM (simplified - in practice would use EM algorithm)
    chmm = create_synthetic_chmm(n_states=n_speakers, n_features=mfccs.shape[1])
    
    # Viterbi decoding
    predicted_states = chmm.viterbi(mfccs)
    
    return predicted_states


def evaluate_diarization(true_labels: np.ndarray, predicted_labels: np.ndarray) -> dict:
    """Evaluate diarization performance"""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    
    return {
        'adjusted_rand_index': ari,
        'normalized_mutual_info': nmi,
        'n_speakers_true': len(np.unique(true_labels)),
        'n_speakers_pred': len(np.unique(predicted_labels))
    }
