import os
import numpy as np
import matplotlib.pyplot as plt
from src.features import create_synthetic_audio, extract_mfcc
from src.diarization import diarize_speakers, evaluate_diarization


def main():
    # Create synthetic multi-speaker audio
    audio, true_labels = create_synthetic_audio(n_speakers=3, duration=8.0)
    sr = 22050
    
    # Perform diarization
    predicted_labels = diarize_speakers(audio, sr, n_speakers=3)
    
    # Evaluate performance
    metrics = evaluate_diarization(true_labels, predicted_labels)
    
    os.makedirs("outputs", exist_ok=True)
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    
    # Audio waveform
    t = np.linspace(0, len(audio)/sr, len(audio))
    ax1.plot(t, audio, alpha=0.7)
    ax1.set_title("Synthetic Multi-Speaker Audio")
    ax1.set_ylabel("Amplitude")
    
    # True speaker labels
    ax2.plot(t, true_labels, 'o-', markersize=2)
    ax2.set_title("True Speaker Labels")
    ax2.set_ylabel("Speaker ID")
    ax2.set_ylim(-0.5, 2.5)
    
    # Predicted speaker labels
    ax3.plot(t, predicted_labels, 'o-', markersize=2, color='red')
    ax3.set_title("CHMM Predicted Speaker Labels")
    ax3.set_ylabel("Speaker ID")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylim(-0.5, 2.5)
    
    plt.tight_layout()
    plt.savefig("outputs/diarization_results.png", dpi=150)
    
    print(f"Evaluation Metrics:")
    print(f"Adjusted Rand Index: {metrics['adjusted_rand_index']:.3f}")
    print(f"Normalized Mutual Info: {metrics['normalized_mutual_info']:.3f}")
    print(f"True speakers: {metrics['n_speakers_true']}, Predicted: {metrics['n_speakers_pred']}")


if __name__ == "__main__":
    main()
