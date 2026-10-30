import os

import matplotlib.pyplot as plt
import numpy as np

from src.features import create_synthetic_audio
from src.diarization import diarize_speakers, evaluate_diarization


def main():
    audio, sample_labels = create_synthetic_audio(n_speakers=3, duration=8.0)
    sr = 22050
    predicted_labels = diarize_speakers(audio, sr, n_speakers=3)

    # Align sample labels to the frame-level MFCC/Viterbi sequence for plotting.
    sample_indices = np.linspace(
        0, len(sample_labels) - 1, num=len(predicted_labels), dtype=int
    )
    frame_labels = sample_labels[sample_indices]
    metrics = evaluate_diarization(frame_labels, predicted_labels)

    duration = len(audio) / sr
    frame_times = np.linspace(0, duration, num=len(predicted_labels), endpoint=False)
    os.makedirs("outputs", exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    sample_times = np.arange(len(audio)) / sr
    ax1.plot(sample_times, audio, alpha=0.7)
    ax1.set_title("Synthetic Audio")
    ax1.set_ylabel("Amplitude")

    ax2.step(frame_times, frame_labels, where="post")
    ax2.set_title("Synthetic Segment Labels (frame-aligned)")
    ax2.set_ylabel("Label")
    ax2.set_ylim(-0.5, 2.5)

    ax3.step(frame_times, predicted_labels, where="post", color="red")
    ax3.set_title("CHMM Output Using Random Synthetic Parameters")
    ax3.set_ylabel("State")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylim(-0.5, 2.5)

    plt.tight_layout()
    plt.savefig("outputs/diarization_results.png", dpi=150)

    print("Synthetic decoder demonstration (not diarization performance)")
    print(f"Adjusted Rand Index: {metrics['adjusted_rand_index']:.3f}")
    print(f"Normalized Mutual Info: {metrics['normalized_mutual_info']:.3f}")


if __name__ == "__main__":
    main()
