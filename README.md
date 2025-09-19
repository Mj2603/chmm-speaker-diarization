# Conditional Hidden Markov Model for Speaker Diarization

This repo implements a novel trainable algorithm for speaker diarization using a conditional Hidden Markov Model (CHMM) that adapts transition probabilities based on acoustic features.

Inspired by: elmella/new-phone-who-dis (https://github.com/elmella/new-phone-who-dis)

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python examples/demo.py
```

## Components
- src/chmm.py — Conditional HMM implementation with trainable transitions
- src/features.py — MFCC and spectral features extraction
- src/diarization.py — Speaker diarization pipeline
- examples/demo.py — synthetic audio demo with visualization

Outputs saved to outputs/.
