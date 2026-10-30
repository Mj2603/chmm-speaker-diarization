# Conditional HMM Decoder Prototype

This repository contains a small Python implementation of a Gaussian hidden Markov model whose transition probabilities depend on the previous frame's acoustic feature vector.

## Current implementation

- MFCC feature extraction with librosa.
- Feature-conditioned transition probabilities computed from fixed weight tensors.
- Gaussian state emission log probabilities.
- Viterbi decoding and ARI/NMI utilities.
- A reproducible synthetic parameter generator and plotting demo.

**Training and real speaker diarization are not implemented.** The demo uses randomly generated model parameters and synthetic tones; its ARI/NMI scores are not evidence of diarization accuracy. This repository should currently be described as a decoder prototype, not as a validated or trainable speaker-diarization system.

## Run the synthetic demonstration

Install dependencies from `requirements.txt`, then run from the repository root:

```bash
python examples/demo.py
```

The script writes a plot to `outputs/diarization_results.png`. The displayed metrics compare synthetic labels with output from an untrained decoder and have no performance interpretation.

## Model outline

For each previous state, the transition logits are a linear function of the previous frame's feature vector and are normalized with softmax. State emissions use multivariate Gaussian parameters supplied to `CHMM`. The decoder uses a uniform initial-state prior and computes the most likely path with Viterbi dynamic programming.

## Next research steps

1. Add parameter estimation using speaker-labeled training sequences or a documented unsupervised objective.
2. Evaluate on a public diarization corpus with speaker-disjoint train/validation/test splits.
3. Compare against simple clustering and established diarization baselines.
4. Report DER/JER, overlap handling, feature configuration, and reproducible seeds.
5. Keep synthetic decoder demonstrations separate from real-world performance claims.

## License

See the repository license file.
