# Novel Conditional Hidden Markov Model Algorithm for Multi-Speaker Audio Diarization and Voice Activity Detection

## Overview

This repository presents a groundbreaking approach to speaker diarization using a novel Conditional Hidden Markov Model (CHMM) architecture that dynamically adapts transition probabilities based on acoustic feature characteristics. Unlike traditional HMMs with static transition matrices, this implementation introduces feature-dependent transitions that significantly improve speaker separation accuracy in multi-speaker audio environments.

## Research Innovation

Speaker diarization, the task of determining "who spoke when" in audio recordings, is a fundamental challenge in speech processing with applications ranging from meeting transcription to forensic audio analysis. Traditional approaches often struggle with overlapping speech, background noise, and speaker similarity. This work introduces a paradigm shift by making the HMM transition probabilities conditional on acoustic features, enabling the model to adapt its behavior based on the current acoustic context.

## Key Technical Contributions

- **Conditional Transition Probabilities**: Novel CHMM architecture with feature-dependent transition matrices
- **Adaptive Acoustic Modeling**: Dynamic adaptation to changing acoustic environments and speaker characteristics
- **Multi-Feature Integration**: Comprehensive feature extraction combining MFCC, spectral, and temporal characteristics
- **Robust Viterbi Decoding**: Enhanced state sequence estimation with improved convergence properties
- **Performance Evaluation**: Comprehensive metrics including Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI)

## Technical Architecture

The system consists of four core components:

1. **Feature Extraction Engine**: Multi-dimensional acoustic feature computation (MFCC, spectral centroid, rolloff, bandwidth)
2. **Conditional HMM Core**: Novel CHMM implementation with feature-dependent transition modeling
3. **Viterbi Decoder**: Enhanced state sequence estimation with improved numerical stability
4. **Evaluation Framework**: Comprehensive performance assessment with multiple metrics

## Mathematical Foundation

### Conditional Transition Modeling
Traditional HMMs use static transition probabilities:
```
P(s_t = j | s_{t-1} = i) = A_{ij}
```

Our CHMM introduces feature-dependent transitions:
```
P(s_t = j | s_{t-1} = i, f_t) = σ(W_{ij} · f_t + b_{ij})
```
where `f_t` represents the acoustic feature vector at time `t`, and `σ` is the softmax function.

### Feature-Dependent Emission Probabilities
The emission probabilities are modeled as multivariate Gaussians with feature-conditional parameters:
```
P(o_t | s_t = i, f_t) = N(μ_i(f_t), Σ_i(f_t))
```

## Applications and Use Cases

- **Meeting Transcription**: Automatic speaker identification in conference calls and meetings
- **Forensic Audio Analysis**: Speaker identification in legal and security applications
- **Accessibility Technology**: Real-time speaker separation for hearing-impaired individuals
- **Content Analysis**: Automated podcast and video content analysis
- **Research Applications**: Academic studies in speech processing and machine learning

## Performance Advantages

- **Improved Accuracy**: Feature-dependent transitions provide better speaker separation
- **Robustness**: Adaptive modeling handles varying acoustic conditions
- **Scalability**: Efficient implementation suitable for real-time applications
- **Flexibility**: Configurable architecture for different audio domains

## Quickstart

```bash
# Clone the repository
git clone https://github.com/Mj2603/chmm-speaker-diarization.git
cd chmm-speaker-diarization

# Set up environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the demonstration
python examples/demo.py
```

## Expected Outputs

- Multi-speaker audio visualization with true and predicted speaker labels
- Performance metrics (ARI, NMI, accuracy)
- Acoustic feature analysis and visualization
- Model convergence and training statistics

## Project Structure

```
chmm-speaker-diarization/
├── src/
│   ├── chmm.py              # Core CHMM implementation
│   ├── features.py          # Acoustic feature extraction
│   └── diarization.py       # Speaker diarization pipeline
├── examples/
│   └── demo.py              # Complete demonstration
├── outputs/                 # Generated plots and results
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Dependencies

- **NumPy**: Numerical computations and array operations
- **SciPy**: Statistical functions and optimization
- **Matplotlib**: Visualization and plotting
- **Librosa**: Audio processing and feature extraction
- **Scikit-learn**: Machine learning utilities and metrics

## Research Applications

This framework is particularly valuable for:
- **Academic Research**: Novel approaches to speaker diarization and speech processing
- **Industry Applications**: Commercial speech processing and transcription services
- **Accessibility**: Real-time speaker separation for assistive technologies
- **Forensic Analysis**: Speaker identification in legal and security contexts

## Performance Benchmarks

The CHMM approach has been evaluated on various datasets:
- **Synthetic Multi-Speaker Audio**: Demonstrated superior performance over traditional HMMs
- **Noisy Environments**: Robust performance in challenging acoustic conditions
- **Real-Time Processing**: Efficient implementation suitable for live applications

## Future Enhancements

- Integration with deep learning frameworks for end-to-end training
- Extension to handle overlapping speech and simultaneous speakers
- Real-time implementation with streaming audio processing
- Integration with automatic speech recognition (ASR) systems

## Citation

If you use this work in your research, please cite:

```bibtex
@software{chmm_speaker_diarization,
  title={Novel Conditional Hidden Markov Model Algorithm for Multi-Speaker Audio Diarization and Voice Activity Detection},
  author={Balkrishnan, Mrityunjay},
  year={2024},
  url={https://github.com/Mj2603/chmm-speaker-diarization}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Contact

For questions or collaborations, please contact [your-email@domain.com] or open an issue on GitHub.
