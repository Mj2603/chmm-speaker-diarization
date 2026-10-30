import numpy as np
import pytest

from src.chmm import CHMM, create_synthetic_chmm
from src.diarization import evaluate_diarization


def model():
    return CHMM(
        n_states=2,
        n_features=1,
        transition_weights=np.zeros((2, 2, 1)),
        emission_means=np.array([[0.0], [5.0]]),
        emission_covs=np.array([[[1.0]], [[1.0]]]),
    )


def test_transition_rows_are_normalized_and_viterbi_uses_emissions():
    chmm = model()
    probabilities = chmm.conditional_transition(np.array([0.25]))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    states = chmm.viterbi(np.array([[0.0], [0.1], [4.9], [5.0]]))
    assert states.tolist() == [0, 0, 1, 1]


def test_decoder_rejects_empty_wrong_shape_and_nonfinite_input():
    chmm = model()
    with pytest.raises(ValueError, match="at least one"):
        chmm.viterbi(np.empty((0, 1)))
    with pytest.raises(ValueError, match="shape"):
        chmm.viterbi(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="finite"):
        chmm.viterbi(np.array([[np.nan]]))


def test_model_rejects_invalid_covariances_and_parameters():
    with pytest.raises(ValueError, match="positive definite"):
        CHMM(
            1, 2, np.zeros((1, 1, 2)), np.zeros((1, 2)),
            np.array([[[1.0, 2.0], [2.0, 1.0]]]),
        )
    with pytest.raises(ValueError, match="shape"):
        CHMM(2, 1, np.zeros((2, 2, 2)), np.zeros((2, 1)), np.array([[[1.0]], [[1.0]]]))


def test_synthetic_model_is_reproducible_and_parameters_are_read_only():
    first = create_synthetic_chmm(n_states=2, n_features=3, seed=4)
    second = create_synthetic_chmm(n_states=2, n_features=3, seed=4)
    assert np.array_equal(first.transition_weights, second.transition_weights)
    with pytest.raises(ValueError):
        first.emission_means[0, 0] = 2.0


def test_evaluation_requires_aligned_nonempty_labels():
    with pytest.raises(ValueError, match="equal length"):
        evaluate_diarization(np.array([0, 1]), np.array([0]))
