from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CHMM:
    """Conditional-transition HMM with Gaussian state emissions.

    This class decodes observations from supplied parameters. It does not fit
    those parameters; callers must provide a trained model before using it for
    real diarization.
    """

    n_states: int
    n_features: int
    transition_weights: np.ndarray  # (n_states, n_states, n_features)
    emission_means: np.ndarray  # (n_states, n_features)
    emission_covs: np.ndarray  # (n_states, n_features, n_features)
    _cholesky: tuple[np.ndarray, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.n_states, int) or not isinstance(self.n_features, int):
            raise ValueError("n_states and n_features must be integers")
        if self.n_states < 1 or self.n_features < 1:
            raise ValueError("n_states and n_features must be positive")

        self.transition_weights = np.array(self.transition_weights, dtype=float, copy=True)
        self.emission_means = np.array(self.emission_means, dtype=float, copy=True)
        self.emission_covs = np.array(self.emission_covs, dtype=float, copy=True)

        expected_shapes = (
            ("transition_weights", self.transition_weights, (self.n_states, self.n_states, self.n_features)),
            ("emission_means", self.emission_means, (self.n_states, self.n_features)),
            ("emission_covs", self.emission_covs, (self.n_states, self.n_features, self.n_features)),
        )
        for name, value, expected in expected_shapes:
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")

        # Symmetrize numerical noise, regularize lightly, and validate that
        # every covariance can be factored before decoding starts.
        self.emission_covs = 0.5 * (
            self.emission_covs + np.swapaxes(self.emission_covs, 1, 2)
        )
        self.emission_covs = self.emission_covs + 1e-6 * np.eye(self.n_features)[None, :, :]
        factors = []
        for covariance in self.emission_covs:
            try:
                factors.append(np.linalg.cholesky(covariance))
            except np.linalg.LinAlgError as exc:
                raise ValueError("emission covariances must be positive definite") from exc
        self._cholesky = tuple(factors)

        # Parameters are immutable after validation so cached factors stay valid.
        for array in (self.transition_weights, self.emission_means, self.emission_covs):
            array.setflags(write=False)
        for factor in self._cholesky:
            factor.setflags(write=False)

    def conditional_transition(self, features: np.ndarray) -> np.ndarray:
        """Return row-normalized transition probabilities for one feature vector."""
        features = np.asarray(features, dtype=float)
        if features.shape != (self.n_features,) or not np.isfinite(features).all():
            raise ValueError(f"features must be a finite vector of length {self.n_features}")

        with np.errstate(over="ignore", invalid="ignore"):
            logits = np.einsum("ijk,k->ij", self.transition_weights, features)
        if not np.isfinite(logits).all():
            raise ValueError("transition logits overflowed; rescale features or parameters")
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def emission_log_prob(self, features: np.ndarray) -> np.ndarray:
        """Return Gaussian emission log probabilities, one per state."""
        features = np.asarray(features, dtype=float)
        if features.shape != (self.n_features,) or not np.isfinite(features).all():
            raise ValueError(f"features must be a finite vector of length {self.n_features}")

        log_probs = np.empty(self.n_states, dtype=float)
        constant = self.n_features * np.log(2.0 * np.pi)
        for state in range(self.n_states):
            chol = self._cholesky[state]
            diff = features - self.emission_means[state]
            with np.errstate(over="ignore", invalid="ignore"):
                whitened = np.linalg.solve(chol, diff)
                quadratic = whitened @ whitened
            log_det = 2.0 * np.log(np.diag(chol)).sum()
            log_probs[state] = -0.5 * (quadratic + log_det + constant)
        if not np.isfinite(log_probs).all():
            raise ValueError("emission likelihood overflowed; rescale features")
        return log_probs

    def viterbi(self, observations: np.ndarray) -> np.ndarray:
        """Decode the maximum-probability state path for a non-empty sequence."""
        observations = np.asarray(observations, dtype=float)
        if observations.ndim != 2 or observations.shape[1] != self.n_features:
            raise ValueError(f"observations must have shape (time, {self.n_features})")
        if observations.shape[0] == 0:
            raise ValueError("observations must contain at least one frame")
        if not np.isfinite(observations).all():
            raise ValueError("observations must contain only finite values")

        n_frames = observations.shape[0]
        log_emissions = np.vstack([self.emission_log_prob(row) for row in observations])
        log_delta = np.full((n_frames, self.n_states), -np.inf)
        backpointers = np.zeros((n_frames, self.n_states), dtype=np.int64)

        # Uniform initial-state prior.
        log_delta[0] = log_emissions[0] - np.log(self.n_states)
        for t in range(1, n_frames):
            transition = self.conditional_transition(observations[t - 1])
            log_transition = np.log(np.maximum(transition, np.finfo(float).tiny))
            candidates = log_delta[t - 1][:, None] + log_transition
            backpointers[t] = np.argmax(candidates, axis=0)
            log_delta[t] = log_emissions[t] + np.max(candidates, axis=0)

        states = np.empty(n_frames, dtype=np.int64)
        states[-1] = int(np.argmax(log_delta[-1]))
        for t in range(n_frames - 2, -1, -1):
            states[t] = backpointers[t + 1, states[t + 1]]
        return states


def create_synthetic_chmm(n_states: int = 3, n_features: int = 13, seed: int = 42) -> CHMM:
    """Create reproducible synthetic parameters for decoder demonstrations only."""
    if n_states < 1 or n_features < 1:
        raise ValueError("n_states and n_features must be positive")

    rng = np.random.default_rng(seed)
    transition_weights = rng.normal(0, 1, (n_states, n_states, n_features))
    emission_means = rng.normal(0, 2, (n_states, n_features))
    emission_covs = []
    for _ in range(n_states):
        factor = rng.normal(size=(n_features, n_features))
        emission_covs.append(factor @ factor.T / n_features + 0.5 * np.eye(n_features))

    return CHMM(
        n_states,
        n_features,
        transition_weights,
        emission_means,
        np.asarray(emission_covs),
    )
