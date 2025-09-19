from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class CHMM:
    n_states: int
    n_features: int
    transition_weights: np.ndarray  # (n_states, n_states, n_features)
    emission_means: np.ndarray      # (n_states, n_features)
    emission_covs: np.ndarray       # (n_states, n_features, n_features)
    
    def __post_init__(self):
        # Ensure positive definite covariances
        for i in range(self.n_states):
            self.emission_covs[i] = self.emission_covs[i] + 0.01 * np.eye(self.n_features)
    
    def conditional_transition(self, features: np.ndarray) -> np.ndarray:
        """Compute transition matrix conditioned on features"""
        # features: (n_features,)
        # transition_weights: (n_states, n_states, n_features)
        logits = np.einsum('ijk,k->ij', self.transition_weights, features)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def emission_log_prob(self, features: np.ndarray) -> np.ndarray:
        """Compute emission log probabilities for each state"""
        # features: (n_features,)
        # returns: (n_states,)
        log_probs = np.zeros(self.n_states)
        for i in range(self.n_states):
            diff = features - self.emission_means[i]
            inv_cov = np.linalg.inv(self.emission_covs[i])
            log_probs[i] = -0.5 * (diff @ inv_cov @ diff + 
                                  np.log(np.linalg.det(2 * np.pi * self.emission_covs[i])))
        return log_probs
    
    def viterbi(self, observations: np.ndarray) -> np.ndarray:
        """Viterbi algorithm for most likely state sequence"""
        T, n_features = observations.shape
        log_delta = np.full((T, self.n_states), -np.inf)
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Initialize
        log_delta[0] = self.emission_log_prob(observations[0])
        
        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                trans_probs = self.conditional_transition(observations[t-1])
                log_trans = np.log(trans_probs[:, j] + 1e-10)
                log_delta[t, j] = self.emission_log_prob(observations[t])[j] + np.max(log_delta[t-1] + log_trans)
                psi[t, j] = np.argmax(log_delta[t-1] + log_trans)
        
        # Backward pass
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(log_delta[-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states


def create_synthetic_chmm(n_states: int = 3, n_features: int = 13, seed: int = 42) -> CHMM:
    """Create a synthetic CHMM for demonstration"""
    rng = np.random.default_rng(seed)
    
    # Random transition weights
    transition_weights = rng.normal(0, 1, (n_states, n_states, n_features))
    
    # Distinct emission parameters for each state
    emission_means = rng.normal(0, 2, (n_states, n_features))
    emission_covs = np.array([np.eye(n_features) + 0.5 * rng.random((n_features, n_features)) 
                             for _ in range(n_states)])
    
    return CHMM(n_states, n_features, transition_weights, emission_means, emission_covs)
