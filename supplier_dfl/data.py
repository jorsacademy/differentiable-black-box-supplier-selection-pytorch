from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ContextualSourcingData:
    features: np.ndarray
    objective_costs: np.ndarray


def generate_contextual_sourcing_data(
    n_samples: int,
    *,
    seed: int = 42,
    feature_dim: int = 10,
) -> ContextualSourcingData:
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, feature_dim))

    # 12 allocation coefficients: supplier/product-specific nonlinear responses.
    base_alloc = np.array([
        31, 35, 39,
        34, 32, 37,
        38, 36, 33,
        40, 41, 38,
    ], dtype=float)
    linear = rng.normal(0.0, 2.8, size=(feature_dim, 12))
    latent = x @ linear
    nonlinear = np.empty((n_samples, 12))
    for j in range(12):
        a = j % feature_dim
        b = (j * 3 + 2) % feature_dim
        nonlinear[:, j] = (
            4.5 * np.tanh(0.8 * x[:, a])
            + 2.8 * np.sin(x[:, b])
            + 1.5 * x[:, a] * x[:, b]
        )

    # Common supplier shocks make allocation choices context-dependent.
    supplier_shock = np.column_stack([
        5.0 * np.tanh(x[:, 0] + 0.4 * x[:, 6]),
        4.5 * np.sin(x[:, 1] - 0.3 * x[:, 7]),
        5.5 * np.tanh(x[:, 2] * x[:, 8]),
        4.0 * np.sin(x[:, 3] + x[:, 9]),
    ])
    supplier_shock = np.repeat(supplier_shock, 3, axis=1)
    alloc = base_alloc + latent + nonlinear + supplier_shock + rng.normal(0.0, 1.3, size=(n_samples, 12))
    alloc = np.clip(alloc, 8.0, None)

    # 4 fixed-charge coefficients are also contextual.
    base_activation = np.array([260, 230, 205, 175], dtype=float)
    act = np.empty((n_samples, 4))
    for s in range(4):
        act[:, s] = (
            base_activation[s]
            + 35.0 * np.tanh(x[:, (s+4) % feature_dim])
            + 18.0 * np.sin(x[:, (s+7) % feature_dim])
            + 12.0 * x[:, s] * x[:, (s+5) % feature_dim]
            + rng.normal(0.0, 7.0, size=n_samples)
        )
    act = np.clip(act, 60.0, None)

    return ContextualSourcingData(
        features=x.astype(np.float32),
        objective_costs=np.c_[alloc, act].astype(np.float32),
    )
