#!/usr/bin/env python
"""
Generate a synthetic dataframe for card population scoring.

- ID: 101, 102, ..., 999
- ts: time since snapshot, 1..18 (most IDs have full 18 points, some shorter)
- x1..x4: features roughly scaled so that
    linear_score = 1.2*x1 + 0.4*x2 + 0.9*x3 + 1.4*x4
  has a typical logit around -6.5, implying PD ≈ 0.0015 if passed through sigmoid.
"""

import numpy as np
import pandas as pd


def generate_population(seed: int = 42,
                             mu: float = -1.7,
                             sigma: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # IDs 101..999
    ids = np.arange(101, 1000)
    id_list = []
    ts_list = []

    for cid in ids:
        # Majority of IDs have full 18 time steps
        if rng.random() < 0.8:
            max_ts = 18
        else:
            # Some IDs have shorter histories (between 5 and 17)
            max_ts = int(rng.integers(5, 18))

        ts_vals = np.arange(1, max_ts + 1)
        id_list.extend([cid] * len(ts_vals))
        ts_list.extend(ts_vals)

    n_rows = len(id_list)

    # Features:
    # Choose x1..x4 ~ Normal(mu, sigma) with mu ≈ -1.7 so that
    # E[linear_score] ≈ -6.5 -> PD ≈ 0.0015 via sigmoid
    # mu = -1.7
    # sigma = 1.0

    x1 = rng.normal(mu, sigma, size=n_rows)
    x2 = rng.normal(mu, sigma, size=n_rows)
    x3 = rng.normal(mu, sigma, size=n_rows)
    x4 = rng.normal(mu, sigma, size=n_rows)

    #linear_score = 1.2 * x1 + 0.4 * x2 + 0.9 * x3 + 1.4 * x4

    df = pd.DataFrame(
        {
            "ID": id_list,
            "ts": ts_list,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
    #        "linear_score": linear_score,
        }
    )

    return df

def generate_card_population():
    df = generate_population(seed=42, mu=-1.7, sigma=1.0)
    return df

def generate_ploc_population():
    df = generate_population(seed=35, mu=-1.8, sigma=1.5)
    return df


if __name__ == "__main__":
    df = generate_card_population()
    print(df.head())
    print(df.describe())