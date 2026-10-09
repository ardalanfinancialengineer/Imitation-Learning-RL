"""Tests for the config-driven expert demonstration runner."""

from __future__ import annotations

import json
import sys
from textwrap import dedent

import numpy as np
import pandas as pd

from scripts import run_expert


def _write_config(path):
    path.write_text(
        dedent(
            """
            simulation:
              n_assets: 3
              T: 12
              n_regimes: 2
              seed: 123
              transition_matrix:
                - [0.90, 0.10]
                - [0.20, 0.80]
              initial_distribution: [0.60, 0.40]
              regime_means:
                - [0.0010, 0.0015, 0.0020]
                - [-0.0010, 0.0000, 0.0010]
              covariance:
                method: constant_correlation
                volatilities:
                  - [0.010, 0.012, 0.014]
                  - [0.020, 0.025, 0.030]
                correlations: [0.20, 0.35]

            expert:
              window: 4
              risk_aversion: 1.0
              long_only: true
              max_weight: 0.60

            paths:
              expert_output_dir: data/unused
            """
        ),
        encoding="utf-8",
    )


def _run_with_config(monkeypatch, config_path, output_dir):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_expert.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
    )
    run_expert.main()


def test_run_expert_writes_expected_files_and_shapes(tmp_path, monkeypatch):
    config_path = tmp_path / "base.yaml"
    output_dir = tmp_path / "expert"
    _write_config(config_path)

    _run_with_config(monkeypatch, config_path, output_dir)

    expected_files = {
        "returns.csv",
        "regimes.csv",
        "expert_actions.csv",
        "demonstrations.npz",
        "metadata.json",
    }
    assert expected_files.issubset({path.name for path in output_dir.iterdir()})

    returns = pd.read_csv(output_dir / "returns.csv")
    regimes = pd.read_csv(output_dir / "regimes.csv")
    actions = pd.read_csv(output_dir / "expert_actions.csv")
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

    assert returns.shape == (12, 3)
    assert regimes.shape == (12, 2)
    assert actions.shape == (8, 4)
    assert actions["decision_t"].tolist() == list(range(4, 12))
    assert metadata["seed"] == 123
    assert metadata["observations_shape"] == [8, 4, 3]
    assert metadata["actions_shape"] == [8, 3]
    assert metadata["diagnostic_only"] == ["regimes"]

    with np.load(output_dir / "demonstrations.npz") as data:
        assert data["observations"].shape == (8, 4, 3)
        assert data["actions"].shape == (8, 3)
        assert data["returns"].shape == (12, 3)
        assert data["regimes"].shape == (12,)
        np.testing.assert_array_equal(data["decision_times"], np.arange(4, 12))
        np.testing.assert_allclose(data["actions"].sum(axis=1), np.ones(8), atol=1e-6)
        assert np.all(data["actions"] >= -1e-8)
        assert np.all(data["actions"] <= 0.60 + 1e-6)


def test_run_expert_is_reproducible_for_fixed_seed(tmp_path, monkeypatch):
    config_path = tmp_path / "base.yaml"
    output_a = tmp_path / "expert_a"
    output_b = tmp_path / "expert_b"
    _write_config(config_path)

    _run_with_config(monkeypatch, config_path, output_a)
    _run_with_config(monkeypatch, config_path, output_b)

    with np.load(output_a / "demonstrations.npz") as data_a:
        arrays_a = {key: data_a[key].copy() for key in data_a.files}
    with np.load(output_b / "demonstrations.npz") as data_b:
        arrays_b = {key: data_b[key].copy() for key in data_b.files}

    for key, value in arrays_a.items():
        np.testing.assert_allclose(value, arrays_b[key])
