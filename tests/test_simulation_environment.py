"""End-to-end tests for the completed Weeks 5-8 simulation environment."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.check_simulation_scale import run_scale_check
from scripts.run_expert import build_simulator
from scripts.run_simulation_demo import run_demo
from src.simulation.costs import (
    ProportionalTransactionCost,
    validate_transaction_cost_grid,
)
from src.simulation.diagnostics import (
    DiagnosticThresholds,
    empirical_transition_matrix,
    evaluate_simulation,
    expected_regime_frequencies,
)
from src.simulation.presets import (
    CANONICAL_REGIME_NAMES,
    canonical_three_regime_preset,
    constant_correlation_covariance,
    load_simulation_preset,
)


class TestCanonicalThreeRegimePreset:
    @pytest.mark.parametrize("n_assets", [1, 50, 100, 200])
    def test_preset_scales_to_project_asset_universes(self, n_assets):
        preset = canonical_three_regime_preset(n_assets)

        assert preset.regime_names == CANONICAL_REGIME_NAMES
        assert preset.transition_matrix.shape == (3, 3)
        assert preset.regime_means.shape == (3, n_assets)
        assert preset.regime_covariances.shape == (3, n_assets, n_assets)
        np.testing.assert_allclose(preset.transition_matrix.sum(axis=1), 1.0)
        assert np.all(np.linalg.eigvalsh(preset.regime_covariances) >= -1e-12)

    def test_regime_parameters_worsen_from_calm_to_stressed(self):
        preset = canonical_three_regime_preset(8)
        average_means = preset.regime_means.mean(axis=1)
        average_volatilities = np.sqrt(
            np.diagonal(preset.regime_covariances, axis1=1, axis2=2)
        ).mean(axis=1)

        assert average_means[0] > average_means[1] > average_means[2]
        assert average_volatilities[0] < average_volatilities[1] < average_volatilities[2]
        assert np.diag(preset.transition_matrix).min() >= 0.80

    def test_unknown_preset_is_rejected(self):
        with pytest.raises(ValueError, match="unknown simulation preset"):
            load_simulation_preset("not-a-preset", n_assets=4)

    def test_expert_pipeline_accepts_preset_without_duplicate_parameters(self):
        simulator = build_simulator(
            {
                "preset": "canonical_three_regime",
                "n_assets": 4,
            }
        )

        assert simulator.n_assets == 4
        assert simulator.regime_model.n_regimes == 3

        with pytest.raises(ValueError, match="cannot be combined"):
            build_simulator(
                {
                    "preset": "canonical_three_regime",
                    "n_assets": 4,
                    "n_regimes": 3,
                }
            )

    def test_expert_pipeline_treats_null_preset_as_explicit_configuration(self):
        simulator = build_simulator(
            {
                "preset": None,
                "n_assets": 2,
                "n_regimes": 1,
                "transition_matrix": [[1.0]],
                "initial_distribution": [1.0],
                "regime_means": [[0.0, 0.0]],
                "regime_covs": [[[0.01, 0.0], [0.0, 0.02]]],
            }
        )

        assert simulator.n_assets == 2
        assert simulator.regime_model.n_regimes == 1

    def test_constant_correlation_constructor_validates_psd_range(self):
        covariance = constant_correlation_covariance(
            np.array([0.1, 0.2, 0.3]),
            correlation=-0.5,
        )
        assert np.linalg.eigvalsh(covariance).min() >= -1e-12

        with pytest.raises(ValueError, match="too negative"):
            constant_correlation_covariance(
                np.array([0.1, 0.2, 0.3]),
                correlation=-0.51,
            )


class TestTransactionCosts:
    def test_cost_matches_expose_full_l1_definition(self):
        model = ProportionalTransactionCost(rate=0.0025)
        previous = np.array([0.5, 0.3, 0.2])
        current = np.array([0.2, 0.5, 0.3])

        assert model.turnover(current, previous) == pytest.approx(0.6)
        assert model.cost(current, previous) == pytest.approx(0.0015)
        assert model.net_return(0.01, current, previous) == pytest.approx(0.0085)

    def test_cost_model_vectorises_over_portfolios(self):
        model = ProportionalTransactionCost(rate=0.001)
        previous = np.array([[0.5, 0.5], [0.2, 0.8]])
        current = np.array([[0.6, 0.4], [0.7, 0.3]])

        np.testing.assert_allclose(model.turnover(current, previous), [0.2, 1.0])
        np.testing.assert_allclose(
            model.net_return(np.array([0.01, -0.01]), current, previous),
            [0.0098, -0.011],
        )

        with pytest.raises(ValueError, match="exactly match"):
            model.net_return(np.ones((2, 1)), current, previous)

    def test_experiment_grid_enforces_prescribed_range_and_order(self):
        assert validate_transaction_cost_grid([0.001, 0.0025, 0.005]) == (
            0.001,
            0.0025,
            0.005,
        )

        for invalid in ([0.0009], [0.0051], [0.002, 0.001], [0.001, 0.001]):
            with pytest.raises(ValueError):
                validate_transaction_cost_grid(invalid)


class TestSimulationDiagnostics:
    def test_expected_frequencies_respect_initial_distribution_and_horizon(self):
        transition_matrix = np.array([[0.5, 0.5], [0.0, 1.0]])
        initial_distribution = np.array([1.0, 0.0])

        frequencies = expected_regime_frequencies(
            transition_matrix,
            initial_distribution,
            periods=3,
        )

        np.testing.assert_allclose(frequencies, [1.75 / 3.0, 1.25 / 3.0])

    def test_empirical_transition_matrix_matches_known_path(self):
        regimes = np.array([0, 0, 1, 1, 0, 1])

        matrix = empirical_transition_matrix(regimes, n_regimes=2)

        np.testing.assert_allclose(matrix, [[1.0 / 3.0, 2.0 / 3.0], [0.5, 0.5]])

    def test_canonical_simulation_passes_quantitative_diagnostics(self):
        preset = canonical_three_regime_preset(6)
        returns, regimes = preset.build_simulator().simulate(T=10_000, seed=42)

        diagnostics = evaluate_simulation(
            returns,
            regimes,
            regime_names=preset.regime_names,
            transition_matrix=preset.transition_matrix,
            initial_distribution=preset.initial_distribution,
            target_means=preset.regime_means,
            target_covariances=preset.regime_covariances,
        )

        assert diagnostics.passed, diagnostics.to_dict()
        assert set(diagnostics.checks) == {
            "regime_frequencies",
            "regime_transitions_and_persistence",
            "regime_means",
            "regime_variances",
            "regime_covariances",
        }

    def test_diagnostics_reject_invalid_target_covariance(self):
        preset = canonical_three_regime_preset(3)
        returns, regimes = preset.build_simulator().simulate(T=600, seed=42)
        asymmetric_covariances = preset.regime_covariances.copy()
        asymmetric_covariances[0, 0, 1] += 0.001

        with pytest.raises(ValueError, match="must be symmetric"):
            evaluate_simulation(
                returns,
                regimes,
                regime_names=preset.regime_names,
                transition_matrix=preset.transition_matrix,
                initial_distribution=preset.initial_distribution,
                target_means=preset.regime_means,
                target_covariances=asymmetric_covariances,
            )

    def test_demo_writes_visible_and_machine_readable_outputs(self, tmp_path):
        cfg = {
            "simulation": {
                "preset": "canonical_three_regime",
                "n_assets": 4,
                "T": 600,
                "seed": 7,
                "transaction_costs": [0.001, 0.0025, 0.005],
            },
            "diagnostics": {
                "max_frequency_error": 1.0,
                "max_transition_error": 1.0,
                "max_mean_z_score": 100.0,
                "max_variance_standardized_error": 100.0,
                "max_covariance_error_ratio": 100.0,
            },
            "final_scale": {"n_assets": [50, 100, 200], "T": 10_000},
        }

        diagnostics = run_demo(cfg, tmp_path)

        assert diagnostics.passed
        assert {
            "returns.csv",
            "regimes.csv",
            "transaction_cost_examples.csv",
            "diagnostics.json",
            "simulation_dashboard.png",
        }.issubset(path.name for path in tmp_path.iterdir())
        report = json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))
        assert report["simulation"]["regime_names"] == list(CANONICAL_REGIME_NAMES)
        assert report["diagnostics"]["passed"] is True
        assert (tmp_path / "simulation_dashboard.png").stat().st_size > 10_000

    def test_final_scale_config_runs_all_required_asset_universes(self):
        cfg = {
            "simulation": {
                "preset": "canonical_three_regime",
                "seed": 42,
            },
            "final_scale": {"n_assets": [50, 100, 200], "T": 10_000},
        }

        results = run_scale_check(cfg)

        assert [result["n_assets"] for result in results] == [50, 100, 200]
        assert all(result["returns_rows"] == 10_000 for result in results)
        assert all(
            result["returns_columns"] == result["n_assets"] for result in results
        )

    def test_thresholds_reject_nonpositive_values(self):
        with pytest.raises(ValueError, match="max_frequency_error"):
            DiagnosticThresholds(max_frequency_error=0.0)
