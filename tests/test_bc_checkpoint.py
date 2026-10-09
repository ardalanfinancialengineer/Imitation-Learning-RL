"""Tests for the BC trainer's mid-training checkpoint and resume.

The load-bearing property is **resume equivalence**: a run interrupted at epoch
K and resumed must land bit-for-bit where an uninterrupted run of the same length
would. That only holds if the optimiser moments, the cosine-schedule position,
and every RNG stream (including the DataLoader shuffle generator) are restored —
so this test is the real guard on the checkpoint format, not a smoke test.
"""

from __future__ import annotations

import pytest
import torch

from src.dataset.demonstrations import DemonstrationDataset
from src.dataset.paths import generate_path
from src.dataset.state import StateSpec
from src.expert.constraints import PortfolioConstraints
from src.imitation.bc import (
    BCConfig,
    BehaviouralCloningTrainer,
    seed_everything,
)
from src.policy.builder import build_policy
from src.simulation.presets import canonical_three_regime_preset

N_ASSETS = 4
N_REGIMES = 3
SPEC = StateSpec(lookback=10)
CONSTRAINTS = PortfolioConstraints(
    long_only=True, min_weight=0.0, max_weight=0.5, leverage_limit=1.0
)
EXPERT_CONFIG = {
    "horizon": 5,
    "risk_aversion": 5.0,
    "transaction_cost": 0.0025,
    "constraints": {
        "long_only": True,
        "min_weight": 0.0,
        "max_weight": 0.5,
        "leverage_limit": 1.0,
    },
    "moments": {"estimator": "rolling", "covariance": "ledoit_wolf", "window": 60},
}
TINY_POLICY = {"kind": "mlp", "hidden_dimensions": [32], "dropout": 0.0}

# High patience so early stopping never fires within the short test runs — that
# keeps "the remaining epochs" unambiguous for the equivalence comparison.
_EPOCHS = 6
_SPLIT = 3


@pytest.fixture(scope="module")
def datasets() -> dict[str, DemonstrationDataset]:
    preset = canonical_three_regime_preset(N_ASSETS)
    paths = [
        generate_path(
            path_id=index,
            seed=100 + index,
            preset=preset,
            periods=260,
            expert_config=EXPERT_CONFIG,
        )
        for index in range(3)
    ]
    return {
        "train": DemonstrationDataset(paths[:2], spec=SPEC),
        "validation": DemonstrationDataset(paths[2:], spec=SPEC),
    }


def _make_policy():
    return build_policy(
        dict(TINY_POLICY),
        spec=SPEC,
        n_assets=N_ASSETS,
        n_regimes=N_REGIMES,
        constraints=CONSTRAINTS,
    )


def _config(**overrides) -> BCConfig:
    base = dict(epochs=_EPOCHS, batch_size=64, seed=7, patience=_EPOCHS + 1)
    base.update(overrides)
    return BCConfig(**base)


def test_checkpoint_round_trips_the_resume_fields(datasets, tmp_path):
    seed_everything(7)
    trainer = BehaviouralCloningTrainer(_make_policy(), _config())
    ckpt = tmp_path / "run.fit.pt"
    trainer.fit(
        datasets["train"],
        datasets["validation"],
        checkpoint_path=ckpt,
        checkpoint_every=2,
    )
    assert ckpt.exists()
    payload = torch.load(ckpt, weights_only=False)
    assert payload["format"] == "bc-fit-checkpoint-v1"
    for key in (
        "epoch",
        "policy_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "best_state",
        "since_improvement",
        "records",
        "bc_config",
    ):
        assert key in payload
    for stream in ("python", "numpy", "torch", "generator"):
        assert stream in payload["rng"]


def test_resume_equivalence_is_bit_identical(datasets, tmp_path):
    """A split-then-resumed run matches an uninterrupted run bit-for-bit.

    The interruption is simulated the way it actually happens — the process dies
    mid-run *after* a checkpoint has been written — rather than by shortening the
    config. That matters: the cosine LR schedule's ``T_max`` is the full epoch
    count, so the first leg must run under the same ``BCConfig`` as the reference,
    not a truncated one.
    """

    # Uninterrupted reference.
    seed_everything(7)
    reference = BehaviouralCloningTrainer(_make_policy(), _config())
    ref_history = reference.fit(datasets["train"], datasets["validation"])
    ref_state = reference.policy.state_dict()

    # First leg: run under the full config, checkpoint at the split, then die.
    class _Interrupt(RuntimeError):
        pass

    ckpt = tmp_path / "run.fit.pt"
    seed_everything(7)
    first = BehaviouralCloningTrainer(_make_policy(), _config())
    real_train_epoch = first._train_epoch
    calls = {"n": 0}

    def interrupting_train_epoch(loader, optimiser):
        # After the checkpoint at epoch _SPLIT has been written, the next epoch's
        # first act blows up — as an OOM kill would.
        if calls["n"] >= _SPLIT:
            raise _Interrupt
        calls["n"] += 1
        return real_train_epoch(loader, optimiser)

    first._train_epoch = interrupting_train_epoch  # type: ignore[method-assign]
    with pytest.raises(_Interrupt):
        first.fit(
            datasets["train"],
            datasets["validation"],
            checkpoint_path=ckpt,
            checkpoint_every=_SPLIT,
        )
    assert ckpt.exists()

    # Second leg: a fresh trainer resumes and runs to the full epoch count.
    seed_everything(999)  # deliberately different — resume must override it
    resumed = BehaviouralCloningTrainer(_make_policy(), _config())
    res_history = resumed.fit(
        datasets["train"],
        datasets["validation"],
        checkpoint_path=ckpt,
        resume=True,
    )

    assert len(res_history.records) == _EPOCHS
    assert res_history.best_tracking_error == pytest.approx(
        ref_history.best_tracking_error, rel=0, abs=0
    )
    res_state = resumed.policy.state_dict()
    assert ref_state.keys() == res_state.keys()
    for key in ref_state:
        assert torch.equal(ref_state[key], res_state[key]), f"mismatch in {key}"


def test_resume_config_mismatch_raises(datasets, tmp_path):
    ckpt = tmp_path / "run.fit.pt"
    seed_everything(7)
    first = BehaviouralCloningTrainer(_make_policy(), _config())
    first.fit(
        datasets["train"],
        datasets["validation"],
        checkpoint_path=ckpt,
        checkpoint_every=_SPLIT,
    )

    seed_everything(7)
    # Different learning rate -> different BCConfig -> resume must refuse.
    mismatched = BehaviouralCloningTrainer(
        _make_policy(), _config(learning_rate=0.5)
    )
    with pytest.raises(ValueError, match="different BCConfig"):
        mismatched.fit(
            datasets["train"],
            datasets["validation"],
            checkpoint_path=ckpt,
            resume=True,
        )
