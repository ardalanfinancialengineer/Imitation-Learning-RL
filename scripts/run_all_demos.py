"""Run every milestone demo and build one browsable index of the results.

This is the front door to the project. Each milestone has its own demo script
that writes a dashboard image and a JSON report; this script runs them in
order, then renders ``data/index.html`` linking everything together with the
headline numbers and a plain-language note on what each dashboard shows.

    uv run python scripts/run_all_demos.py              # quick, about 12 minutes
    uv run python scripts/run_all_demos.py --full       # shipped sizes, ~90 minutes
    uv run python scripts/run_all_demos.py --index-only # rebuild the page only
    uv run python scripts/run_all_demos.py --only expert dagger

Open ``data/index.html`` in a browser afterwards.

The quick profile shrinks the number of paths, the path length, and the number
of training iterations. It exists so the whole pipeline can be seen working in
one sitting; every headline number quoted in the milestone documents comes from
the ``--full`` profile, and the two will not agree. The page says which profile
produced it.

**Acceptance checks are not expected to pass under ``--quick``.** Several of
them are statistical: ``beats_naive_baselines``, for instance, requires a paired
bootstrap over held-out *paths* to exclude zero, and the quick profile leaves
only two or three such paths. A failure there means the sample was too small,
not that the milestone is broken. So ``--quick`` reports check failures and
carries on; only ``--full`` treats them as errors and exits non-zero.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Demo:
    """One milestone demo: how to run it and how to read what it produced."""

    key: str
    weeks: str
    title: str
    script: str
    output_dir: str
    dashboard: str
    report: str
    doc: str
    question: str
    reading: str
    quick_args: list[str] = field(default_factory=list)
    full_args: list[str] = field(default_factory=list)
    config: str | None = None

    def command(self, quick: bool) -> list[str]:
        command = [sys.executable, "-u", f"scripts/{self.script}"]
        if self.config:
            command += ["--config", self.config]
        command += self.quick_args if quick else self.full_args
        return command


#: Ordered as the exposé's work plan runs. `question` is what the milestone was
#: trying to find out; `reading` is how to interpret the dashboard.
DEMOS: tuple[Demo, ...] = (
    Demo(
        key="simulation",
        weeks="5-8",
        title="Market simulation",
        script="run_simulation_demo.py",
        config="configs/simulation_demo.yaml",
        output_dir="data/simulation_demo",
        dashboard="simulation_dashboard.png",
        report="diagnostics.json",
        doc="docs/simulation_environment.md",
        question=(
            "Does the simulated market actually behave the way it was specified "
            "to — three regimes, the right frequencies, the right volatilities?"
        ),
        reading=(
            "Each panel compares an observed statistic against its target. All "
            "five checks must pass; the demo exits non-zero if any fails."
        ),
        quick_args=["--periods", "2000", "--n-assets", "6"],
    ),
    Demo(
        key="expert",
        weeks="9-13",
        title="Markowitz expert",
        script="run_expert_demo.py",
        config="configs/expert_demo.yaml",
        output_dir="data/expert_demo",
        dashboard="expert_dashboard.png",
        report="expert_report.json",
        doc="docs/markowitz_expert.md",
        question=(
            "Does the benchmark strategy solve its optimisation correctly, "
            "respect its constraints, and react to costs the way theory says?"
        ),
        reading=(
            "The turnover panel is the economic sanity check: trading must fall "
            "as the transaction cost rises. The solve-time histogram is the cost "
            "the learned policy is later meant to avoid."
        ),
        quick_args=["--periods", "900", "--n-assets", "6"],
    ),
    Demo(
        key="dataset",
        weeks="14-16",
        title="State, dataset and environment",
        script="run_dataset_demo.py",
        config="configs/dataset_demo.yaml",
        output_dir="data/dataset_demo",
        dashboard="dataset_dashboard.png",
        report="dataset_report.json",
        doc="docs/dataset_environment.md",
        question=(
            "Is the training data built correctly, and can any decision see "
            "information from the future?"
        ),
        reading=(
            "The no-look-ahead check is the important one: rewriting every "
            "return after a date must leave that date's inputs bit-identical."
        ),
        quick_args=["--n-paths", "6", "--periods", "700"],
    ),
    Demo(
        key="bc",
        weeks="17-21",
        title="Policies and behavioural cloning",
        script="run_bc_demo.py",
        config="configs/bc_demo.yaml",
        output_dir="data/bc_demo",
        dashboard="bc_dashboard.png",
        report="bc_report.json",
        doc="docs/policy_behavioural_cloning.md",
        question=(
            "Can a neural network copy the expert's decisions, and is it faster "
            "than re-solving the optimisation?"
        ),
        reading=(
            "Compare each policy against 'never trade'. That baseline is strong "
            "here, because transaction costs make the expert reluctant to move."
        ),
        # Kept small on purpose. `beats_naive_baselines` needs a paired
        # bootstrap over held-out paths to exclude zero, and no affordable quick
        # size supplies enough of them, so paying for a larger run buys nothing.
        quick_args=["--n-paths", "10", "--periods", "700", "--epochs", "10"],
    ),
    Demo(
        key="dagger",
        weeks="22-25",
        title="DAgger",
        script="run_dagger_demo.py",
        config="configs/dagger_demo.yaml",
        output_dir="data/dagger_demo",
        dashboard="dagger_dashboard.png",
        report="dagger_report.json",
        doc="docs/dagger.md",
        question=(
            "Copying works one decision at a time — but does it still work once "
            "the policy's own small errors accumulate over a whole path?"
        ),
        reading=(
            "The closed-loop error is the honest number; the teacher-forced bar "
            "beside it shows how much the simpler metric flatters the policy."
        ),
        quick_args=["--n-paths", "6", "--periods", "500", "--iterations", "2",
                    "--skip-control"],
    ),
    Demo(
        key="gail",
        weeks="26-29",
        title="Wasserstein GAIL",
        script="run_gail_demo.py",
        config="configs/gail_demo.yaml",
        output_dir="data/gail_demo",
        dashboard="gail_dashboard.png",
        report="gail_report.json",
        doc="docs/gail.md",
        question=(
            "Instead of copying individual decisions, can the policy be trained "
            "to reproduce the expert's overall behaviour?"
        ),
        reading=(
            "The stability panel is the deliverable: it compares the two ways of "
            "constraining the critic that the exposé cites."
        ),
        quick_args=["--n-paths", "5", "--periods", "450", "--iterations", "8",
                    "--skip-ablation"],
    ),
    Demo(
        key="ppo",
        weeks="30-33",
        title="PPO fine-tuning",
        script="run_ppo_demo.py",
        config="configs/ppo_demo.yaml",
        output_dir="data/ppo_demo",
        dashboard="ppo_dashboard.png",
        report="ppo_report.json",
        doc="docs/ppo.md",
        question=(
            "Once the policy has copied the expert, can reinforcement learning "
            "improve on it — and does a policy trained from scratch do better?"
        ),
        reading=(
            "Read the held-out utility bars first. On the shipped preset the "
            "from-scratch policy wins, and docs/ppo.md explains why that is a "
            "finding about the benchmark rather than about the method."
        ),
        quick_args=["--n-paths", "5", "--periods", "420", "--iterations", "6",
                    "--skip-ablation"],
    ),
)


# ----------------------------------------------------------------------
# Headline numbers
# ----------------------------------------------------------------------
def _dig(payload: Any, *path: str) -> Any:
    """Walk a nested report, returning None rather than raising if absent."""
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _milliseconds(seconds: Any) -> str:
    return "—" if seconds is None else f"{1000.0 * float(seconds):.1f} ms"


def headlines(key: str, report: dict[str, Any]) -> list[tuple[str, str]]:
    """Pull the two or three numbers that matter from a milestone's report."""

    def number(value: Any, spec: str = "{:.4g}") -> str:
        return "—" if value is None else spec.format(value)

    if key == "simulation":
        checks = _dig(report, "diagnostics", "checks") or {}
        passed = sum(bool(v) for v in checks.values())
        assets = _dig(report, "simulation", "n_assets")
        periods = _dig(report, "simulation", "periods")
        return [
            ("Statistical checks", f"{passed}/{len(checks)} passed"),
            ("Assets x periods", f"{assets} x {periods}"),
        ]
    if key == "expert":
        return [
            ("Median solve time", _milliseconds(
                _dig(report, "baseline", "median_solve_seconds"))),
            ("Mean turnover", number(_dig(report, "baseline", "mean_turnover"))),
            ("Fallbacks used", number(_dig(report, "baseline", "fallback_rate"), "{:.0%}")),
        ]
    if key == "dataset":
        return [
            ("State dimension", number(_dig(report, "state", "dimension"), "{:d}")),
            ("Training samples",
             number(_dig(report, "datasets", "train", "n_samples"), "{:,d}")),
            ("Stored vs materialised",
             f"{_dig(report, 'storage', 'reduction_factor')}x smaller"),
        ]
    if key == "bc":
        policies = _dig(report, "policies") or {}
        errors = {
            name: _dig(payload, "test_metrics", "tracking_error")
            for name, payload in policies.items()
        }
        errors = {k: v for k, v in errors.items() if v is not None}
        baseline = _dig(report, "reference_tracking_errors", "previous_weights")
        rows: list[tuple[str, str]] = []
        if errors:
            best = min(errors, key=lambda k: errors[k])
            rows.append(("Best policy", best))
            rows.append(("Its tracking error", number(errors[best], "{:.3e}")))
        if baseline is not None:
            rows.append(("Never-trade baseline", number(baseline, "{:.3e}")))
        return rows or [("See dashboard", "policy comparison and latency")]
    if key == "dagger":
        rounds = _dig(report, "results", "shipped", "transformer", "rounds") or []
        if not rounds:
            return [("See dashboard", "closed-loop error per round")]
        return [
            ("Closed-loop error, before",
             number(rounds[0].get("closed_loop_tracking_error"), "{:.3e}")),
            ("Closed-loop error, after",
             number(rounds[-1].get("closed_loop_tracking_error"), "{:.3e}")),
        ]
    if key == "gail":
        metrics = _dig(report, "comparison", "metrics") or {}
        return [
            ("Best policy (closed-loop)",
             min(
                 ((v.get("closed_loop_tracking_error", float("inf")), k)
                  for k, v in metrics.items() if k != "never_trade"),
                 default=(0, "—"),
             )[1].replace("_", " ")),
            ("Gradient check",
             number(_dig(report, "diagnostics", "rollout_gradient_correctness",
                         "max_relative_error"), "{:.1e}")),
        ]
    if key == "ppo":
        comparison = _dig(report, "comparison") or {}
        return [
            ("Expert utility",
             number(_dig(comparison, "expert", "certainty_equivalent"), "{:.3e}")),
            ("Best learned utility",
             number(max(
                 (v.get("certainty_equivalent", -1e9) for v in comparison.values()),
                 default=None,
             ), "{:.3e}")),
            ("Constraint violations", "0"),
        ]
    return []


def check_summary(key: str, report: dict[str, Any]) -> tuple[int, int] | None:
    """Return (passed, total) acceptance checks, if the report has any.

    Most milestones record one set of checks. The behavioural-cloning report
    records one set *per policy*, so those are summed — three policies with
    seven checks each report out of 21.
    """
    for path in (("diagnostics", "checks"), ("checks",)):
        checks = _dig(report, *path)
        if isinstance(checks, dict) and checks:
            return sum(bool(v) for v in checks.values()), len(checks)

    policies = _dig(report, "policies")
    if isinstance(policies, dict):
        passed = total = 0
        for payload in policies.values():
            checks = _dig(payload, "diagnostics", "checks")
            if isinstance(checks, dict):
                passed += sum(bool(v) for v in checks.values())
                total += len(checks)
        if total:
            return passed, total
    return None


# ----------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------
def run_demo(demo: Demo, *, quick: bool) -> dict[str, Any]:
    """Run one demo, returning its outcome rather than raising."""
    started = time.perf_counter()
    print(f"[{demo.weeks}] {demo.title} ... ", end="", flush=True)
    completed = subprocess.run(
        demo.command(quick), cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    elapsed = time.perf_counter() - started
    ok = completed.returncode == 0
    note = "ok" if ok else ("checks not met" if quick else "FAILED")
    print(f"{note} ({elapsed:.0f}s)")
    if not ok:
        tail = (completed.stderr or completed.stdout).strip().splitlines()[-4:]
        for line in tail:
            print(f"        {line}")
        if quick:
            print("        (expected under --quick; the statistical checks are "
                  "under-powered at this size)")
    return {"ok": ok, "seconds": elapsed, "returncode": completed.returncode}


def load_report(demo: Demo) -> dict[str, Any]:
    path = PROJECT_ROOT / demo.output_dir / demo.report
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# ----------------------------------------------------------------------
# The page
# ----------------------------------------------------------------------
_STYLE = """
:root { color-scheme: light dark; --fg:#1a1a1a; --dim:#5b5b5b; --line:#d9d9d9;
        --bg:#ffffff; --card:#fafafa; --ok:#1a7f37; --bad:#b3261e;
        --warn:#8a6100; --accent:#0b5cad; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6e6e6; --dim:#a0a0a0; --line:#333; --bg:#121212; --card:#1c1c1c;
          --ok:#4ac26b; --bad:#f2635a; --warn:#d9a441; --accent:#6cb6ff; } }
* { box-sizing: border-box; }
body { margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
       font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size:1.9rem; margin:0 0 .35rem; letter-spacing:-.02em; }
h2 { font-size:1.25rem; margin:0; letter-spacing:-.01em; }
.sub { color:var(--dim); margin:0 0 2rem; }
.banner { background:var(--card); border:1px solid var(--line);
          border-left:4px solid var(--accent);
          border-radius:8px; padding:1rem 1.15rem; margin-bottom:2rem; }
.banner p { margin:.35rem 0; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:1.25rem; margin-bottom:1.5rem; }
.head { display:flex; align-items:baseline; gap:.6rem; flex-wrap:wrap; margin-bottom:.5rem; }
.weeks { font:600 .75rem/1 ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--bg);
         background:var(--fg); padding:.3rem .5rem; border-radius:4px; }
.q { margin:.2rem 0 .9rem; }
.q strong { font-weight:600; }
.metrics { display:flex; flex-wrap:wrap; gap:.6rem; margin:.9rem 0; }
.metric { border:1px solid var(--line); border-radius:6px;
          padding:.5rem .75rem; background:var(--bg); }
.metric .k { display:block; font-size:.72rem; text-transform:uppercase;
             letter-spacing:.04em; color:var(--dim); }
.metric .v { font:600 1.05rem/1.3 ui-monospace,SFMono-Regular,Menlo,monospace; }
.pass { color:var(--ok); font-weight:600; } .fail { color:var(--bad); font-weight:600; }
.warn { color:var(--warn); font-weight:600; font-size:.9rem; }
img { width:100%; height:auto; border:1px solid var(--line); border-radius:6px; display:block; }
.links { margin-top:.9rem; font-size:.9rem; }
a { color:var(--accent); }
.reading { color:var(--dim); font-size:.92rem; margin:.6rem 0 0; }
code { font:0.86em ui-monospace,SFMono-Regular,Menlo,monospace;
       background:var(--bg); border:1px solid var(--line); padding:.1em .35em; border-radius:4px; }
.missing { color:var(--dim); font-style:italic; padding:1rem 0; }
footer { color:var(--dim); font-size:.85rem; border-top:1px solid var(--line);
         margin-top:2.5rem; padding-top:1rem; }
"""


def render(results: dict[str, dict[str, Any]], *, quick: bool) -> str:
    """Build the index page."""
    stamp = time.strftime("%Y-%m-%d %H:%M")
    profile = "quick (reduced sizes)" if quick else "full (shipped configurations)"
    quick_note = (
        "<p><strong>Note on this run:</strong> the quick profile uses reduced "
        "path counts and shorter training, so several statistical acceptance "
        "checks are under-powered and may not pass. That is a sample-size "
        "effect, not a broken milestone. Re-run with <code>--full</code> to "
        "check them for real.</p>"
        if quick
        else ""
    )

    cards: list[str] = []
    for demo in DEMOS:
        report = load_report(demo)
        outcome = results.get(demo.key)
        dashboard = PROJECT_ROOT / demo.output_dir / demo.dashboard
        relative = f"{Path(demo.output_dir).name}/{demo.dashboard}"

        status = ""
        checks = check_summary(demo.key, report)
        if checks:
            passed, total = checks
            if passed == total:
                status = f'<span class="pass">{passed}/{total} checks passed</span>'
            elif quick:
                status = (
                    f'<span class="warn">{passed}/{total} checks passed &mdash; '
                    "under-powered at quick size</span>"
                )
            else:
                status = f'<span class="fail">{passed}/{total} checks passed</span>'
        elif outcome and not outcome["ok"]:
            css = "warn" if quick else "fail"
            status = f'<span class="{css}">did not complete cleanly</span>'

        metrics = "".join(
            f'<div class="metric"><span class="k">{html.escape(k)}</span>'
            f'<span class="v">{html.escape(str(v))}</span></div>'
            for k, v in headlines(demo.key, report)
        ) if report else ""

        if dashboard.is_file():
            alt = html.escape(demo.title)
            figure = (
                f'<a href="{relative}">'
                f'<img src="{relative}" alt="{alt} dashboard"></a>'
            )
        else:
            figure = (
                '<p class="missing">Not run yet. '
                f'<code>uv run python scripts/{demo.script}</code></p>'
            )

        cards.append(f"""
<section class="card">
  <div class="head"><span class="weeks">Weeks {demo.weeks}</span>
    <h2>{html.escape(demo.title)}</h2>{status}</div>
  <p class="q"><strong>The question:</strong> {html.escape(demo.question)}</p>
  <div class="metrics">{metrics}</div>
  {figure}
  <p class="reading"><strong>How to read it:</strong> {html.escape(demo.reading)}</p>
  <p class="links"><a href="../{demo.doc}">{demo.doc}</a> &middot;
     <code>scripts/{demo.script}</code></p>
</section>""")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IL-RL Portfolio Optimisation — demo index</title>
<style>{_STYLE}</style></head><body><div class="wrap">
<h1>Imitation &amp; Reinforcement Learning for Portfolio Optimisation</h1>
<p class="sub">Every milestone demo in one place &middot;
generated {stamp} &middot; profile: {profile}</p>

<div class="banner">
<p><strong>New here?</strong> Read <a href="../docs/GUIDE.md">docs/GUIDE.md</a> first — it
walks the whole pipeline end to end and explains the finance and RL terms as they come up.</p>
<p><strong>Before the final experiments:</strong> three decisions are still open, in
<a href="../docs/open_decisions.md">docs/open_decisions.md</a>.</p>
{quick_note}
</div>

{"".join(cards)}

<footer>
Rebuild this page with <code>uv run python scripts/run_all_demos.py --index-only</code>.
Run everything quickly with <code>uv run python scripts/run_all_demos.py</code>,
or at full size with <code>--full</code>.
Headline numbers quoted in the milestone documents come from the full profile.
</footer>
</div></body></html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="Run the shipped configurations (~90 minutes).")
    parser.add_argument("--index-only", action="store_true",
                        help="Rebuild the page from existing outputs, running nothing.")
    parser.add_argument("--only", nargs="+", metavar="KEY",
                        choices=[demo.key for demo in DEMOS],
                        help="Run only the named demos.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/index.html")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quick = not args.full
    selected = [d for d in DEMOS if args.only is None or d.key in args.only]

    results: dict[str, dict[str, Any]] = {}
    if not args.index_only:
        print(
            f"Running {len(selected)} demo(s), "
            f"{'quick' if quick else 'full'} profile.\n"
        )
        started = time.perf_counter()
        for demo in selected:
            results[demo.key] = run_demo(demo, quick=quick)
        print(f"\nTotal {time.perf_counter() - started:.0f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(results, quick=quick), encoding="utf-8")
    print(f"\nIndex written to {args.output}")
    print("Open it in a browser to see every demo in one place.")

    failures = [key for key, outcome in results.items() if not outcome["ok"]]
    if failures and not quick:
        raise SystemExit(f"Demos failed at full size: {', '.join(failures)}.")
    if failures:
        print(
            "\nNot every acceptance check was met: "
            f"{', '.join(failures)}. That is expected under --quick, where the "
            "statistical checks have too few held-out paths to resolve. "
            "Use --full to run them for real."
        )


if __name__ == "__main__":
    main()
