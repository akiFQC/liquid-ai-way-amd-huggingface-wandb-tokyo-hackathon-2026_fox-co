#!/usr/bin/env python3
"""Post-training verification: confirm metrics actually landed
(not just "the training script exited 0").

  python scripts/verify_run.py --run-name <WANDB_RUN_NAME> \
                               [--require-metrics train/loss train/lr] \
                               [--require-hf-repo user/repo]

Exits 0 only if every required check passes; non-zero otherwise.

Checks: the W&B run exists and finished, each --require-metrics entry
has > 0 scalar samples, system metrics landed (the GPU was actually
used), and (with --require-hf-repo) that the exact commit this run
pushed is present on the Hub.

Auth via env: WANDB_API_KEY, HF_TOKEN. Reads WANDB_ENTITY / WANDB_PROJECT
defaults from env too.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # wandb's public-API types, for annotation only (string annotations via
    # `from __future__ import annotations`, so no runtime import cost).
    from wandb.apis.public import Api, Run

_ANSI = {"G": "\033[32m", "R": "\033[31m", "Y": "\033[33m", "B": "\033[1m", "N": "\033[0m"}


def color(name: str) -> str:
    return _ANSI.get(name, "") if sys.stdout.isatty() else ""


def ok(msg: str) -> None:
    print(f"  {color('G')}✓{color('N')} {msg}")


def bad(msg: str) -> None:
    print(f"  {color('R')}✗{color('N')} {msg}")


def info(msg: str) -> None:
    print(f"  {color('Y')}·{color('N')} {msg}")


def section(title: str) -> None:
    print(f"\n{color('B')}{title}{color('N')}")


def _summary_value(run: Run, key: str) -> str | None:
    """Read one summary key, tolerating the wandb 0.22.x quirk where
    `run.summary._json_dict` arrives as a JSON-encoded STRING (on which
    `summary.get()` raises TypeError). Returns None when the summary is
    missing or unparseable so the hf_revision gate fails closed with its
    actionable message instead of a traceback."""
    summary = getattr(run, "summary", None)
    if summary is None:
        return None
    raw = getattr(summary, "_json_dict", None)
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except ValueError:
            return None
        return decoded.get(key) if isinstance(decoded, dict) else None
    if isinstance(raw, dict):
        return raw.get(key)
    get = getattr(summary, "get", None)
    return get(key) if callable(get) else None


def check_hf_repo(
    repo_id: str,
    hf_token: str | None,
    expected_revision: str,
) -> tuple[bool, str]:
    """Verify the exact commit this run pushed exists on the HF Hub via
    `GET /api/models/<repo>/revision/<sha>`. A teammate's later push to
    the same repo (or a stale empty repo) can't satisfy this the way it
    would a looser repo-exists check.
    """
    url = f"https://huggingface.co/api/models/{repo_id}/revision/{expected_revision}"
    req = urllib.request.Request(url)
    if hf_token:
        req.add_header("Authorization", f"Bearer {hf_token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.load(r)
    except Exception as e:
        return False, (
            f"revision {expected_revision} not found in {repo_id}: "
            f"{type(e).__name__}: {e}. The launcher recorded this commit "
            f"to wandb.run.summary['hf_revision'] but the repo no longer "
            f"has it (overwritten by a later push, repo deleted, or "
            f"private to a different token)."
        )
    return True, (
        f"id={data.get('id')} revision={expected_revision[:12]}... "
        f"(verified via /api/models/.../revision/<sha>)"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity (default: $WANDB_ENTITY)",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT", "hack-the-liquid-way"),
        help="W&B project (default: $WANDB_PROJECT or hack-the-liquid-way)",
    )
    # Run identity. Prefer --run-id (immutable, exact) over --run-name
    # (display_name, ambiguous on collisions / retries). When --run-id is
    # set, --run-name is ignored.
    parser.add_argument(
        "--run-id",
        default=None,
        help="W&B run id (immutable). Overrides --run-name when given.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="WANDB_RUN_NAME (display_name). Ambiguous on collisions; prefer --run-id.",
    )
    parser.add_argument(
        "--allow-running",
        action="store_true",
        help="Accept run.state=='running' as a valid state. Default is "
        "'finished' only; a still-running job hasn't completed post-train "
        "probes / checkpoint save / push, so verification is premature.",
    )
    parser.add_argument(
        "--require-metrics",
        nargs="*",
        default=["train/loss"],
        help="Metric paths that must have at least one sample",
    )
    parser.add_argument(
        "--require-hf-repo",
        default=None,
        help="HF repo id that must exist (e.g. teozosa/lfm2-jenny-walkthrough)",
    )
    return parser.parse_args()


def find_run(
    api: Api, entity: str, project: str, run_id: str | None, run_name: str | None
) -> Run | None:
    """Resolve the run to verify, printing diagnostics. Returns the run, or
    None if it can't be uniquely resolved (caller exits 1). --run-id is an
    exact immutable lookup; --run-name fails closed on 0 or >1 matches rather
    than silently picking 'most recent', which could verify the wrong run."""
    section(f"1. W&B run lookup ({entity}/{project})")
    if run_id:
        try:
            run = api.run(f"{entity}/{project}/{run_id}")
        except Exception as e:
            bad(f"api.run() by id={run_id!r} failed: {type(e).__name__}: {e}")
            return None
        ok(f"found run {run.id} ({run.url})")
        return run
    try:
        runs = list(api.runs(f"{entity}/{project}", filters={"display_name": run_name}))
    except Exception as e:
        bad(f"runs() failed: {type(e).__name__}: {e}")
        return None
    if not runs:
        bad(f"no run named {run_name!r} found")
        return None
    if len(runs) > 1:
        bad(
            f"{len(runs)} runs named {run_name!r}; pass --run-id to disambiguate. "
            f"Candidates: {[(r.id, str(r.created_at)) for r in runs]}"
        )
        return None
    ok(f"found run {runs[0].id} ({runs[0].url})")
    return runs[0]


def check_state(run: Run, allow_running: bool) -> list[str]:
    """A still-running job hasn't saved / pushed yet, so a green verify on
    one is not proof of end-to-end success; require 'finished' unless
    --allow-running. Returns any failure descriptions."""
    section("2. Run state")
    ok(f"state = {run.state}")
    allowed = {"finished"} | ({"running"} if allow_running else set())
    if run.state in allowed:
        return []
    bad(
        f"unexpected state {run.state}; pass --allow-running for in-progress "
        f"inspection, otherwise wait for the run to finish."
    )
    return [f"run.state={run.state} (expected {sorted(allowed)!r})"]


def check_metrics(run: Run, require_metrics: list[str]) -> list[str]:
    """Each required metric must have >= 1 non-null scalar sample. Fail
    closed if history can't be fetched at all; unverified is not a pass."""
    section("3. Scalar metric history")
    try:
        history = run.history(samples=10_000)  # returns a pandas DataFrame
    except Exception as e:
        bad(f"run.history() failed: {type(e).__name__}: {e}")
        if require_metrics:
            return [
                f"could not verify required metrics {sorted(require_metrics)!r}: "
                f"run.history() fetch failed (see diagnostic above)"
            ]
        return []

    ok(f"history rows = {len(history)}")
    ok(f"history columns = {sorted([c for c in history.columns if not c.startswith('_')])}")
    failures: list[str] = []
    for metric in require_metrics:
        if metric not in history.columns:
            bad(f"metric {metric!r} missing from history")
            failures.append(f"metric {metric!r} missing")
            continue
        samples = history[metric].dropna()
        if len(samples) == 0:
            bad(f"metric {metric!r} has zero non-null samples")
            failures.append(f"metric {metric!r} has no samples")
        else:
            ok(
                f"{metric}: {len(samples)} samples, first={float(samples.iloc[0]):.4f}, last={float(samples.iloc[-1]):.4f}"
            )
    return failures


def report_system_metrics(run: Run) -> None:
    """Info-only: GPU/CPU telemetry proves the run actually used hardware. A
    very short run may legitimately have none, so this never fails the gate."""
    section("4. System metrics (GPU / CPU telemetry)")
    try:
        sys_history = run.history(stream="system", samples=200)
    except Exception as e:
        info(f"run.history(stream='system') unavailable: {type(e).__name__}: {e}")
        return

    if sys_history is not None and len(sys_history) > 0:
        ok(f"system metric rows = {len(sys_history)}")
        gpu_keys = sorted([c for c in sys_history.columns if "gpu" in c.lower()])[:5]
        if gpu_keys:
            ok(f"GPU keys (first 5): {gpu_keys}")
    else:
        info("no system metrics found (could mean very short run); not blocking")


def check_pushed_revision(run: Run, repo_id: str) -> list[str]:
    """Verify the exact commit this run pushed is on the Hub. Reads the
    `hf_revision` the launcher stamped into the run summary and checks that
    SHA; fails closed when the stamp is missing, since a teammate's later
    push to the same repo must not satisfy the gate."""
    section(f"5. HF Hub model presence ({repo_id})")
    expected_revision = _summary_value(run, "hf_revision")
    if not expected_revision:
        bad(
            f"run.summary['hf_revision'] is missing; cannot prove this run "
            f"uploaded to {repo_id!r}. The launcher stamps `hf_revision` after "
            f"a successful Hub push; missing means PUSH_TO_HUB was unset, or the "
            f"push step crashed before stamping. Re-run with PUSH_TO_HUB set, or "
            f"drop --require-hf-repo if you didn't intend to publish."
        )
        return [f"HF repo {repo_id!r}: no hf_revision recorded in run.summary"]
    present, detail = check_hf_repo(
        repo_id, os.environ.get("HF_TOKEN"), expected_revision=expected_revision
    )
    if present:
        ok(detail)
        return []
    bad(detail)
    return [f"HF repo {repo_id!r} not accessible"]


def main() -> int:
    args = _parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        print("ERROR: WANDB_API_KEY must be set", file=sys.stderr)
        return 64
    if not args.entity:
        print("ERROR: --entity or $WANDB_ENTITY required", file=sys.stderr)
        return 64
    if not args.run_id and not args.run_name:
        print("ERROR: --run-id or --run-name required", file=sys.stderr)
        return 64

    import wandb

    run = find_run(wandb.Api(), args.entity, args.project, args.run_id, args.run_name)
    if run is None:
        return 1
    ok(f"created_at = {run.created_at}")

    failures = check_state(run, args.allow_running)
    failures += check_metrics(run, args.require_metrics)
    report_system_metrics(run)
    if args.require_hf_repo:
        failures += check_pushed_revision(run, args.require_hf_repo)

    section("Summary")
    if failures:
        bad(f"{len(failures)} failure(s):")
        for f in failures:
            print(f"    - {f}")
        return 1
    ok("all required checks passed")
    print(f"\n  W&B run URL: {color('B')}{run.url}{color('N')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
