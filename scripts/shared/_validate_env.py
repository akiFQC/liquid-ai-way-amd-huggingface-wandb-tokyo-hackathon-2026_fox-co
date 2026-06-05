#!/usr/bin/env python3
"""Pre-flight environment validator, run as `make validate` (or
`make validate-audio` for the audio extra): required env vars, HF +
W&B auth probes, Python imports. Exit 0 if all required checks pass;
warnings don't fail.

`--check-push-scope` (used by the launchers at submit time): verify
HF_TOKEN can write the PUSH_TO_HUB target, so a read-only token or
wrong-namespace target fails BEFORE the paid run instead of 403ing at
its end.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import warnings

# Hide unactionable pydantic deprecation noise from huggingface_hub/weave.
warnings.filterwarnings("ignore", message=r".*was provided to the `Field\(\)`.*")


class Probe:
    """Tally pass / fail / warn counts and pretty-print per-check results."""

    def __init__(self) -> None:
        self.passed = self.failed = self.warned = 0

    def ok(self, msg: str) -> None:
        print(f"  ✓ {msg}")
        self.passed += 1

    def bad(self, msg: str) -> None:
        print(f"  ✗ {msg}")
        self.failed += 1

    def warn(self, msg: str) -> None:
        print(f"  ! {msg}")
        self.warned += 1

    def section(self, title: str) -> None:
        print(f"\n{title}")


REQUIRED_ENV_VARS = ("HF_TOKEN", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT")


def _check_env_vars(p: Probe) -> None:
    p.section("1. Required environment variables")
    for name in REQUIRED_ENV_VARS:
        val = os.environ.get(name, "")
        if not val:
            p.bad(f"{name} unset")
        elif val.startswith("<") and val.endswith(">"):
            p.bad(f"{name} still has the .env.example placeholder ({val})")
        else:
            p.ok(f"{name} set ({len(val)} chars)")


def _check_hf_auth(p: Probe) -> None:
    p.section("2. HuggingFace auth")
    if not os.environ.get("HF_TOKEN", ""):
        p.warn("skipped (HF_TOKEN unset; fix §1 first)")
        return
    try:
        from huggingface_hub import HfApi

        info = HfApi().whoami()
    except Exception as e:
        p.bad(f"whoami probe failed: {type(e).__name__}: {e}")
        return
    name = info.get("name", "?") if isinstance(info, dict) else "?"
    p.ok(f"whoami: '{name}' (token resolves to a user)")

    # Classic tokens report a role; "read" means PUSH_TO_HUB would 403
    # at the END of a paid run. Advisory here; the launcher's
    # --check-push-scope gate is the authoritative check at submit time.
    role = (
        (((info.get("auth") or {}).get("accessToken") or {}).get("role"))
        if isinstance(info, dict)
        else None
    )
    if role == "read":
        p.warn(
            "HF_TOKEN is READ-ONLY; PUSH_TO_HUB would fail with 403. Re-issue "
            "at https://huggingface.co/settings/tokens with write scope."
        )
    push_target = os.environ.get("PUSH_TO_HUB", "")
    if not push_target and isinstance(name, str) and name not in ("", "?"):
        p.ok(
            f"PUSH_TO_HUB unset (publishing the fine-tune is optional). "
            f"When you're ready: `PUSH_TO_HUB={name}/your-finetune-name "
            f"./scripts/text/launch_hf_job.sh`"
        )


def _check_wandb_auth(p: Probe) -> None:
    p.section("3. Weights & Biases auth")
    key = os.environ.get("WANDB_API_KEY", "")
    if not key:
        p.warn("skipped (WANDB_API_KEY unset; fix §1 first)")
        return
    try:
        import wandb

        viewer = wandb.Api(api_key=key).viewer
        username = getattr(viewer, "username", None) or getattr(viewer, "name", None)
        if not username:
            p.bad("W&B viewer object has no username; check API key validity")
            return
        p.ok(f"W&B viewer: '{username}' (API key valid)")
        entity = os.environ.get("WANDB_ENTITY", "") or username
        project = os.environ.get("WANDB_PROJECT", "") or "(set at submit time)"
        p.ok(f"will log to {entity}/{project}")
    except Exception as e:
        p.bad(f"W&B auth probe failed: {type(e).__name__}: {e}")


def _check_imports(p: Probe, with_audio: bool) -> None:
    p.section("4. Python package imports")
    modules = ["transformers", "datasets", "wandb", "weave", "huggingface_hub"]
    if with_audio:
        modules += ["liquid_audio"]
    for mod in modules:
        try:
            m = importlib.import_module(mod)
            p.ok(f"{mod}={getattr(m, '__version__', '?')}")
        except ImportError as e:
            hint = "uv sync --extra audio" if mod == "liquid_audio" else "uv sync"
            p.bad(f"import {mod} failed ({e}); run '{hint}'")
        except Exception as e:
            # liquid_audio used to crash here on a missing FFmpeg (its
            # torchcodec dep dlopened libav at import); the [audio] extra now
            # pins a torchcodec-free SHA, so any crash is a genuine fault.
            p.bad(f"import {mod} crashed ({type(e).__name__}: {e})")


def _gate_push_scope() -> int:
    """Submit-time gate: `create_repo(..., exist_ok=True)` is a write
    endpoint, so a read-only token or uncovered namespace fails here with
    HF's own 403 instead of at the end-of-run push. The training job runs
    the same call before uploading, so creating the (private) repo a
    little early is the only side effect. Exit 0 = writeable or nothing
    to gate; 1 = not.
    """
    push_target = os.environ.get("PUSH_TO_HUB", "")
    if not push_target:
        return 0
    if "/" not in push_target:
        print(
            f"  ✗ PUSH_TO_HUB={push_target!r} must be fully qualified as "
            f"'username/repo-name' (your username is shown at "
            f"https://huggingface.co/settings/account)."
        )
        return 1
    if not os.environ.get("HF_TOKEN", ""):
        print("  ✗ HF_TOKEN unset; cannot verify PUSH_TO_HUB writeability")
        return 1
    print(f"\nProbing HF write access to PUSH_TO_HUB={push_target}...")
    try:
        from huggingface_hub import HfApi

        HfApi().create_repo(push_target, private=True, exist_ok=True)
    except Exception as e:
        print(
            f"  ✗ HF_TOKEN cannot write {push_target!r} "
            f"({type(e).__name__}: {e}). Re-issue the token at "
            f"https://huggingface.co/settings/tokens with write scope for "
            f"this namespace, or change PUSH_TO_HUB to a repo you own. "
            f"Caught before training, so no GPU credit was spent."
        )
        return 1
    print(f"  ✓ writeability confirmed for {push_target}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate env before submitting an HF Job.")
    parser.add_argument(
        "--audio", action="store_true", help="also probe the [audio] extra (liquid_audio)"
    )
    parser.add_argument(
        "--check-push-scope",
        action="store_true",
        help="standalone submit-time gate: verify HF_TOKEN can write "
        "PUSH_TO_HUB and exit 0/1; skip all other probes",
    )
    args = parser.parse_args()

    if args.check_push_scope:
        return _gate_push_scope()

    p = Probe()
    _check_env_vars(p)
    _check_hf_auth(p)
    _check_wandb_auth(p)
    _check_imports(p, with_audio=args.audio)

    print(f"\n{p.passed} passed, {p.failed} failed, {p.warned} warnings")
    if p.failed:
        if any(not os.environ.get(v, "") for v in REQUIRED_ENV_VARS) and not os.path.exists(".env"):
            print("\nFix:  cp .env.example .env  (then edit .env with your credentials)")
        else:
            print("\nFix the items marked ✗ above, then re-run.")
        return 1
    print("\nReady for remote HF Jobs submission:")
    print("  ./scripts/text/launch_hf_job.sh      # LFM2 text fine-tune")
    if args.audio:
        print("  ./scripts/audio/launch_hf_job.sh     # LFM2.5-Audio fine-tune")
    return 0


if __name__ == "__main__":
    sys.exit(main())
