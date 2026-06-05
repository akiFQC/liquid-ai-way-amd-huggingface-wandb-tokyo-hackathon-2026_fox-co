#!/usr/bin/env bash
# Submit the LFM2.5-Audio TTS fine-tune (scripts/audio/train.py) to
# HuggingFace Jobs. Reads credentials from .env, runs the fail-closed
# gates, then submits with `hf jobs uv run --detach`.
#
# Usage:
#   ./scripts/audio/launch_hf_job.sh                    # defaults (a100-large, 2h timeout, full Jenny dataset)
#   HF_FLAVOR=h200 ./scripts/audio/launch_hf_job.sh
#   MAX_STEPS=500 ./scripts/audio/launch_hf_job.sh
#   DRY_RUN=1 ./scripts/audio/launch_hf_job.sh          # run the gates, skip the submit
#
# Required env (loaded from .env or your shell):
#   HF_TOKEN, WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT
#
# Optional env (forwarded to the training script; see the docstring at the
# top of scripts/audio/train.py for what each does):
#   MAX_STEPS, BATCH_SIZE, LR, DATASET_SLICE,
#   PUSH_TO_HUB, MODEL_ID, OUTPUT_DIR, WANDB_RUN_NAME
#
# Training on your own dataset? That's a code change, not an env var:
# rewrite `TrainingSamples.__iter__` in scripts/audio/train.py.
#
# Hardware flavors: https://huggingface.co/docs/hub/jobs-pricing

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../shared/_load_env.sh
source "$REPO_ROOT/scripts/shared/_load_env.sh"
_load_env "$REPO_ROOT/.env"

: "${HF_TOKEN:?HF_TOKEN not set; populate .env or export it}"
: "${WANDB_API_KEY:?WANDB_API_KEY not set; populate .env or export it}"
: "${WANDB_ENTITY:?WANDB_ENTITY not set; populate .env or export it}"
: "${WANDB_PROJECT:?WANDB_PROJECT not set; populate .env or export it}"

HF_FLAVOR="${HF_FLAVOR:-a100-large}"
HF_TIMEOUT="${HF_TIMEOUT:-2h}"
SCRIPT="$REPO_ROOT/scripts/audio/train.py"

# Forward each optional training override as `--env VAR=value` (only those
# that are set).
ENV_ARGS=()
for var in MAX_STEPS BATCH_SIZE LR DATASET_SLICE \
  PUSH_TO_HUB MODEL_ID OUTPUT_DIR WANDB_RUN_NAME; do
  if [[ -n "${!var:-}" ]]; then
    ENV_ARGS+=(--env "$var=${!var}")
  fi
done

NAMESPACE_ARGS=()
if [[ -n "${HF_NAMESPACE:-}" ]]; then
  NAMESPACE_ARGS=(--namespace "$HF_NAMESPACE")
fi

echo "Submitting LFM2.5-Audio TTS fine-tune:"
echo "  flavor:    $HF_FLAVOR  (rates: https://huggingface.co/docs/hub/jobs-pricing)"
echo "  timeout:   $HF_TIMEOUT  (billing stops here at the latest)"
echo "  W&B:       $WANDB_ENTITY/$WANDB_PROJECT"
# Echo PUSH_TO_HUB before submit so typos (`my-finetnue` vs `my-finetune`)
# are visible against the resolved target.
[[ -n "${PUSH_TO_HUB:-}" ]] && echo "  push to:   $PUSH_TO_HUB"
[[ -n "${HF_NAMESPACE:-}" ]] && echo "  namespace: $HF_NAMESPACE"
case "${DRY_RUN:-}" in 1 | true | TRUE | True) _dry_run=1 ;; *) _dry_run=0 ;; esac
[[ "$_dry_run" == "1" ]] && echo "  dry run:   YES (gates will run; hf jobs submission will be skipped)"
echo

# Fail-closed gate, run through the uv-managed interpreter so imports
# resolve against the kit's pinned versions. PUSH_TO_HUB writeability is
# the most expensive failure mode (403 surfaces at the END of paid
# training). There is no dataset preflight here: the audio data source is
# the TrainingSamples iterator in the training script itself.
if [[ -n "${PUSH_TO_HUB:-}" ]]; then
  uv run --no-sync python "$REPO_ROOT/scripts/shared/_validate_env.py" --check-push-scope
fi

if [[ "$_dry_run" == "1" ]]; then
  echo
  echo "DRY_RUN=1: gates passed. The following hf jobs invocation would have been submitted:"
  echo
  printf '  uv run --no-sync hf jobs uv run'
  for arg in "${NAMESPACE_ARGS[@]}"; do printf ' %q' "$arg"; done
  printf ' --flavor %q --timeout %q --secrets-file <generated>' "$HF_FLAVOR" "$HF_TIMEOUT"
  for arg in "${ENV_ARGS[@]}"; do printf ' %q' "$arg"; done
  printf ' --detach %q\n' "$SCRIPT"
  echo
  echo "Re-run without DRY_RUN to actually submit."
  exit 0
fi

# Pass credentials via a 0600 temp file instead of `--secrets KEY=value`
# argv, which would expose HF_TOKEN / WANDB_API_KEY in `ps aux` and shell
# history. The EXIT trap cleans the file up even if the hf CLI fails.
SECRETS_FILE=$(mktemp -t hf-job-secrets.XXXXXX)
trap 'rm -f "$SECRETS_FILE"' EXIT
cat >"$SECRETS_FILE" <<EOF
HF_TOKEN=$HF_TOKEN
WANDB_API_KEY=$WANDB_API_KEY
WANDB_ENTITY=$WANDB_ENTITY
WANDB_PROJECT=$WANDB_PROJECT
EOF

# `uv run --no-sync` uses the venv-installed `hf` CLI (from
# huggingface-hub[cli] in pyproject) rather than whatever is on PATH.
uv run --no-sync hf jobs uv run \
  "${NAMESPACE_ARGS[@]}" \
  --flavor "$HF_FLAVOR" \
  --timeout "$HF_TIMEOUT" \
  --secrets-file "$SECRETS_FILE" \
  "${ENV_ARGS[@]}" \
  --detach \
  "$SCRIPT"

cat <<EOF

Next steps:
  Watch logs:   make logs JOB=<job-id>     # (job ID printed above)
  Verify run:   make verify RUN=<wandb-run-name>  [HF_REPO=user/repo]
                # (wandb run name is in the W&B run URL the job will print
                # in its first log lines, or set WANDB_RUN_NAME yourself)
EOF
