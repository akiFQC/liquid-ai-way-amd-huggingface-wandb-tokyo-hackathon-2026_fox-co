#!/usr/bin/env bash
# Submit the LFM2 text LoRA fine-tune (scripts/text/train.py) to
# HuggingFace Jobs. Reads credentials from .env, runs the fail-closed
# gates, then submits with `hf jobs uv run --detach`.
#
# Usage:
#   ./scripts/text/launch_hf_job.sh                          # defaults (a100-large, 1h timeout)
#   HF_FLAVOR=l40sx1 ./scripts/text/launch_hf_job.sh
#   MAX_STEPS=500 BATCH_SIZE=8 ./scripts/text/launch_hf_job.sh
#   DRY_RUN=1 ./scripts/text/launch_hf_job.sh                # run the gates, skip the submit
#
# Required env (loaded from .env or your shell):
#   HF_TOKEN, WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT
#
# Optional env (forwarded to the training script; see the docstring at the
# top of scripts/text/train.py for what each does):
#   MODEL_ID, DATASET, DATASET_SLICE, DATASET_MAPPER, MAX_STEPS, BATCH_SIZE,
#   LR, OUTPUT_DIR, PUSH_TO_HUB, WANDB_RUN_NAME
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
HF_TIMEOUT="${HF_TIMEOUT:-1h}"
SCRIPT="$REPO_ROOT/scripts/text/train.py"

# Forward each optional training override as `--env VAR=value` (only those
# that are set).
ENV_ARGS=()
for var in MODEL_ID DATASET DATASET_SLICE DATASET_MAPPER MAX_STEPS BATCH_SIZE \
  LR OUTPUT_DIR PUSH_TO_HUB WANDB_RUN_NAME; do
  if [[ -n "${!var:-}" ]]; then
    ENV_ARGS+=(--env "$var=${!var}")
  fi
done

NAMESPACE_ARGS=()
if [[ -n "${HF_NAMESPACE:-}" ]]; then
  NAMESPACE_ARGS=(--namespace "$HF_NAMESPACE")
fi

echo "Submitting LFM2 text fine-tune:"
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

# Fail-closed gates, run through the uv-managed interpreter so imports
# resolve against the kit's pinned versions.
#
# Push gate first: PUSH_TO_HUB writeability is the most expensive failure
# mode (403 surfaces at the END of paid training).
if [[ -n "${PUSH_TO_HUB:-}" ]]; then
  uv run --no-sync python "$REPO_ROOT/scripts/shared/_validate_env.py" --check-push-scope
fi

# Dataset preflight: stream the first rows of DATASET locally and run the
# same schema check the in-job code runs. Catches DATASET typos and
# column-name mismatches in ~5s here instead of after a ~5min
# submit → bootstrap → fail round trip on HF Jobs.
uv run --no-sync python "$REPO_ROOT/scripts/shared/_check_dataset.py"

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
