#!/usr/bin/env bash
# Submit the GGUF conversion job (experiments/03_model_format_converting/convert_to_gguf.py)
# to HuggingFace Jobs. Reads credentials from .env, runs the fail-closed
# push-scope gate, then submits with `hf jobs uv run --detach`.
#
# Usage:
#   ./experiments/03_model_format_converting/launch_hf_job.sh                  # defaults
#   HF_FLAVOR=cpu-upgrade ./experiments/03_model_format_converting/launch_hf_job.sh
#   QUANT_TYPES=Q8_0 ./experiments/03_model_format_converting/launch_hf_job.sh  # single quant
#   DRY_RUN=1 ./experiments/03_model_format_converting/launch_hf_job.sh         # gates only
#
# Required env (loaded from .env or your shell):
#   HF_TOKEN      (write スコープ必須 — TARGET_REPO へアップロードするため)
#
# Note: WANDB は不要（変換ジョブのため）。
#
# Optional env (forwarded to the conversion script):
#   SOURCE_MODEL, TARGET_REPO, QUANT_TYPES, CREATE_PUSH_TARGET
#
# Hardware flavors: https://huggingface.co/docs/hub/jobs-pricing

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../../scripts/shared/_load_env.sh
source "$REPO_ROOT/scripts/shared/_load_env.sh"
_load_env "$REPO_ROOT/.env"

: "${HF_TOKEN:?HF_TOKEN not set; populate .env or export it}"

HF_FLAVOR="${HF_FLAVOR:-l4x1}"
HF_TIMEOUT="${HF_TIMEOUT:-1h}"
SCRIPT="$REPO_ROOT/experiments/03_model_format_converting/convert_to_gguf.py"

SOURCE_MODEL="${SOURCE_MODEL:-akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract}"
TARGET_REPO="${TARGET_REPO:-akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract-GGUF}"
QUANT_TYPES="${QUANT_TYPES:-Q4_K_M,Q8_0,BF16}"
CREATE_PUSH_TARGET="${CREATE_PUSH_TARGET:-1}"

ENV_ARGS=(
  --env "SOURCE_MODEL=$SOURCE_MODEL"
  --env "TARGET_REPO=$TARGET_REPO"
  --env "QUANT_TYPES=$QUANT_TYPES"
  --env "CREATE_PUSH_TARGET=$CREATE_PUSH_TARGET"
)

NAMESPACE_ARGS=()
if [[ -n "${HF_NAMESPACE:-}" ]]; then
  NAMESPACE_ARGS=(--namespace "$HF_NAMESPACE")
fi

echo "Submitting GGUF conversion job:"
echo "  flavor:       $HF_FLAVOR  (rates: https://huggingface.co/docs/hub/jobs-pricing)"
echo "  timeout:      $HF_TIMEOUT  (billing stops here at the latest)"
echo "  source model: $SOURCE_MODEL"
echo "  target repo:  $TARGET_REPO"
echo "  quant types:  $QUANT_TYPES"
[[ -n "${HF_NAMESPACE:-}" ]] && echo "  namespace:    $HF_NAMESPACE"
case "${DRY_RUN:-}" in 1 | true | TRUE | True) _dry_run=1 ;; *) _dry_run=0 ;; esac
[[ "$_dry_run" == "1" ]] && echo "  dry run:      YES (gates will run; hf jobs submission will be skipped)"
echo

# Push gate: TARGET_REPO への書き込み権限を事前に確認。
# _validate_env.py は PUSH_TO_HUB を読むので一時的にセットして呼ぶ。
PUSH_TO_HUB="$TARGET_REPO" \
  uv run --no-sync python "$REPO_ROOT/scripts/shared/_validate_env.py" --check-push-scope

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

# Pass HF_TOKEN via a 0600 temp file to avoid exposing it in `ps aux`.
SECRETS_FILE=$(mktemp -t hf-job-secrets.XXXXXX)
trap 'rm -f "$SECRETS_FILE"' EXIT
cat >"$SECRETS_FILE" <<EOF
HF_TOKEN=$HF_TOKEN
EOF

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
  Watch logs:    make logs JOB=<job-id>   (job ID printed above)
  Check result:  https://huggingface.co/$TARGET_REPO
EOF
