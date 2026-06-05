# Makefile: user-facing command runner for the Hack the Liquid WAY starter kit.
#
# Run `make` (no args) for the documented target list. Each target below has
# a `## docstring` that `make help` greps and prints.

.SILENT:
.DEFAULT_GOAL := help
.PHONY: help install validate validate-audio smoke-text smoke-audio smoke-eval logs verify clean

help:  ## show this help
	echo "Hack the Liquid WAY starter kit commands"
	echo
	grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sort | awk -F'[:#]+ *' '{printf "  make %-15s %s\n", $$1, $$3}'

install:  ## uv sync (install dependencies)
	uv sync
	echo
	echo "✓ Installed. Next: cp .env.example .env, fill in HF_TOKEN/WANDB_API_KEY,"
	echo "  then run 'make validate' before your first HF Jobs submission."

validate:  ## auth probes + import checks before any paid HF Job
	bash -c 'source ./scripts/shared/_load_env.sh && _load_env .env && \
	  uv run --no-sync python scripts/shared/_validate_env.py'

validate-audio:  ## like validate, also probes the [audio] extra (liquid_audio)
	bash -c 'source ./scripts/shared/_load_env.sh && _load_env .env && \
	  uv run --no-sync python scripts/shared/_validate_env.py --audio'

smoke-text:  ## load LFM2-350M locally + generate a 3-word answer (no W&B / Hub)
	uv run python scripts/text/_smoke.py

smoke-audio:  ## synthesize ~1.3s TTS (CUDA box only; liquid_audio decode is CUDA-only, Macs use smoke-text + HF Jobs)
	uv run --extra audio python scripts/audio/_smoke.py

smoke-eval:  ## scripts/run_eval.py --wandb against base only (~3 min on Mac MPS)
	bash -c 'source ./scripts/shared/_load_env.sh && _load_env .env && \
	  uv run python scripts/run_eval.py \
	    --base LiquidAI/LFM2-350M --finetune LiquidAI/LFM2-350M \
	    --wandb --max-new-tokens 32'

logs:  ## tail HF Jobs logs (make logs JOB=<job-id-from-launcher>)
	@if [ -z "$(JOB)" ]; then \
	  echo "Usage: make logs JOB=<job-id>"; \
	  echo "  (Job ID is printed by the launcher under 'Job started with ID: ...')"; \
	  exit 64; \
	fi
	uv run --no-sync hf jobs logs $(JOB)

verify:  ## verify a finished training run (make verify RUN=<wandb-run-name> [HF_REPO=user/repo])
	@if [ -z "$(RUN)" ]; then \
	  echo "Usage: make verify RUN=<wandb-run-name> [HF_REPO=user/repo]"; \
	  echo "  RUN is the WANDB_RUN_NAME the launcher logged at submit time."; \
	  echo "  Optional HF_REPO confirms the trained checkpoint landed on the Hub."; \
	  exit 64; \
	fi
	bash -c 'source ./scripts/shared/_load_env.sh && _load_env .env && \
	  uv run python scripts/verify_run.py --run-name $(RUN) \
	    --require-metrics train/loss \
	    $(if $(HF_REPO),--require-hf-repo $(HF_REPO))'

clean:  ## remove local caches (wandb run cache, build artifacts)
	rm -rf wandb/ build/ dist/ *.egg-info
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	echo "✓ Cleaned caches."
