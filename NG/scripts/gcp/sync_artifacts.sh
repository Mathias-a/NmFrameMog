#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-push}"
CONFIG_PATH="${2:-configs/cloud/main.json}"
DRY_RUN="${DRY_RUN:-1}"

eval "$(uv run python -m src.ng_data.cloud.print_paths --config "$CONFIG_PATH" --format env)"

declare -a COMMANDS

case "$MODE" in
  push)
    COMMANDS=(
      "gcloud storage rsync --recursive artifacts/models/detector $GCS_RUNS"
      "gcloud storage rsync --recursive artifacts/checkpoints $GCS_CHECKPOINTS"
      "gcloud storage rsync --recursive artifacts/eval $GCS_EVAL_OUTPUTS"
      "gcloud storage rsync --recursive dist $GCS_RELEASE_BUNDLES"
      "gcloud storage cp artifacts/run_manifest.json $GCS_RUN_MANIFESTS"
    )
    ;;
  pull)
    COMMANDS=(
      "gcloud storage rsync --recursive $GCS_RUNS artifacts/models/detector"
      "gcloud storage rsync --recursive $GCS_CHECKPOINTS artifacts/checkpoints"
      "gcloud storage rsync --recursive $GCS_EVAL_OUTPUTS artifacts/eval"
      "gcloud storage rsync --recursive $GCS_RELEASE_BUNDLES dist"
      "gcloud storage cp ${GCS_RUN_MANIFESTS%/}/run_manifest.json artifacts/run_manifest.json"
    )
    ;;
  *)
    printf 'Unsupported sync mode: %s\n' "$MODE" >&2
    exit 1
    ;;
esac

for command in "${COMMANDS[@]}"; do
  if [ "$DRY_RUN" = "1" ]; then
    printf 'Dry run: %s\n' "$command"
  else
    bash -lc "$command"
  fi
done
