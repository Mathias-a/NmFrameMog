#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="ainm26osl-707"
REGION="europe-west1"
ZONE="europe-west1-b"
CONFIG_PATH="configs/cloud/main.json"
BUCKET_URI="gs://ainm26osl-707-ng-artifacts/"

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'Preflight prerequisite missing: %s\n' "$command_name" >&2
    exit 1
  fi
}

run_stage() {
  local stage_name="$1"
  shift

  printf '==> %s\n' "$stage_name"
  "$@"
  printf '{"stage":"%s","status":"ok"}\n' "$stage_name"
}

require_non_empty_output() {
  local stage_name="$1"
  local empty_output_message="$2"
  shift
  shift

  printf '==> %s\n' "$stage_name"

  local output
  output=$("$@")
  if [ -z "$output" ]; then
    printf 'Preflight failed at %s: %s\n' "$stage_name" "$empty_output_message" >&2
    exit 1
  fi

  printf '%s\n' "$output"
  printf '{"stage":"%s","status":"ok","value":"%s"}\n' "$stage_name" "$output"
}

require_exact_output() {
  local stage_name="$1"
  local expected_value="$2"
  shift 2

  printf '==> %s\n' "$stage_name"

  local output
  output=$("$@")
  if [ "$output" != "$expected_value" ]; then
    printf 'Preflight failed at %s: expected %s but received %s\n' \
      "$stage_name" \
      "$expected_value" \
      "${output:-<empty>}" >&2
    exit 1
  fi

  printf '%s\n' "$output"
  printf '{"stage":"%s","status":"ok","value":"%s"}\n' "$stage_name" "$output"
}

require_command uv
require_command gcloud

run_stage \
  "local.doctor" \
  uv run python -m src.ng_data.cli.doctor --root .

run_stage \
  "local.validate_config" \
  uv run python -m src.ng_data.cloud.validate_config --config "$CONFIG_PATH" --project "$PROJECT_ID" --region "$REGION" --dry-run

run_stage \
  "local.print_paths" \
  uv run python -m src.ng_data.cloud.print_paths --config "$CONFIG_PATH"

require_exact_output \
  "gcloud.current_project" \
  "$PROJECT_ID" \
  gcloud config get-value project

require_non_empty_output \
  "gcloud.active_account" \
  "expected active output but received none" \
  gcloud auth list --filter=status:ACTIVE '--format=value(account)'

require_non_empty_output \
  "gcloud.enabled_services" \
  "expected enabled service output but received none" \
  gcloud services list --enabled '--filter=name:(compute.googleapis.com storage.googleapis.com)' '--format=value(name)'

require_non_empty_output \
  "gcloud.accelerator_type" \
  "expected accelerator output but received none" \
  gcloud compute accelerator-types list '--filter=zone:( europe-west1-b ) AND name:nvidia-tesla-t4' '--format=value(name)'

run_stage \
  "gcloud.bucket_access" \
  gcloud storage ls "$BUCKET_URI"
