#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/cloud/main.json}"
DRY_RUN="${DRY_RUN:-1}"

eval "$(uv run python -m src.ng_data.cloud.print_paths --config "$CONFIG_PATH" --format env)"

CREATE_VM_CMD=(
  gcloud compute instances create "$GCE_VM_NAME"
  "--project=$GCP_PROJECT_ID"
  "--zone=$GCE_ZONE"
  "--machine-type=$GCE_MACHINE_TYPE"
  "--accelerator=type=$GCE_ACCELERATOR_TYPE,count=$GCE_ACCELERATOR_COUNT"
  "--boot-disk-size=${GCE_BOOT_DISK_GB}GB"
  "--image-project=$GCE_IMAGE_PROJECT"
  "--image-family=$GCE_IMAGE_FAMILY"
  "--metadata=install-nvidia-driver=True"
  "--maintenance-policy=TERMINATE"
  "--restart-on-failure"
  "--scopes=https://www.googleapis.com/auth/cloud-platform"
  "--labels=workload=ng-training,mode=persistent"
)

REMOTE_BOOTSTRAP_CMD=$(cat <<EOF
REMOTE_HOME="\$HOME"
REMOTE_WORKSPACE_DIR="\${REMOTE_HOME}${GCE_WORKSPACE_DIR#/home/ng}"
REMOTE_LOCAL_ARTIFACT_DIR="\${REMOTE_HOME}${GCE_LOCAL_ARTIFACT_DIR#/home/ng}"
sudo apt-get update
sudo apt-get install -y git curl python3-pip build-essential pkg-config
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
uv --version
python3 --version
uv python install 3.11
export UV_PROJECT_ENVIRONMENT="\${REMOTE_WORKSPACE_DIR}/.venv"
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  if nvidia-smi; then
    break
  fi
  if [ "\$attempt" -eq 10 ]; then
    printf 'nvidia-smi did not become ready after %s attempts\n' "\$attempt" >&2
    exit 1
  fi
  printf 'nvidia-smi not ready yet (attempt %s/10); sleeping 30s\n' "\$attempt"
  sleep 30
done
mkdir -p "\$REMOTE_WORKSPACE_DIR" "\$REMOTE_LOCAL_ARTIFACT_DIR"
printf 'Canonical GCS namespace: %s\n' "$GCS_NAMESPACE_ROOT"
printf 'Checkpoint sync target: %s\n' "$GCS_CHECKPOINTS"
printf 'Vertex AI mode: %s\n' "$VERTEX_AI_MODE"
printf 'Remote workspace dir: %s\n' "\$REMOTE_WORKSPACE_DIR"
printf 'Remote artifact dir: %s\n' "\$REMOTE_LOCAL_ARTIFACT_DIR"
cd "\$REMOTE_WORKSPACE_DIR"
if [ -f pyproject.toml ]; then
  uv sync --python 3.11
  uv run --python 3.11 python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
else
  printf 'Skipping repo environment sync because %s/pyproject.toml is not present yet\n' "\$REMOTE_WORKSPACE_DIR"
fi
EOF
)

SSH_BOOTSTRAP_CMD=(
  gcloud compute ssh "$GCE_VM_NAME"
  "--project=$GCP_PROJECT_ID"
  "--zone=$GCE_ZONE"
  "--command=$REMOTE_BOOTSTRAP_CMD"
)

if [ "$DRY_RUN" = "1" ]; then
  printf 'Dry run. Create VM command:\n'
  printf '  %q' "${CREATE_VM_CMD[@]}"
  printf '\n'
  printf 'Dry run. Remote bootstrap command:\n'
  printf '  %q' "${SSH_BOOTSTRAP_CMD[@]}"
  printf '\n'
  exit 0
fi

"${CREATE_VM_CMD[@]}"
"${SSH_BOOTSTRAP_CMD[@]}"
