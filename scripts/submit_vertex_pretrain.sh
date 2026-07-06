#!/usr/bin/env bash
set -euo pipefail

GCP_CONFIG="${1:-configs/gcp.yaml}"
TRAIN_CONFIG="${2:-configs/pretrain.yaml}"

python - "$GCP_CONFIG" "$TRAIN_CONFIG" <<'PY'
import subprocess
import sys
import yaml

gcp_path, train_config = sys.argv[1], sys.argv[2]
with open(gcp_path, "r", encoding="utf-8") as f:
    gcp = yaml.safe_load(f)
vertex = gcp["vertex"]
cmd = [
    "gcloud", "ai", "custom-jobs", "create",
    "--region", gcp["region"],
    "--display-name", "minigpt-pretrain",
    "--worker-pool-spec",
    (
        f"machine-type={vertex['machine_type']},"
        f"accelerator-type={vertex['accelerator_type']},"
        f"accelerator-count={vertex['accelerator_count']},"
        f"replica-count={vertex['replica_count']},"
        f"container-image-uri={gcp['image_uri']}"
    ),
    "--args", f"src.train_pretrain,--config,{train_config}",
    "--service-account", gcp["service_account"],
]
print(" ".join(cmd))
subprocess.check_call(cmd)
PY
