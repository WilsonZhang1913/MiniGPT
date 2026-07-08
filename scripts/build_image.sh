#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/gcp.yaml}"

IMAGE_URI="$(python - "$CONFIG_PATH" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(yaml.safe_load(f)["image_uri"])
PY
)"

docker build --platform linux/amd64 -t "$IMAGE_URI" .
docker push "$IMAGE_URI"
