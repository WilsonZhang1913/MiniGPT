#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/gcp.yaml}"

read_config() {
  python - "$CONFIG_PATH" "$1" <<'PY'
import sys
import yaml

path, key = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
value = cfg
for part in key.split("."):
    value = value[part]
print(value)
PY
}

PROJECT="$(read_config project)"
REGION="$(read_config region)"
BUCKET="$(read_config bucket)"
REPO="$(read_config artifact_registry_repo)"
SERVICE_ACCOUNT="$(read_config service_account)"

gcloud config set project "$PROJECT"
gcloud services enable \
  storage.googleapis.com \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com

if ! gsutil ls -b "gs://${BUCKET}" >/dev/null 2>&1; then
  gsutil mb -l "$REGION" "gs://${BUCKET}"
fi

if ! gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPO" \
    --repository-format=docker \
    --location="$REGION" \
    --description="MiniGPT training images"
fi

if ! gcloud iam service-accounts describe "$SERVICE_ACCOUNT" >/dev/null 2>&1; then
  NAME="${SERVICE_ACCOUNT%@*}"
  gcloud iam service-accounts create "$NAME" --display-name="MiniGPT Vertex trainer"
fi

gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/storage.objectAdmin"

gcloud auth configure-docker "${REGION}-docker.pkg.dev"

