from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_gcs_path(path: str) -> bool:
    return path.startswith("gs://")


def split_gcs_uri(uri: str) -> tuple[str, str]:
    if not is_gcs_path(uri):
        raise ValueError(f"Not a GCS URI: {uri}")
    bucket, _, blob = uri[5:].partition("/")
    if not bucket or not blob:
        raise ValueError(f"GCS URI must include bucket and object: {uri}")
    return bucket, blob


def download_if_gcs(path: str) -> str:
    if not is_gcs_path(path):
        return path
    from google.cloud import storage

    bucket_name, blob_name = split_gcs_uri(path)
    local_path = os.path.join(tempfile.mkdtemp(prefix="minigpt_gcs_"), os.path.basename(blob_name))
    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).download_to_filename(local_path)
    return local_path


def upload_if_gcs(local_path: str, dest_path: str) -> None:
    if not is_gcs_path(dest_path):
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        if os.path.abspath(local_path) != os.path.abspath(dest_path):
            Path(dest_path).write_bytes(Path(local_path).read_bytes())
        return
    from google.cloud import storage

    bucket_name, blob_name = split_gcs_uri(dest_path)
    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).upload_from_filename(local_path)


def exists_path(path: str) -> bool:
    if not is_gcs_path(path):
        return Path(path).exists()
    from google.cloud import storage

    bucket_name, blob_name = split_gcs_uri(path)
    client = storage.Client()
    return client.bucket(bucket_name).blob(blob_name).exists()


def torch_load(path: str, map_location: str | torch.device = "cpu") -> Any:
    return torch.load(download_if_gcs(path), map_location=map_location)


def torch_save(obj: Any, path: str) -> None:
    if is_gcs_path(path):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            torch.save(obj, tmp_path)
            upload_if_gcs(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, path)
