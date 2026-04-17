from __future__ import annotations

import io
import json
import logging
import os
import random
import time
from pathlib import Path

import boto3  # type: ignore[import-not-found]
import torch
from botocore.config import Config as BotoConfig  # type: ignore[import-not-found]
from botocore.exceptions import BotoCoreError, ClientError  # type: ignore[import-not-found]


class MissingManifestError(RuntimeError):
    pass


class MissingCheckpointError(RuntimeError):
    pass


class InvalidCheckpointPayloadError(RuntimeError):
    pass


class S3ParameterServer:
    def __init__(
        self,
        bucket: str,
        session_id: str,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
    ):
        self.bucket = bucket
        self.session_id = session_id
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.log = logging.getLogger("S3ParameterServer")

        cache_root = os.getenv("POLICY_CACHE_DIR", ".policy_cache")
        self.cache_dir = Path(cache_root).expanduser() / self.session_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_retries = 5
        self.retry_base_delay_s = 0.2
        self.retry_jitter_s = 0.05

        self.s3 = self._make_s3_client()

    @classmethod
    def from_env(cls) -> "S3ParameterServer":
        endpoint = os.getenv("S3_ENDPOINT_URL")
        access_key = os.getenv("S3_ACCESS_KEY_ID")
        secret_key = os.getenv("S3_SECRET_ACCESS_KEY")
        bucket = os.getenv("S3_BUCKET")
        session_id = os.getenv("SESSION_ID")
        region = os.getenv("S3_REGION", "us-east-1")

        missing = [
            name
            for name, value in [
                ("S3_ENDPOINT_URL", endpoint),
                ("S3_ACCESS_KEY_ID", access_key),
                ("S3_SECRET_ACCESS_KEY", secret_key),
                ("S3_BUCKET", bucket),
                ("SESSION_ID", session_id),
            ]
            if not value
        ]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        return cls(
            bucket=bucket or "",
            session_id=session_id or "",
            endpoint_url=endpoint or "",
            access_key=access_key or "",
            secret_key=secret_key or "",
            region=region,
        )

    def _make_s3_client(self):
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
            use_ssl=self.endpoint_url.startswith("https://"),
        )

    def _policy_key(self, version: int) -> str:
        return f"rl/{self.session_id}/policies/policy_{version:08d}.pt"

    def _manifest_key(self) -> str:
        return f"rl/{self.session_id}/manifests/latest.json"

    def _serialize_checkpoint(self, state_dict: dict) -> bytes:
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return buffer.getvalue()

    def _deserialize_checkpoint(self, blob: bytes) -> dict:
        try:
            payload = torch.load(
                io.BytesIO(blob),
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            raise InvalidCheckpointPayloadError(
                f"Failed to deserialize checkpoint payload: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise InvalidCheckpointPayloadError(
                f"Checkpoint payload must be a dict, got {type(payload)}"
            )
        return payload

    def _with_retries(self, op_name: str, fn):
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return fn()
            except (BotoCoreError, ClientError, OSError) as exc:
                if isinstance(exc, ClientError):
                    error_code = str(
                        getattr(exc, "response", {}).get("Error", {}).get("Code", "")
                    )
                    status_code = (
                        getattr(exc, "response", {})
                        .get("ResponseMetadata", {})
                        .get("HTTPStatusCode")
                    )
                    retryable_codes = {
                        "RequestTimeout",
                        "Throttling",
                        "SlowDown",
                        "InternalError",
                        "ServiceUnavailable",
                    }
                    is_retryable_http = (
                        isinstance(status_code, int) and status_code >= 500
                    )
                    is_retryable = error_code in retryable_codes or is_retryable_http
                    if not is_retryable:
                        raise RuntimeError(f"S3 operation '{op_name}' failed") from exc

                last_exc = exc
                if attempt == self.max_retries - 1:
                    break
                delay = self.retry_base_delay_s * (2**attempt) + random.uniform(
                    0.0, self.retry_jitter_s
                )
                self.log.warning(
                    "S3 operation '%s' failed on attempt %d/%d: %s; retrying in %.2fs",
                    op_name,
                    attempt + 1,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"S3 operation '{op_name}' failed after retries"
        ) from last_exc

    def _put_json(self, key: str, payload: dict) -> dict:
        encoded = json.dumps(payload).encode("utf-8")
        return self._put_bytes(key=key, payload=encoded)

    def _get_json(self, key: str) -> dict:
        try:
            blob = self._get_bytes(key)
            data = json.loads(blob.decode("utf-8"))
        except MissingCheckpointError as exc:
            if key == self._manifest_key():
                raise MissingManifestError(
                    f"Manifest not found at key '{key}'"
                ) from exc
            raise
        except json.JSONDecodeError as exc:
            raise InvalidCheckpointPayloadError(
                f"Invalid JSON payload at key '{key}': {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise InvalidCheckpointPayloadError(
                f"JSON payload at key '{key}' must be an object"
            )
        return data

    def _put_bytes(self, key: str, payload: bytes) -> dict:
        def _op():
            return self.s3.put_object(Bucket=self.bucket, Key=key, Body=payload)

        return self._with_retries("put_object", _op)

    def _get_bytes(self, key: str) -> bytes:
        def _op():
            return self.s3.get_object(Bucket=self.bucket, Key=key)

        try:
            response = self._with_retries("get_object", _op)
            body = response["Body"].read()
            if isinstance(body, str):
                return body.encode("utf-8")
            return body
        except RuntimeError as exc:
            cause = exc.__cause__
            if isinstance(cause, ClientError):
                code = getattr(cause, "response", {}).get("Error", {}).get("Code")
                if code in {"NoSuchKey", "404", "NotFound"}:
                    raise MissingCheckpointError(
                        f"Checkpoint key '{key}' not found in bucket '{self.bucket}'"
                    ) from cause
            raise

    def publish_actor(
        self,
        actor_state_dict: dict,
        version: int,
        metadata: dict | None = None,
    ) -> dict:
        policy_key = self._policy_key(version)
        if self.actor_exists(version):
            raise FileExistsError(
                f"Refusing to overwrite immutable policy object '{policy_key}'"
            )

        checkpoint = self._serialize_checkpoint(actor_state_dict)

        upload_start = time.perf_counter()
        put_resp = self._put_bytes(policy_key, checkpoint)
        upload_latency = time.perf_counter() - upload_start

        etag = put_resp.get("ETag")
        if isinstance(etag, str):
            etag = etag.strip('"')

        manifest = {
            "version": int(version),
            "bucket": self.bucket,
            "key": policy_key,
            "etag": etag,
            "published_at_unix": time.time(),
            "metadata": metadata or {},
        }

        manifest_resp = self._put_json(self._manifest_key(), manifest)
        self.log.info(
            "Published actor version=%d upload_latency_s=%.4f manifest_key=%s manifest_etag=%s",
            version,
            upload_latency,
            self._manifest_key(),
            manifest_resp.get("ETag"),
        )
        return manifest

    def get_latest_manifest(self) -> dict | None:
        try:
            manifest = self._get_json(self._manifest_key())
            self.log.info(
                "Latest manifest version=%s key=%s etag=%s",
                manifest.get("version"),
                manifest.get("key"),
                manifest.get("etag"),
            )
            return manifest
        except MissingManifestError:
            return None

    def download_actor(
        self,
        version: int | None = None,
        local_path: str | None = None,
    ) -> dict:
        manifest = self.get_latest_manifest() if version is None else None
        if version is None:
            if manifest is None:
                raise MissingManifestError(
                    f"Manifest not found at key '{self._manifest_key()}'"
                )
            target_version = int(manifest["version"])
        else:
            target_version = int(version)
        policy_key = self._policy_key(target_version)
        if manifest is not None and "key" in manifest:
            policy_key = str(manifest["key"])

        if local_path is None:
            cache_path = self.cache_dir / f"policy_{target_version:08d}.pt"
        else:
            cache_path = Path(local_path)

        if cache_path.exists():
            self.log.info(
                "Cache hit for actor version=%d path=%s", target_version, cache_path
            )
            blob = cache_path.read_bytes()
            return self._deserialize_checkpoint(blob)

        download_start = time.perf_counter()
        blob = self._get_bytes(policy_key)
        download_latency = time.perf_counter() - download_start
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(blob)
        self.log.info(
            "Fetched actor version=%d source=remote latency_s=%.4f path=%s",
            target_version,
            download_latency,
            cache_path,
        )
        return self._deserialize_checkpoint(blob)

    def latest_version(self) -> int | None:
        manifest = self.get_latest_manifest()
        if manifest is None:
            return None
        return int(manifest["version"])

    def actor_exists(self, version: int) -> bool:
        key = self._policy_key(version)

        def _op():
            return self.s3.head_object(Bucket=self.bucket, Key=key)

        try:
            self._with_retries("head_object", _op)
            return True
        except RuntimeError as exc:
            cause = exc.__cause__
            if isinstance(cause, ClientError):
                code = getattr(cause, "response", {}).get("Error", {}).get("Code")
                if code in {"NoSuchKey", "404", "NotFound"}:
                    return False
            raise


class PolicyFetcher:
    def __init__(self, param_server: S3ParameterServer, actor: torch.nn.Module, device):
        self.param_server = param_server
        self.actor = actor
        self.device = device
        self.log = logging.getLogger("PolicyFetcher")
        self.local_version: int | None = None

    def maybe_refresh(self, version: int | None = None) -> int | None:
        if version is None:
            manifest = self.param_server.get_latest_manifest()
            if manifest is None:
                return None

            latest = int(manifest["version"])
            self.log.info(
                "Latest version seen=%d local_version=%s", latest, self.local_version
            )
            if self.local_version is not None and latest <= self.local_version:
                return None
        else:
            latest = version
            if self.local_version is not None and latest <= self.local_version:
                self.log.info(
                    "Requested version=%d is not newer than local_version=%s",
                    latest,
                    self.local_version,
                )
                return None
        state_dict = self.param_server.download_actor(version=latest)
        self.actor.load_state_dict(state_dict, strict=False)
        self.actor.to(self.device)
        self.actor.eval()
        self.local_version = latest
        return latest
