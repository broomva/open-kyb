"""Approved GCP Vertex Gemini transport (vertex-gcp).

Project, quota project, and location are read from environment variables (KYB_VERTEX_PROJECT,
KYB_VERTEX_QUOTA_PROJECT, KYB_VERTEX_LOCATION, default global). Auth is an OAuth token from
Application Default Credentials via gcloud, held in memory for one request and never logged,
persisted, or placed in argv/env. Developer/AI Studio API keys are never consulted. Outcome and
evidence handling is shared with GeminiWitness.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.request
from collections.abc import Callable
from typing import Any

from kyb.witness.errors import WitnessRefused
from kyb.witness.gemini import RETRIES, GeminiWitness

TRANSPORT = "vertex-gcp"
DEFAULT_LOCATION = "global"
HOST = "aiplatform.googleapis.com"
TOKEN_TIMEOUT_S = 30
_MODEL = re.compile(r"gemini-[a-z0-9][a-z0-9.\-]{0,60}")
_PROJECT = re.compile(r"[a-z][a-z0-9\-]{4,28}[a-z0-9]")
_LOCATION = re.compile(r"global|[a-z]+-[a-z0-9]+")
_TOKEN = re.compile(r"[A-Za-z0-9._~+/=\-]{20,4096}")
_GCLOUD_ENV = ("PATH", "HOME", "CLOUDSDK_CONFIG", "LANG", "LC_ALL")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def adc_token() -> str:
    """OAuth access token from the existing local ADC; failures never echo gcloud output."""
    gcloud = shutil.which("gcloud")
    if gcloud is None:
        raise WitnessRefused("Vertex auth unavailable: gcloud not found")
    env = {k: os.environ[k] for k in _GCLOUD_ENV if k in os.environ}
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [gcloud, "auth", "application-default", "print-access-token"],
            capture_output=True,
            text=True,
            timeout=TOKEN_TIMEOUT_S,
            env=env,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except OSError, subprocess.SubprocessError:
        raise WitnessRefused("Vertex auth failed: token acquisition error") from None
    token = done.stdout.strip()
    if done.returncode != 0 or not _TOKEN.fullmatch(token):
        raise WitnessRefused("Vertex auth failed: no valid access token")
    return token


class VertexWitness(GeminiWitness):
    transport = TRANSPORT

    def __init__(
        self,
        *,
        model: str | None = None,
        retries: int = RETRIES,
        project: str | None = None,
        quota_project: str | None = None,
        location: str | None = None,
        host: str = HOST,
        token_source: Callable[[], str] | None = None,
        urlopen: Callable[..., Any] | None = None,
    ) -> None:
        """token_source and urlopen exist only so tests can run with zero network."""
        super().__init__(model=model, retries=retries)
        env_project = os.environ.get("KYB_VERTEX_PROJECT", "").strip()
        env_quota = os.environ.get("KYB_VERTEX_QUOTA_PROJECT", "").strip() or env_project
        env_location = os.environ.get("KYB_VERTEX_LOCATION", "").strip() or DEFAULT_LOCATION
        resolved_project = env_project if project is None else project
        resolved_quota = (env_quota or resolved_project) if quota_project is None else quota_project
        resolved_location = env_location if location is None else location
        if (
            host != HOST
            or not isinstance(resolved_project, str)
            or not _PROJECT.fullmatch(resolved_project)
            or not isinstance(resolved_quota, str)
            or not _PROJECT.fullmatch(resolved_quota)
            or not isinstance(resolved_location, str)
            or not _LOCATION.fullmatch(resolved_location)
            or (env_project and resolved_project != env_project)
            or (env_quota and resolved_quota != env_quota)
            or (resolved_location != env_location)
        ):
            raise WitnessRefused("Vertex route is not the approved project/location/host")
        if not _MODEL.fullmatch(self.model):
            raise WitnessRefused("Vertex model must be one exact gemini model id")
        self.project = resolved_project
        self.quota_project = resolved_quota
        self.location = resolved_location
        self.route_identity = {
            "host": host,
            "project": resolved_project,
            "location": resolved_location,
            "quota_project": resolved_quota,
            "api": "v1:generateContent",
        }
        self.url = (
            f"https://{host}/v1/projects/{resolved_project}/locations/{resolved_location}"
            f"/publishers/google/models/{self.model}:generateContent"
        )
        self._token_source = token_source or adc_token
        self._urlopen = (
            urlopen
            or urllib.request.build_opener(
                urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(), _NoRedirect()
            ).open
        )
        self.route_identity["injected_io"] = bool(token_source or urlopen)

    def _shape(self, body: dict) -> dict:
        return {**body, "contents": [{"role": "user", **c} for c in body["contents"]]}

    def _opener(self) -> Callable[..., Any]:
        def open_request(body: dict, *, timeout: float):
            try:
                token = self._token_source()
            except WitnessRefused:
                self.last_provenance["refusal_evidence"] = ["vertex_auth_failed"]
                raise
            except Exception:
                self.last_provenance["refusal_evidence"] = ["vertex_auth_failed"]
                raise WitnessRefused("Vertex auth failed") from None
            if not isinstance(token, str) or not _TOKEN.fullmatch(token):
                self.last_provenance["refusal_evidence"] = ["vertex_auth_failed"]
                raise WitnessRefused("Vertex auth failed: no valid access token")
            headers = {
                "Authorization": f"Bearer {token}",
                "x-goog-user-project": self.quota_project,
                "Content-Type": "application/json",
            }
            req = urllib.request.Request(
                self.url, json.dumps(body).encode(), headers, method="POST"
            )
            return self._urlopen(req, timeout=timeout)

        return open_request


__all__ = ["DEFAULT_LOCATION", "HOST", "TOKEN_TIMEOUT_S", "TRANSPORT", "VertexWitness", "adc_token"]
