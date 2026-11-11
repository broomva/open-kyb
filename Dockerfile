# syntax=docker/dockerfile:1
# The kyb *service* image (spec): API + chat tier on ECS Fargate. It is never the agent sandbox.
# Non-root (UID/GID 10001), read-only root filesystem friendly, no secrets or .env baked in.

ARG PYTHON_IMAGE=python:3.15-slim

# ---- build: resolve the locked environment with uv, no dev tools -------------------------------
FROM ${PYTHON_IMAGE} AS build
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
# The venv is built where it will live: console-script shebangs embed its absolute path.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/kyb
WORKDIR /src
# Dependencies first so the layer is reused until the lockfile changes.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --no-editable
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---- runtime: same base (so the venv's interpreter path matches), only what `kyb` needs ---------
FROM ${PYTHON_IMAGE} AS runtime
# poppler renders page crops for the witness read (no OCR engine, no other system tools).
RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 kyb \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin kyb \
    && mkdir /data \
    && chown 10001:10001 /data

COPY --from=build /opt/kyb /opt/kyb
ENV PATH="/opt/kyb/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    KYB_WORKSPACE=/data

# Writable paths when the root filesystem is read-only: mount /tmp and /data as volumes.
VOLUME ["/tmp", "/data"]
WORKDIR /data
USER 10001:10001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["kyb", "--help"]
ENTRYPOINT ["kyb"]
CMD ["--help"]
