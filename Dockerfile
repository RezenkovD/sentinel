# Sentinel — CPU image (no CUDA: torch and torchvision come from the PyTorch CPU index pinned in uv.lock).
# Weights, data, results and the .env are mounted at run time: datasets and secrets never go into the image.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /bin/

# opencv-python needs libGL and glib; tzdata gives Telegram messages the operator's local time
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 tzdata \
 && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HOME=/tmp \
    TZ=Europe/Kyiv \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

# dependencies first (cached layer), exactly as locked; the uv cache is a build mount, not part of the image
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project

COPY src ./src
COPY configs ./configs
COPY tests ./tests
COPY scripts ./scripts
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

ENTRYPOINT ["sentinel"]
CMD ["--help"]
