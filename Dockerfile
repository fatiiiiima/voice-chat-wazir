# Agent image for the future Linux GPU server (see deploy/docker-compose.yml).
FROM python:3.12-slim
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY agent ./agent
COPY web ./web
# Bake models into the image so containers start fast.
RUN uv run python -m agent.download
ENV HOST=0.0.0.0 PORT=7860
CMD ["uv", "run", "python", "-m", "agent.server"]
