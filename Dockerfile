FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN pip install --no-cache-dir uv

# Copy dependency and package metadata first so dependency installation can be cached
# independently from the rest of the repository contents.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --extra dev

COPY . .

ENV PATH="/app/.venv/bin:${PATH}"

CMD ["distill-abm", "--help"]
