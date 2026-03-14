FROM python:3.11-slim

WORKDIR /app

# Set environment variables to improve installation reliability
ENV PIP_DEFAULT_TIMEOUT=1000 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_RETRIES=10 \
    PDM_CHECK_UPDATE=false \
    PDM_USE_VENV=0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libgomp1 cmake git \
    && rm -rf /var/lib/apt/lists/*

# Install PDM
RUN pip install --no-cache-dir pdm

# Pre-install heavy dependencies via pip to leverage its robust retry/download logic
# This prevents one massive package failure from rolling back the entire group installation
# We match versions from the lockfile where possible
RUN pip install --no-cache-dir torch torchvision pyarrow

# Copy dependency files
COPY pyproject.toml pdm.lock* ./

# Install remaining dependencies from the lockfile
RUN pdm install --prod --no-editable --no-self -v

# Copy project source code
COPY src/ ./src/

# Final install to ensure everything is synced (should be fast due to caching)
RUN pdm install --prod --frozen-lockfile --no-editable

# Expose ports for both FastAPI and Gradio
EXPOSE 8000 7860

# Default command
CMD ["pdm", "run", "api"]
