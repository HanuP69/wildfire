FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PDM
RUN pip install pdm

# Copy project files
COPY pyproject.toml pdm.lock* ./
COPY src/ ./src/

# Install dependencies directly into the system for Docker
RUN pdm install --prod --frozen-lockfile --no-editable

# Expose ports for both FastAPI and Gradio
EXPOSE 8000 7860

# We use docker-compose commands to specify whether this container
# runs the API or the Frontend
CMD ["pdm", "run", "api"]
