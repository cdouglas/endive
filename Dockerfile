# Dockerfile for Endive - Iceberg Catalog Simulator
#
# This image runs the Iceberg catalog saturation experiments and exports
# results via a mounted volume.
#
# Usage:
#   docker build -t endive-sim .
#   docker run -v $(pwd)/experiments:/app/experiments endive-sim
#
# Or use docker-compose:
#   docker-compose up

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir tomli

# Copy source code
COPY endive/ ./endive/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY experiment_configs/ ./experiment_configs/
COPY pyproject.toml .
COPY README.md .

# Install the package in development mode
RUN pip install -e .

# Create directories for outputs
RUN mkdir -p /app/experiments /app/plots /app/experiment_logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Volume for experiment results
VOLUME ["/app/experiments", "/app/plots", "/app/experiment_logs"]

# Default command: run baseline experiments
# Can be overridden with docker run command
CMD ["bash", "-c", "scripts/run_baseline_experiments.sh 2>&1 | tee experiment_logs/run_$(date +%Y%m%d_%H%M%S).log"]
