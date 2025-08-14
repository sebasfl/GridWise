# Dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# System deps for building some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Workdir inside the container (Linux paths)
WORKDIR /app

# Install Python deps first (cache-friendly)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (small; data will be mounted as a volume at runtime)
COPY src ./src
COPY dash ./dash

# Optional: create folders so they exist inside the container
RUN mkdir -p /app/data /app/models /app/notebooks

# Non-root user (safer)
RUN useradd -m runner && chown -R runner:runner /app
USER runner

# Environment
ENV PYTHONUNBUFFERED=1 TZ=UTC

# Exposed ports (API 8000, Dashboard 8501)
EXPOSE 8000 8501

# Default command; compose will override with app-specific commands
CMD ["bash"]
