FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Copy the rest of the application
COPY . .

# Expose the API port
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "src.server.main:app", "--host", "0.0.0.0", "--port", "8000"] 