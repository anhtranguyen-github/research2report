# Installation Guide

This guide provides detailed instructions for installing the AI Trends Report Generator in various environments.

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- Git

## Local Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourorganization/update-latest-trends.git
cd update-latest-trends
```

### Step 2: Set Up a Virtual Environment

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install the Package

```bash
# Install the package in development mode with development dependencies
pip install -e ".[dev]"
```

## Docker Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourorganization/update-latest-trends.git
cd update-latest-trends
```

### Step 2: Configure Environment Variables

Copy the example environment file and update it with your credentials:

```bash
cp .env.example .env
# Edit .env with your text editor
```

### Step 3: Start Docker Containers

```bash
docker-compose up -d
```

This will start all required services:
- API server
- Database (if applicable)
- Vector store

## Cloud Deployment

For production deployments, we recommend using container orchestration platforms such as:

- Kubernetes
- AWS ECS
- Google Cloud Run

### Example: Deploying to Google Cloud Run

```bash
# Build the container
gcloud builds submit --tag gcr.io/your-project/update-latest-trends

# Deploy to Cloud Run
gcloud run deploy update-latest-trends \
  --image gcr.io/your-project/update-latest-trends \
  --platform managed \
  --memory 2Gi \
  --cpu 2 \
  --allow-unauthenticated
```

## Troubleshooting

### Common Issues

**Missing Dependencies:**
```
ERROR: Could not find a version that satisfies the requirement...
```

Solution: Try installing the specific version:
```bash
pip install package==version
```

**Permission Errors:**
```
PermissionError: [Errno 13] Permission denied
```

Solution: Try running with elevated privileges or use a virtual environment.

**Docker Issues:**
```
Error response from daemon: driver failed programming...
```

Solution: Check your Docker installation and ensure Docker daemon is running. 