# Documentation

This directory contains detailed documentation for the AI Trends Report Generator project.

## Contents

- [Installation Guide](installation.md) - Detailed installation instructions for various environments
- [Configuration Guide](configuration.md) - How to configure the system for different use cases
- [Model Reference](models.md) - Information about the AI models supported by the system
- [API Reference](api.md) - Documentation for the REST API endpoints

## Generated Documentation

To generate API documentation:

```bash
# Install documentation tools
pip install sphinx sphinx-rtd-theme

# Generate API documentation
sphinx-build -b html docs/source docs/build/html
```

The generated documentation will be available in the `docs/build/html` directory. 