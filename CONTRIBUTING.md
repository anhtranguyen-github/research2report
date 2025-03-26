# Contributing to AI Trends Report Generator

Thank you for considering contributing to this project! Here's how you can help.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/your-username/update-latest-trends.git
   cd update-latest-trends
   ```
3. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes**
5. **Run the tests**
   ```bash
   pytest
   ```
6. **Lint your code**
   ```bash
   black src tests
   isort src tests
   flake8 src tests
   ```
7. **Commit your changes**
   ```bash
   git commit -m "Add your descriptive commit message"
   ```
8. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
9. **Open a pull request**

## Development Setup

Follow the installation instructions in the [README.md](README.md) file. Make sure to install development dependencies:

```bash
pip install -e ".[dev]"
```

## Testing

Write tests for all new features and bug fixes. Run the test suite before submitting a pull request:

```bash
pytest
```

## Code Style

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Configuration for these tools is in `pyproject.toml` and `setup.cfg`.

## Documentation

Update documentation when necessary. This includes:
- Code comments
- Docstrings
- README updates
- Additional documentation in the `docs` folder

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the documentation if needed
3. The PR should work on the main branch
4. Your PR needs to pass all CI checks
5. Your PR needs to be reviewed by at least one maintainer

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE). 