# Contributing to Cat Watcher

Thank you for your interest in contributing to Cat Watcher! This document provides guidelines and information for contributors.

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all skill levels.

## Getting Started

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/cat-watcher.git
   cd cat-watcher
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # .venv\Scripts\activate   # Windows
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/cat_watcher --cov-report=html

# Run specific test file
pytest tests/unit/test_homeassistant.py -v

# Run only fast tests (skip integration)
pytest -m "not integration"
```

### Code Quality

Before submitting a PR, ensure your code passes all checks:

```bash
# Linting
ruff check src/ tests/

# Type checking
mypy src/cat_watcher

# Formatting
ruff format src/ tests/
```

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - Python version
   - OS and version
   - Steps to reproduce
   - Expected vs actual behavior
   - Relevant logs

### Suggesting Features

1. Check existing issues and discussions
2. Use the feature request template
3. Describe the use case and proposed solution

### Submitting Pull Requests

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes:**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   # or
   git commit -m "fix: resolve bug in X"
   ```
   
   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Adding or updating tests
   - `refactor:` Code refactoring
   - `chore:` Maintenance tasks

4. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

### PR Checklist

- [ ] Tests pass locally (`pytest`)
- [ ] Linting passes (`ruff check src/ tests/`)
- [ ] Type checking passes (`mypy src/cat_watcher`)
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventional commits
- [ ] PR description explains the changes

## Project Structure

```
cat-watcher/
├── src/cat_watcher/         # Main source code
│   ├── frigate/             # Frigate NVR integration
│   ├── collection/          # Data collection
│   ├── labeling/            # Labeling service
│   ├── training/            # ML training
│   ├── inference/           # Inference service
│   └── homeassistant/       # HA integration
├── tests/                   # Test suite
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── docs/                    # Documentation
├── k8s/                     # Kubernetes manifests
└── homeassistant/           # HA configs
```

## Development Guidelines

### Code Style

- Use type hints for all functions
- Write docstrings for public functions and classes
- Keep functions focused and small
- Use `structlog` for logging

### Testing

- Write unit tests for new functionality
- Use `pytest` fixtures for setup
- Mock external dependencies
- Aim for >80% coverage on new code

### Documentation

- Update README for user-facing changes
- Add docstrings for public APIs
- Update docs/ for significant features

## Areas for Contribution

### Good First Issues

Look for issues labeled `good first issue` - these are simpler tasks suitable for new contributors.

### Feature Ideas

- Additional behavior detection classes
- Support for more NVR systems
- Audio-based detection improvements
- Dashboard enhancements
- Performance optimizations

### Documentation

- Tutorials and guides
- API documentation
- Example configurations
- Translations

## Questions?

- Open a GitHub Discussion for questions
- Check existing documentation
- Review closed issues for solutions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
