# Contributing to Vela

Thank you for your interest in contributing! This guide will help you get started.

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/vela-causal.git
   cd vela-causal
   ```
3. **Install** dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## Development Workflow

### Running Tests

```bash
python tests/test_engine.py
```

All 28+ assertions must pass before submitting.

### Code Style

- Follow PEP 8 conventions.
- Use type hints for all public function signatures.
- Use descriptive variable names.
- Add docstrings to all public functions.

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add CSV batch training support
fix: handle empty input in tokenizer
docs: update API reference for /explain endpoint
test: add multi-hop inference tests
```

---

## What to Contribute

### Good First Issues

- Improve test coverage for edge cases
- Add more causal connector patterns to `extractor.py`
- Improve error messages
- Add type hints where missing

### Feature Ideas

- Graph visualization
- Batch training from CSV/JSON files
- Additional inference strategies
- Docker support

---

## Pull Request Process

1. Ensure all tests pass.
2. Update documentation if your change affects public APIs.
3. Keep PRs focused — one feature or fix per PR.
4. Write a clear PR description explaining what changed and why.
5. Link any related issues.

---

## Code of Conduct

This project follows a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold its standards.

---

## Questions?

Open a [discussion](https://github.com/aryan/vela-causal/discussions) or file an [issue](https://github.com/aryan/vela-causal/issues).
