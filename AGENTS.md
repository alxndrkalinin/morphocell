# Project Guide for AI Agents

This file provides guidance for AI agents working with the **morphocell** repository.

## Project Overview

*morphocell* is a Python library for morphometric analysis of multidimensional bioimages with optional CUDA acceleration.  Source code lives under the `morphocell/` package and tests are located in `tests/`.

## Directory Structure

- `morphocell/` – Python package containing all library modules
- `tests/` – pytest test suite
- `examples/` – example notebooks and data (read‑only)
- `build/` – build artefacts (should not be modified)
- `.github/workflows/` – CI configuration

## Coding Conventions

- Target Python version is **3.10+** and type annotations are required.
- Follow the existing style: snake_case for functions, PascalCase for classes, and triple‑quoted docstrings.
- Ruff is used for linting and formatting; run with automatic fixes when possible.
- Keep functions and classes concise with descriptive names and inline comments for complex logic.

## Programmatic Checks

Before committing, ensure the following commands succeed from the repository root:

```bash
ruff check .
ruff format --check .
mypy --ignore-missing-imports morphocell/
pytest
```

Tests may skip automatically when GPU hardware is not available but should still be executed.

## Pull Request Guidelines

- Provide a clear description of the change and its rationale.
- Ensure all programmatic checks pass.
- Keep PRs focused on a single objective and reference related issues when relevant.

