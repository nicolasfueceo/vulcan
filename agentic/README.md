# Agentic Core Logic Refactor

This directory will contain the new, modular, and fully-tested agentic core logic for the FuegoRecommender system. All code here will be written using strict TDD (tests first), with immediate linting and logical reflection after every step.

## Principles
- Test-Driven Development (TDD): Write tests before implementation for every module.
- Immediate Linting: Check and fix all linter issues after each code change.
- Logical Soundness: Reflect on correctness and design after every implementation.
- No contingency/ logic: This folder is focused on core agentic orchestration, agent management, and utilities only.

## Structure (initial)
```
agentic/
  core/           # Core logic (session state, database, orchestration)
  agents/         # Agent base classes and factories
  utils/          # Atomic utilities (validation, error handling, etc.)
  tests/          # All tests (unit, integration, TDD-first)
  README.md
```

## Getting Started
- All new modules must have corresponding tests in `agentic/tests/`.
- Run `pytest` and `flake8` (or `ruff`) after every code change.
- Document design reflections in this README or a `REFLECTIONS.md` if needed.
