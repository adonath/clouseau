# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Clouseau is a library-independent forward-pass inspector for neural nets. It records the
intermediate activations of a model's forward pass to `.safetensors` files for later inspection,
which is useful for debugging or checking equivalence when porting a model between frameworks.
It supports both **PyTorch** and **JAX/Equinox** models.

## Commands

This project uses `uv`. Most workflows go through the `Makefile`:

- `make install` — `uv sync` + install pre-commit hooks
- `make check` — verify lock file, run pre-commit (ruff lint/format), and run `ty check` (the type checker; configured under `[tool.ty]` in `pyproject.toml`)
- `make test` — run pytest with coverage
- `make docs` — serve the mkdocs site locally; `make docs-test` builds it strictly
- `tox` — run the test suite across Python 3.11–3.13 (also runs in CI)

Run a single test: `uv run python -m pytest tests/test_inspector.py::test_jax -v`

The CLI entry point is `clouseau` (defined in `clouseau/__main__.py`), e.g.
`clouseau show --filename activations.safetensors --key-pattern "0."` or
`clouseau diff --filename a.safetensors --filename-ref b.safetensors`.

## Architecture

The package has two largely independent halves: **recording** activations and **visualizing** them.

### Recording (`inspector.py` + framework utils)

`inspector.tail(model, ...)` returns a `_Recorder` context manager. On `__enter__` it detects the
framework (`is_torch_model` / `is_jax_model`) and wraps the model; on `__exit__` it tears the
wrapping down and flushes the cache to disk. The two frameworks need fundamentally different
mechanics, isolated in `torch_utils.py` and `jax_utils.py`:

- **PyTorch** (`torch_utils.py`): registers forward hooks on each submodule that write the layer
  output into a module-global `CACHE`. Hooks are de-registered on exit.
- **JAX/Equinox** (`jax_utils.py`): JAX is functional, so hooks don't work. Instead each callable
  node in the model pytree is replaced with a `_ClouseauJaxWrapper`, and recording happens as a
  side effect via `jax.experimental.io_callback` into a module-global `CACHE`. The `is_leaf`
  callback controls traversal granularity. (JAX callers should `.block_until_ready()` inside the
  context so the async callbacks complete before flush.)

Both framework modules hold their own module-level `CACHE: ArrayCache`. `ArrayCache` (in
`io_utils.py`) accumulates arrays keyed by dotted path and auto-flushes to numbered safetensors
files (`activations-{idx:03d}.safetensors`) once `max_size_mb` is exceeded — this is why a single
recording may produce multiple files.

`inspector.magnify(filename)` reads a file back and prints a tree view (the interactive counterpart
to the `clouseau show` CLI).

### I/O (`io_utils.py`)

Paths are flattened/unflattened with `PATH_SEP = "."`. `read_from_safetensors` supports a
`key_pattern` regex to select a subset of leaves. Framework-specific savers are dispatched through
`WRITE_REGISTRY` keyed by `FrameworkEnum`.

### Visualization (`visualize.py` + `__main__.py`)

Rendering is driven by `FORMATTER_REGISTRY`, a dict mapping a leaf's Python type to a formatter
callable, consumed by `print_tree` / `_add_dict_to_tree`. The CLI **mutates this registry at
runtime** to compose output: `show` registers a combined shape+stats+values formatter for
`np.ndarray`; `diff` registers an `ArrayDiffFormatter` for `tuple` (the merged `(value, ref)` pairs).
When adding a new formatter or output mode, follow this registry pattern rather than special-casing
inside the tree walker.

The CLI uses `tyro.cli` over a `Commands = Show | Diff` union of dataclasses; formatters are nested
dataclass fields, so their parameters are exposed automatically as CLI flags.

## Conventions

- Type checking is strict (`ty check`); keep functions fully annotated.
- Ruff is configured with a broad ruleset including flake8-bandit (`S`); `tests/*` ignore `S101`
  (asserts). Line length lint (`E501`) is disabled but `line-length = 88` governs formatting.
- New functionality should add tests under `tests/` and, per the contributing guide, be added to
  the feature list in `README.md`.
