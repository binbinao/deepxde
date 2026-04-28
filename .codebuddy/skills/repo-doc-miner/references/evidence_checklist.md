# Evidence Checklist

Minimum files / symbols that MUST be read during Phase 2 before writing any prose. Pick the repo-flavor row that best matches the target project; multiple rows may apply.

## Universal (always)

| Target | Why |
| --- | --- |
| `README.md` | Author-curated capability list, install commands, citation. |
| `LICENSE` | Licence identifier for metadata snapshot. |
| `CITATION.cff` (if present) | Canonical citation. |
| Top-level build/metadata file (`pyproject.toml` / `setup.py` / `package.json` / `Cargo.toml` / `go.mod` / `pom.xml`) | Package name, supported language versions, dependency list, entry points. |
| `requirements*.txt` / `environment.yml` (if present) | Runtime extras not covered by build file. |
| `.github/workflows/` (top-level only) | CI install commands, supported platforms. |
| `Dockerfile` / `docker/` | GPU / reproducible install hints. |
| `Makefile` | Common developer commands. |
| Existing `docs/` tree listing | Avoid style / vocabulary drift. |

## Python ML / scientific-computing library (e.g. DeepXDE)

| Target | Read strategy |
| --- | --- |
| `<pkg>/__init__.py` | Full file; record `__all__` and alias re-exports. |
| `<pkg>/model.py` (or `trainer.py`) | Full `Model.__init__`, `.compile`, `.train`, `.predict`, `.save`, `.restore`. |
| `<pkg>/config.py` | All public setters. |
| `<pkg>/backend/` or `<pkg>/engine/` | `__init__.py`, `backend.py` (interface spec), loader logic. |
| `<pkg>/data/__init__.py` | Full `__all__` list; sample one concrete data class. |
| `<pkg>/geometry/__init__.py` (if applicable) | Full `__all__` list; read base `Geometry` class signature. |
| `<pkg>/icbc/` or `<pkg>/constraints/` | Base class + all BC/IC subclasses' signatures. |
| `<pkg>/nn/__init__.py` | Backend-dispatch mechanism; one concrete network per supported backend. |
| `<pkg>/gradients/` or AD module | `jacobian` / `hessian` / `clear` contract. |
| `<pkg>/optimizers/__init__.py` + `config.py` | Supported optimizer names, per-backend LR schedules. |
| `<pkg>/callbacks.py` | Every callback class and its lifecycle method. |
| `<pkg>/losses.py` / `metrics.py` | Full dict of supported identifiers. |
| `examples/<forward>/*.py` | Pick 2–3 short forward examples. |
| `examples/<inverse>/*.py` | Pick 1 inverse example (usually Lorenz-style parameter identification). |
| `examples/<operator>/*.py` | Pick 1 operator-learning example. |

## Python CLI / web-service

| Target | Read strategy |
| --- | --- |
| `<pkg>/__main__.py` / `cli.py` | Entry point, subcommand map. |
| `<pkg>/server.py` / `app.py` / `main.py` | Framework (FastAPI/Flask/…), routing table. |
| `<pkg>/config.py` / `settings.py` | Env vars, default ports. |
| `<pkg>/handlers/` or `routers/` | Enumerate routes; sample 1 handler. |
| `openapi.json` / `swagger/` | If present, extract endpoint list. |

## Node / TypeScript library

| Target | Read strategy |
| --- | --- |
| `package.json` | `main`, `types`, `exports`, `scripts`. |
| `src/index.ts` | Public re-exports. |
| `tsconfig.json` | Language target, module system. |
| `src/<core>/` | Main class(es) + public types. |
| `examples/` or `demo/` | One runnable example. |

## Rust crate

| Target | Read strategy |
| --- | --- |
| `Cargo.toml` | crate name, edition, features. |
| `src/lib.rs` / `src/main.rs` | `pub use` re-exports. |
| `src/<module>/mod.rs` | Public types per module. |
| `examples/` | One small example. |

## Go module

| Target | Read strategy |
| --- | --- |
| `go.mod` | Module path, Go version. |
| Top-level `*.go` files with `package <name>` | Exported types (Capitalized). |
| `cmd/` subfolders | Binaries. |
| `internal/` vs. `pkg/` split | Public vs. private surface. |
| `examples/` or `_examples/` | One runnable. |

## Output of Phase 2

Maintain a scratch note (internal, not a deliverable) with:

- `name`, `version`, `license`, `python_requires` / `engines` / `rust_edition`, dependency list.
- `__all__` per top-level sub-package.
- Main class signatures (constructor + 3–5 key methods).
- List of chosen example files with a 1-line summary each.

This note feeds directly into the template `{{PLACEHOLDER}}` substitutions in Phase 4.
