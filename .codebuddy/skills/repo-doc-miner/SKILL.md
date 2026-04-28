---
name: repo-doc-miner
description: Mine an existing Git repository and generate three coordinated Markdown documents (developer guide, user guide, and from-beginner-to-expert tutorial) saved into the repository's docs folder. Use this skill when the user asks to "dig into / mine / analyze this project and produce documentation", "generate developer/user/tutorial docs for this repo", "write onboarding docs from the codebase", or any similar request where the deliverable is a coordinated doc set derived from reading the current repository.
---

# Repo Doc Miner

## Overview

Generate a coordinated set of three Markdown documents — **developer guide**, **user guide**, and **from-beginner-to-expert tutorial** — directly from a Git repository's source code, configuration, and examples. The output targets three distinct audiences (framework contributors, end users, and learners) and is grounded in concrete code paths, class/function names, and runnable examples from the repo, not generic knowledge.

## When to Use

Trigger this skill when the user's request matches any of these patterns:

- "挖掘当前项目/仓库，生成开发文档、用户文档和教程"
- "Mine this repo and produce developer / user / tutorial docs"
- "为这个项目写一套完整的文档（开发者 + 用户 + 入门到精通）"
- "Generate onboarding documentation for this codebase, save into docs/"
- Any request where (a) the primary input is the current repository, and (b) the required output is a multi-audience doc set written in Markdown.

Do NOT trigger for: single-file API docs, README refresh only, translation tasks, or when the user already has a target doc framework (Sphinx/MkDocs site generation).

## Workflow

Follow these six phases in order. Each phase lists what to do and which bundled resource to use.

### Phase 1 · Scope & Output Location

1. Determine the repository root (usually the workspace root).
2. Choose the output directory. Default: `docs/guides/` under the repo root. If `docs/` does not exist, create `docs/guides/` anyway (it is a self-contained sub-folder and will not clash with Sphinx/MkDocs). Prefer an existing docs directory if it is clearly the convention of the repo.
3. Confirm the three target files: `README.md` (navigation), `developer_guide.md`, `user_guide.md`, `tutorial.md`.

### Phase 2 · Repository Reconnaissance

Collect the minimum evidence required to write grounded docs. Issue parallel read-only tool calls wherever possible.

1. **Project metadata**: read `README.md`, `pyproject.toml` / `setup.py` / `package.json` / `Cargo.toml` / `go.mod`, `LICENSE`, `requirements*.txt`, `CITATION.cff`.
2. **Top-level layout**: list the repository root and the main source package(s); note file counts per extension.
3. **Public API surface**: read the top-level `__init__.py` (Python) / `index.ts` / `mod.rs` / `lib.rs` to learn what is exported. Record the `__all__` list and alias re-exports.
4. **Core classes**: identify the main Model/Engine/Trainer/App class and read its key methods (constructor, compile/build, train/run, predict/serve, save/load).
5. **Sub-packages**: for every important sub-package (data, nn, backend, optimizers, callbacks, geometry, icbc, utils …), open its `__init__.py` to harvest the public symbols; sample 1–2 representative implementations inside.
6. **Configuration**: read any global config module (e.g. `config.py`, `settings.py`, `constants.py`).
7. **Examples & tests**: list `examples/`, `tutorials/`, `samples/`, `tests/`, pick 3–6 representative scripts that cover forward / inverse / advanced usage, and read the shortest fully-runnable one in full.
8. **Docs folder**: list existing RST/MD under `docs/` to avoid duplication and align style/vocabulary.
9. **CI / Docker / packaging**: scan `.github/workflows/`, `Dockerfile`, `docker/`, `Makefile` for install and run commands that belong in the user guide.

Use `codebase_search` for open-ended "how / where / what" questions, and `search_content` (ripgrep) for exact symbol lookups. Batch independent reads in one tool-call block.

### Phase 3 · Content Plan (one-screen outline per document)

Before writing prose, draft a one-screen outline for each of the three documents. Reuse the canonical outlines in `references/doc_outlines.md` and specialize their section titles with concrete names harvested in Phase 2 (package names, class names, config flags). Verify that:

- The **developer guide** covers architecture, every top-level sub-package, runtime config, extension/contribution SOPs, and debugging tips.
- The **user guide** walks from installation to the most advanced user-facing feature, using API names that actually exist in the repo.
- The **tutorial** has 8–12 progressive chapters, each anchored in at least one runnable example file from the repo.

### Phase 4 · Writing

Use the templates under `assets/templates/` as the structural scaffold, then fill them with repo-specific content. Follow these rules:

- **Ground every claim in code**: when mentioning a function, class, or behavior, include the file path (relative to repo root) the first time it is introduced (e.g. `deepxde/model.py::Model.compile`). This lets future readers verify the claim.
- **Prefer real code snippets** taken from `examples/` or tests; condense to ≤ ~30 lines. Preserve the original import style and function signatures; do not invent APIs.
- **Cross-link the three documents**: the tutorial references sections in the user guide; the developer guide references the user guide for API usage; `README.md` links all three.
- **Match the repo's host language**: if the existing `docs/` content is primarily Chinese, write in Chinese; primarily English, write in English; bilingual repositories default to the language of the user's current query.
- **Keep each file self-contained**: duplicate a minimum amount of glossary/context so any one file can be opened first.
- **Mark unresolved gaps explicitly** with `> TODO(doc-miner): ...` lines when evidence is insufficient; never fabricate behavior.

### Phase 5 · Validation

Before reporting completion:

1. Spot-check 5 random code snippets against their source files (open the referenced path, compare signatures).
2. Run `rg "TODO\(doc-miner\)"` to surface unresolved gaps; either resolve them or list them in the final summary.
3. Ensure every sub-package mentioned in the developer guide actually exists (`search_file`).
4. Confirm the three files render correctly as Markdown (headings monotonic, code fences closed, relative links valid).

### Phase 6 · Handover

Produce a short final summary containing:

- Absolute paths of the four generated files.
- Line-count per file.
- A coverage bullet list (which sub-packages / examples made it in).
- Any `TODO(doc-miner)` items the user still needs to fill.

## Resources

### assets/templates/

Three Markdown skeletons carrying a fixed section schema. Use them as the literal starting point for each file; do **not** rewrite them from scratch. Replace every `{{PLACEHOLDER}}` with repo-specific content harvested in Phase 2.

- `assets/templates/developer_guide.md` — 15-section schema (overview → architecture → sub-package tours → extension SOPs → debugging).
- `assets/templates/user_guide.md` — 20-section schema (install → backend/runtime config → geometry/data/network equivalents → training loop → advanced features → FAQ → minimal runnable examples).
- `assets/templates/tutorial.md` — 10-chapter "from beginner to expert" schema with slots for real example file paths.
- `assets/templates/README.md` — navigation hub that indexes the three documents.

### references/

Loaded into context on demand.

- `references/doc_outlines.md` — canonical section-by-section outlines, with guidance on what each section must contain and how to source the evidence.
- `references/evidence_checklist.md` — the minimum set of files and symbols to read during Phase 2, grouped by repo flavor (Python DL library, Python CLI, Node web app, Rust crate, Go service).
- `references/writing_style.md` — writing-style conventions (tone, bilingual rules, code-citation format, TODO markers).

### scripts/

- `scripts/scaffold_docs.py` — creates `<repo>/docs/guides/` and materializes the four templates from `assets/templates/` with their `{{PROJECT_NAME}}` placeholder substituted. Usage:

  ```bash
  python scripts/scaffold_docs.py <repo-root> --project-name "<DisplayName>" [--out docs/guides]
  ```

  Run this at the very beginning of Phase 4 to lay down the skeleton; then edit each file in place.

## Failure Modes to Avoid

- Writing generic docs that could apply to any library — **always include at least one concrete class name, file path, or example per section**.
- Copying long code blocks verbatim (> 50 lines). Condense and cite the path instead.
- Skipping Phase 2 and guessing the API. If a fact cannot be grounded, mark `TODO(doc-miner)` and continue.
- Clobbering existing `docs/` content. Always write into a dedicated sub-folder (`docs/guides/` by default).
- Forgetting to produce the navigation `README.md`; without it the three documents feel disconnected.
