# Writing-Style Conventions

## Language

- Match the primary language of the repository's existing `docs/`:
  - If existing docs are in English → write in English.
  - If existing docs are in Chinese → write in Chinese.
  - If bilingual / mixed → default to the language of the user's latest query.
- Within a single document, do not mix languages mid-paragraph. Code comments inside snippets may remain in their original language.

## Tone

- Objective, instructional, second-person avoided. Prefer "Use X to do Y" over "You should use X".
- No marketing adjectives ("powerful", "amazing", "cutting-edge").
- Absolute claims ("always", "never") are acceptable only when grounded in source code.

## Code citation format

When referencing a symbol in prose, use the canonical form `path/relative/to/repo.py::Symbol` the first time it appears in a document. Subsequent mentions may drop the path.

Examples:
- First mention: ``The training loop is implemented in `deepxde/model.py::Model.train`.``
- Later mentions: ``During `Model.train` the callbacks fire in this order ...``

## Code snippets

- Use fenced code blocks with the correct language tag (`python`, `bash`, `rust`, …).
- Keep snippets ≤ 30 lines; elide non-essential parts with `# ...` and a comment indicating what was removed.
- Preserve the original repo's import style (e.g. `import deepxde as dde`).
- Never invent function names or parameters. If unsure, verify with `search_content` before committing.

## Tables

- Use GitHub-flavored Markdown tables for:
  - enumerating class hierarchies,
  - mapping environment variables to behaviors,
  - error → cause → fix triplets,
  - comparing backends / adapters.
- Keep rows short; move long explanations into the paragraph above/below.

## ASCII diagrams

- Allowed (encouraged) for architecture overviews.
- Keep width ≤ 72 characters.
- Do not use Unicode box-drawing characters heavier than `┌ ┐ └ ┘ ─ │ ├ ┤ ┬ ┴ ┼ ▶ ◀ ▲ ▼`.

## TODO markers

Use `> TODO(doc-miner): <short description>` as a blockquote when a piece of evidence was not found.

Rules:
- Each TODO lists the missing fact, not the proposed answer.
- Scan with `rg "TODO\(doc-miner\)"` before handover; resolve or list in the summary.

## Cross-document references

- `README.md` → links to all three other files in the same folder.
- `tutorial.md` references concrete sections of `user_guide.md` by anchor ID (Markdown auto-generates from headings).
- `developer_guide.md` references `user_guide.md` for high-level API semantics; avoids duplicating user-level how-tos.

## Section length heuristics

- A top-level section: 100–600 words + optional code block.
- Sub-section (`###`): 50–300 words.
- No section should be empty after Phase 4; if evidence is missing, insert a TODO marker.

## What to cut

- Generic ML or software-engineering tutorials that could apply to any project.
- Advertisements for the project ("the best PINN library").
- Historical anecdotes unless they explain a current design decision.
- Redundant re-statement of content already in the README (link instead).
