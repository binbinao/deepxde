# {{PROJECT_NAME}} 指南中心 / Guide Hub

This folder contains three coordinated documents derived from mining the {{PROJECT_NAME}} source tree:

| File | Audience | Contents |
| --- | --- | --- |
| [`developer_guide.md`](./developer_guide.md) | Framework developers / contributors | Architecture, sub-package tours, extension SOPs, debugging tips. |
| [`user_guide.md`](./user_guide.md) | End users / applied engineers | Installation, core API, typical workflows, FAQ. |
| [`tutorial.md`](./tutorial.md) | Learners at any level | 10-chapter "from beginner to expert" walkthrough grounded in `examples/`. |

> Upstream project: {{PROJECT_HOMEPAGE_OR_TODO}}
> Official docs: {{PROJECT_DOCS_OR_TODO}}

## How these documents were produced

- Based on direct reading of `{{SOURCE_PACKAGE_PATH}}/`, `examples/`, and existing `docs/` material.
- Every claim is grounded in a concrete source path; unresolved facts are marked with `> TODO(doc-miner): ...`.

## Reading order suggestions

1. **New users** — start with `user_guide.md` §1–§11, then do chapters 1–4 of `tutorial.md`.
2. **Applied engineers** with a specific task — jump to the matching chapter of `tutorial.md`.
3. **Contributors** — read `developer_guide.md` end to end; cross-reference `user_guide.md` when a user-facing API is discussed.
