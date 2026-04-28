# Canonical Outlines

Authoritative section skeletons for the three documents. Specialize every bracketed slot with concrete names harvested in Phase 2 of the workflow. Section order is fixed; numbering may be adjusted if a phase is clearly inapplicable (mark the skip reason in the final handover summary).

---

## developer_guide.md (15 sections)

1. **Project positioning & scope** — one paragraph on problem domain; bullet list of the headline capabilities.
2. **Metadata snapshot** — license, supported languages/runtimes, dependency list, version source.
3. **Repository layout** — annotated tree of top-level folders and key files.
4. **Runtime architecture overview** — ASCII diagram linking the main data/model/runtime classes.
5. **Top-level namespace** — table of exported aliases vs. real module paths (from `__init__` / `index`).
6. **Backend / platform abstraction (if any)** — unified interface, adapter selection logic, enabled-API registry.
7. **Data & domain layer** — every data container, geometry primitive, constraint/BC/IC type; evidence: sub-package `__init__.py`.
8. **Automatic differentiation / computation core** — how derivatives / gradients / residuals are computed; caching and cleanup contract.
9. **Networks / models catalog** — list of built-in network or model classes with their constructor signatures.
10. **Optimizers & schedulers** — available optimizer names; per-backend/framework adapters; LR-decay matrix.
11. **Training main loop** — step-by-step narrative of the Model/Trainer.compile / .train / .predict / .save / .restore path.
12. **Callback / hook system** — base class lifecycle; enumeration of built-in callbacks and their trigger phase.
13. **Loss & metric registries** — mapping from string identifier to implementation; rules for contributing new ones.
14. **Parallel / distributed training** — what is supported, how to enable, known limitations.
15. **Contribution SOPs & debugging** — dev-env setup, adding a new backend/BC/network, PR flow, common error → root-cause table.

Grounding rules:
- Every section MUST include at least one `path/to/file.py::Symbol` reference the first time it is introduced.
- Sub-package tours (§5–§13) MUST enumerate the real `__all__` list, not a summarized approximation.

---

## user_guide.md (20 sections)

1. What the library can solve / do (user-facing capability list).
2. Installation — pip / conda / docker / source; per-backend or per-extras matrix.
3. Choosing a backend / runtime (if applicable) — decision table.
4. Global configuration — precision, seed, autodiff mode, JIT, verbosity.
5. Core-object map — one ASCII picture linking the user-facing classes.
6. Geometry / domain definition — primitives, composition (CSG), sampling strategies.
7. Boundary / initial conditions — all supported types with signatures.
8. Data objects — forward problem, time-dependent, inverse, operator-learning, multifidelity.
9. PDE residual / constraint authoring — automatic differentiation helpers.
10. Networks — built-ins with minimal constructor examples; input/output transforms.
11. Training — compile signature, optimizer options, loss_weights, callbacks.
12. Inverse problems — `Variable`, observation BCs, `VariableValue` callback.
13. Operator learning — DeepONet / PI-DeepONet / MIONet (or the repo's analogue).
14. Uncertainty quantification & multifidelity (if available).
15. Persistence — save/restore, portable format caveats.
16. Visualization & post-processing — built-in plotting helpers, residual inspection.
17. Parallel training (if applicable).
18. Troubleshooting — curated FAQ table.
19. Minimal runnable templates — ≥2 end-to-end snippets (forward problem + inverse problem or operator-learning problem).
20. Further reading — pointers to official docs, papers, tutorial.

Grounding rules:
- Every API shown MUST exist in the current repo; verify via `search_content` before citing.
- Prefer signatures over prose when introducing a class.

---

## tutorial.md (10 chapters)

1. Environment setup & "Hello" example (simplest possible runnable task).
2. Domain / input space basics — geometry, sampling.
3. Constraints — soft vs. hard, including hard-constraint ansatz via output transforms or distance functions.
4. Time-dependent problem — a concrete time-dependent example from `examples/`.
5. Accuracy & convergence techniques — adaptive sampling, Adam → L-BFGS pipeline, loss weighting, LR decay, early stopping.
6. Inverse problem — parameter identification with real observation data.
7. Complex geometry / high-dim / multiscale / fractional (pick those the repo supports).
8. Operator learning — data-driven and physics-informed variants.
9. Advanced acceleration — mixed precision, forward-mode AD, ZCS / caching tricks, multi-GPU.
10. Mastery path — custom callbacks, custom networks, custom constraints, custom loss, (optional) custom backend.

Grounding rules:
- Every chapter MUST reference at least one real script in `examples/` (relative path).
- Code snippets are condensed but runnable; do not invent functions.
- End each chapter with a 1–2 item "small exercise" that builds on the preceding material.
