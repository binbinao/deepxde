#!/usr/bin/env python3
"""Scaffold the three-doc output for the repo-doc-miner skill.

Creates <repo-root>/<out>/ and materializes the four Markdown templates
(README, developer_guide, user_guide, tutorial) from the skill's
assets/templates/ directory, with a single {{PROJECT_NAME}} placeholder
substituted. All other {{...}} placeholders are intentionally left in
place so the CodeBuddy agent can fill them during Phase 4 of the
workflow described in SKILL.md.

Usage
-----
    python scripts/scaffold_docs.py <repo-root> \
        --project-name "MyProject" \
        [--out docs/guides] \
        [--force]

Exit codes
----------
    0  success
    1  templates directory not found
    2  output directory already populated and --force not supplied
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

TEMPLATE_FILES = (
    "README.md",
    "developer_guide.md",
    "user_guide.md",
    "tutorial.md",
)


def locate_templates_dir() -> Path:
    """Resolve the templates directory relative to this script's location."""
    here = Path(__file__).resolve().parent
    candidate = here.parent / "assets" / "templates"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"Could not find templates directory at {candidate}. "
        "The skill package appears to be corrupt."
    )


def materialize(
    templates_dir: Path,
    out_dir: Path,
    project_name: str,
    force: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = [p for p in out_dir.iterdir() if p.is_file()]
    overlap = [p for p in existing if p.name in TEMPLATE_FILES]
    if overlap and not force:
        names = ", ".join(sorted(p.name for p in overlap))
        raise SystemExit(
            f"Refusing to overwrite existing files in {out_dir}: {names}. "
            "Re-run with --force to overwrite."
        )

    for name in TEMPLATE_FILES:
        src = templates_dir / name
        dst = out_dir / name
        text = src.read_text(encoding="utf-8")
        # Substitute only the project-name placeholder; all other
        # {{...}} placeholders are intentionally preserved so the
        # agent can fill them with evidence from Phase 2 of the
        # skill workflow.
        text = text.replace("{{PROJECT_NAME}}", project_name)
        dst.write_text(text, encoding="utf-8")
        print(f"  wrote {dst}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scaffold the repo-doc-miner three-doc output."
    )
    parser.add_argument(
        "repo_root",
        type=Path,
        help="Absolute or relative path to the target repository's root.",
    )
    parser.add_argument(
        "--project-name",
        required=True,
        help="Display name of the project (used to replace {{PROJECT_NAME}}).",
    )
    parser.add_argument(
        "--out",
        default="docs/guides",
        help="Output sub-directory under repo-root (default: docs/guides).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing template files in the output directory.",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    if not repo_root.is_dir():
        print(f"error: repo root does not exist: {repo_root}", file=sys.stderr)
        return 1

    out_dir = (repo_root / args.out).resolve()
    try:
        templates_dir = locate_templates_dir()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Scaffolding docs for project '{args.project_name}'")
    print(f"  templates: {templates_dir}")
    print(f"  output:    {out_dir}")
    materialize(templates_dir, out_dir, args.project_name, args.force)
    print("Done. Next: fill the {{...}} placeholders per SKILL.md workflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
