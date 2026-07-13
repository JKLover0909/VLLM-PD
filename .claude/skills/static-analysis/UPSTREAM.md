# Upstream provenance

- Repository: https://github.com/trailofbits/skills
- Source path: `plugins/static-analysis`
- Commit: `cfe5d7b1619e47fb5b38b7e2561dad7e5f1e89af`
- Commit date: 2026-06-30
- Upstream plugin version: 1.2.2
- License: CC BY-SA 4.0 (`LICENSE.txt` in this directory)
- Installed: 2026-07-13

## Local modifications

A project-level `SKILL.md` coordinator was added so Claude Code can discover the
bundle directly and enforce Meibook safety constraints. Upstream CodeQL, Semgrep,
SARIF skills, references, workflows, scripts, agent prompts, and plugin manifest
are otherwise preserved.

Do not install tools/query packs, clone rulesets, invoke network scans, or run
cleanup commands from upstream examples without explicit approval. Prefer output
under `/tmp` and verify every finding against source.
