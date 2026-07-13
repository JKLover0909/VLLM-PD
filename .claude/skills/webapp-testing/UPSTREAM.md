# Upstream provenance

- Repository: https://github.com/anthropics/skills
- Source path: `skills/webapp-testing`
- Commit: `9d2f1ae187231d8199c64b5b762e1bdf2244733d`
- Commit date: 2026-07-01
- License: Apache License 2.0 (`LICENSE.txt` in this directory)
- Installed: 2026-07-13

## Local policy

This directory is vendored and pinned. The helper starts local processes and
uses `shell=True`; only pass commands reviewed for this repository. Do not start
destructive services or install browser/dependency packages without approval.
For Meibook, prefer the existing safe fixture/smoke harness and use frontend
commands through `npm --prefix frontend ...`.
