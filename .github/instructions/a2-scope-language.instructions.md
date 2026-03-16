---
description: "Use when handling any request in this repository. Enforce English-only responses and restrict all code exploration, edits, and command execution to the A2 folder."
name: "A2 Scope And Language Rules"
applyTo: "A2/**"
---
# A2 Scope And Language Rules

- Respond only in English, even if the user prompt is in Hindi or any other language.
- Limit all file reads, edits, searches, and command execution to `A2/`.
- Do not inspect, modify, or reference files outside `A2/` unless the user explicitly overrides this rule.
- When a request would require files outside `A2/`, explain the constraint and ask for confirmation before proceeding.
