# CLAUDE.md

This file defines the default operating rules for Claude Code in this repository. Follow these rules on every task unless the user explicitly overrides them.

## Core Priorities

1. Be correct.
2. Use as few tokens as possible.
3. Minimize unnecessary tool calls, file reads, and repeated explanations.
4. Respect existing work. Never revert or overwrite unrelated user changes.
5. Stay aligned with the current repository patterns and active migration path.

## Repository Context

- This is a real-time power quality monitoring repository.
- The active MCU path is `firmware/teensy/`.
- ESP32-P4 code exists under `firmware/esp32p4/` and is set aside for now.
- Host/runtime code is primarily Python-based.
- Prefer preserving the existing acquisition and feature contract unless the user asks to change it.

## Navigation Rules

1. Prefer Graphify nodes and graph-based code navigation first whenever available.
2. Use Graphify to understand symbol relationships, call paths, ownership, and impact before broader searching.
3. Use `rg` or `grep` only when truly required:
   - simple exact-text lookup
   - confirming a filename or literal string
   - fallback when Graphify cannot answer the question
4. Do not start with broad repo-wide text searches if Graphify can answer the question faster and with less noise.
5. Read only the smallest set of files needed to complete the task.

## Token Discipline

1. Keep responses short, direct, and high-signal.
2. Do not restate the prompt, repo context, or obvious observations unless needed.
3. Avoid long planning unless the task is complex or the user asks for it.
4. Avoid dumping large code blocks or long command output unless the user requests them.
5. Summarize findings instead of narrating every exploration step.
6. Prefer targeted edits over speculative refactors.
7. Do not read large files fully when a focused section is enough.
8. Do not use multiple tools when one precise tool call will do.

## Working Style

1. Inspect before editing, but keep inspection minimal.
2. Make reasonable assumptions and continue unless the risk is meaningful.
3. Ask questions only when the choice materially affects behavior, architecture, safety, or user intent.
4. Prefer concrete progress over discussion-heavy back-and-forth.
5. Preserve the project’s current structure, naming, and implementation style.

## Editing Rules

1. Make the smallest correct change.
2. Do not introduce unrelated cleanup.
3. Do not create new abstractions unless they clearly reduce complexity.
4. Keep comments sparse and useful.
5. Preserve ASCII unless the file already requires non-ASCII.
6. Avoid duplicate logic and avoid expanding file scope without a clear reason.

## Search and Read Strategy

Use this order by default:

1. Graphify nodes
2. Directly relevant file
3. Focused symbol-level read
4. `rg` fallback only if needed

If the answer is already clear, stop searching.

## Command and Tool Usage

1. Use tools only when needed.
2. Prefer precise reads over exploratory scanning.
3. Avoid repeated reads of the same file unless new context requires it.
4. Avoid expensive commands unless they directly support the task.
5. Run tests or verification proportional to the change.

## Code Quality Expectations

1. Keep behavior stable unless the user asked for a change.
2. For firmware changes, be careful with timing, memory, concurrency, and hardware-facing behavior.
3. For Python/runtime changes, preserve CLI contracts, config compatibility, and frame/protocol assumptions unless intentionally changing them.
4. Match existing patterns before introducing new ones.

## Final Response Rules

1. Be concise.
2. State what changed.
3. Mention verification performed, if any.
4. Mention blockers or unverified areas plainly.
5. Do not turn the final response into a long changelog.

## Override Rule

If the user gives direct instructions that conflict with this file, follow the user.
