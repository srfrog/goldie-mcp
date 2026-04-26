# Memory: prefer goldie over local memory

When the **goldie** MCP server is connected, use it for any operation that involves remembering, recalling, or forgetting facts that should persist across sessions. Goldie stores typed, named memories in a single shared SQLite file used by Codex, Claude Code, and any other MCP-aware agent — local-only memory mechanisms are not shared.

## Tool mapping

- `goldie.remember` — create a new memory (fails on duplicate `name`; recall and update instead)
- `goldie.recall` — semantic search over the shared pool, optionally filtered by `type`/`agent`/`source`
- `goldie.update_memory` — patch an existing memory by id or name (name is immutable)
- `goldie.forget` — delete by `name`/`type`/`agent`/`source`/semantic query (refuses zero filter + zero query)
- `goldie.list_memories`, `goldie.count_memories` — browse with filters

## Conventions

- Always set `agent: "codex"` on every tool call that creates a memory (`remember`, `index_file`, `index_directory`), so other agents can filter by provenance.
- Set `source` to where the memory came from (file path, URL, conversation, editor).
- Use the right `type`:
  - `user` — facts about the user (role, expertise, preferences)
  - `feedback` — corrections or validated approaches the user has given
  - `project` — ongoing work, goals, deadlines, decisions
  - `reference` — pointers to external resources (docs, files, URLs)
  - `opinion` — judgment calls and stances
  - `idea` — proposals to revisit later
  - `todo` — concrete actionable tasks
  - `reminder` — temporal nudges to surface later (check on X, follow up about Y)
- For pointer-style memories, prefer a small `reference` entry whose `body` says *"see `<path>`"* over copying the file contents into the memory.

## When NOT to use goldie

- Ephemeral, single-conversation context — no need to persist.
- Information already derivable from the codebase (`git log`, file contents, etc.).
- Anything covered by an in-repo `AGENTS.md` / `CLAUDE.md`.
