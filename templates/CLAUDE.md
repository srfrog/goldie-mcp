# Memory: prefer goldie over `/memory`

When the **goldie** MCP server is connected, prefer its tools over the built-in file-based `/memory` system:

- `goldie.remember` instead of writing memory files
- `goldie.recall` instead of reading `MEMORY.md`
- `goldie.update_memory` to revise an existing memory (names are unique; if `remember` fails with a duplicate-name error, recall the existing one and update it)
- `goldie.forget` to delete by name, type, agent, source, or semantic query
- `goldie.list_memories` / `goldie.count_memories` to browse

Goldie persists across **sessions, projects, and agents** (Claude, Codex, etc.) via a single shared SQLite file. The local `/memory` system is per-project on disk and not shared. Only fall back to `/memory` if goldie is unavailable in the current session.

When creating memories, always set `agent` (e.g. `claude-opus-4-7`) and `source` (e.g. file path, conversation, editor) so future sessions can filter by provenance. This applies to all creation tools: `remember`, `index_file`, and `index_directory`.
