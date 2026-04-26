# Goldie 🐕

A multi-agent memory MCP server written in Go that runs locally on your machine. Goldie stores typed, named **memories** in a shared SQLite vector index so multiple agents (Claude, Codex, etc.) can `remember`, `recall`, `update_memory`, and `forget` from one pool — replacing per-project `MEMORY.md` files with a backend they can all share.

## Features

- **Typed memories**: Each memory has a type (`user`, `feedback`, `project`, `reference`, `opinion`, `idea`, `todo`, `reminder`), a unique name, optional description, body, agent, and source
- **Shared pool**: Scope is a SQLite file — point any number of agents at the same DB and they share memory
- **Semantic recall**: Filtered KNN over chunk embeddings; recall returns the parent memory plus the matched excerpt
- **Multiple embedding backends**: MiniLM (local via ONNX Runtime) or Ollama (any embedding model)
- **File ingestion**: `index_file` / `index_directory` import files as `reference` memories named by absolute path (checksum-gated upsert)
- **Async job queue**: Long-running indexing operations run in the background with progress tracking

## Requirements

### MiniLM Backend (requires ONNX Runtime)

```bash
# macOS
brew install onnxruntime

# Ubuntu/Debian
sudo apt install libonnxruntime-dev

# Fedora/RHEL
sudo dnf install onnxruntime-devel

# Arch Linux
sudo pacman -S onnxruntime
```

### Ollama Backend

If you want to use Ollama instead of MiniLM, you only need Ollama installed:

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull an embedding model
ollama pull nomic-embed-text
```

Skip the ONNX Runtime installation below if you only plan to use Ollama.

## Installation

### From Releases (recommended)

Download a pre-built binary from the [releases page](https://github.com/srfrog/goldie-mcp/releases):

| Platform | Binary |
|----------|--------|
| macOS (Apple Silicon) | `goldie-mcp-darwin-arm64` |
| macOS (Intel) | `goldie-mcp-darwin-amd64` |
| Linux (x86_64) | `goldie-mcp-linux-amd64` |
| Linux (ARM64) | `goldie-mcp-linux-arm64` |

```bash
# Example for macOS Apple Silicon
curl -LO https://github.com/srfrog/goldie-mcp/releases/latest/download/goldie-mcp-darwin-arm64
chmod +x goldie-mcp-darwin-arm64
mv goldie-mcp-darwin-arm64 ~/bin/goldie-mcp
```

The release binaries are ad-hoc codesigned for macOS and include the MiniLM model, so no additional downloads are required.

### Build from Source

Requires Go 1.22+, CGO enabled, and Git LFS (the model file is stored with LFS):

```bash
git lfs install  # if not already configured
git clone https://github.com/srfrog/goldie-mcp
cd goldie-mcp
make build
```

## Configuration

### Command Line Flags

| Flag | Description | Default |
|------|-------------|---------|
| `-b` | Embedding backend: `minilm` or `ollama` | `minilm` |
| `-l` | Log file path | stderr |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOLDIE_DB_PATH` | Path to SQLite database | `~/.local/share/goldie/index.db` |
| `GOLDIE_JOURNAL_MODE` | SQLite journal_mode PRAGMA. Default is safe for cloud-synced storage. Set `WAL` for local-only DBs to enable read-during-write concurrency | `DELETE` |
| `ONNXRUNTIME_LIB_PATH` | Path to libonnxruntime shared library (MiniLM only) | Auto-detected |
| `OLLAMA_HOST` | Ollama API base URL (Ollama only) | `http://localhost:11434` |
| `OLLAMA_EMBED_MODEL` | Ollama embedding model name (Ollama only) | `nomic-embed-text` |
| `OLLAMA_EMBED_DIMENSIONS` | Custom model dimensions (Ollama only) | Auto-detected for known models |

### Supported Ollama Embedding Models

| Model | Dimensions | Notes |
|-------|------------|-------|
| `nomic-embed-text` | 768 | Default, good general purpose |
| `mxbai-embed-large` | 1024 | Higher quality, slower |
| `all-minilm` | 384 | Same as MiniLM backend |

For other models, set `OLLAMA_EMBED_DIMENSIONS` to the model's output dimensions.

## Usage with Claude Code

### With MiniLM (default)

```bash
claude mcp add -s user -e GOLDIE_DB_PATH=~/.local/share/goldie/index.db goldie /path/to/goldie-mcp
```

Note: `ONNXRUNTIME_LIB_PATH` is optional if the library is in a standard location.

### With Ollama

```bash
claude mcp add -s user -e GOLDIE_DB_PATH=~/.local/share/goldie/index.db goldie /path/to/goldie-mcp -- -b ollama
```

Note: Make sure Ollama is running (`ollama serve`) before starting Claude Code.

## Usage with Claude Desktop

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

### With MiniLM (default)

```json
{
  "mcpServers": {
    "goldie": {
      "type": "stdio",
      "command": "/path/to/goldie-mcp",
      "env": {
        "GOLDIE_DB_PATH": "/home/user/.local/share/goldie/index.db"
      }
    }
  }
}
```

### With Ollama

```json
{
  "mcpServers": {
    "goldie": {
      "type": "stdio",
      "command": "/path/to/goldie-mcp",
      "args": ["-b", "ollama"],
      "env": {
        "GOLDIE_DB_PATH": "/home/user/.local/share/goldie/index.db",
        "OLLAMA_EMBED_MODEL": "nomic-embed-text"
      }
    }
  }
}
```

## Usage with OpenAI Codex

Add to your Codex configuration (`~/.codex/config.toml`):

### With MiniLM (default)

```toml
[mcp_servers.goldie]
command = "/path/to/goldie-mcp"

[mcp_servers.goldie.env]
GOLDIE_DB_PATH = "/home/user/.local/share/goldie/index.db"
ONNXRUNTIME_LIB_PATH = "/path/to/libonnxruntime.so"
```

Note: `ONNXRUNTIME_LIB_PATH` is optional if the library is in a standard location. Homebrew will install it to `/opt/homebrew/lib/libonnxruntime.dylib` on macOS. In Linux, find it with `ldconfig -p | grep onnxruntime`.

### With Ollama

```toml
[mcp_servers.goldie]
command = "/path/to/goldie-mcp"
args = ["-b", "ollama"]

[mcp_servers.goldie.env]
GOLDIE_DB_PATH = "/home/user/.local/share/goldie/index.db"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
```

## Memory Model

Goldie's index is a flat pool of **memories**. Each memory has these fields:

| Field         | Required | Notes                                                                 |
|---------------|----------|-----------------------------------------------------------------------|
| `name`        | yes      | Unique within the database. Names collide → `remember` fails (see below) |
| `type`        | yes      | One of: `user`, `feedback`, `project`, `reference`, `opinion`, `idea`, `todo`, `reminder` |
| `body`        | yes      | The full content. Chunked under the hood for embedding-level recall   |
| `description` | no       | One-line summary; participates in semantic recall                     |
| `agent`       | no       | The agent that created the memory (e.g. `claude-opus-4-7`, `codex`)   |
| `source`      | no       | Where the memory came from (file path, editor, URL)                   |

**Sharing.** "Scope" is just the SQLite file. Multiple agents pointed at the same `GOLDIE_DB_PATH` share the same pool of memories — there is no per-agent isolation. Use `agent` and `source` to filter on read/delete.

**Naming and conflicts.** Names must be unique. `remember` is strict — no upsert. If two agents try to create the same name, the second one gets an error and is expected to `recall` the existing memory and call `update_memory` (or pick a different name).

**Update semantics.** Name is immutable. `update_memory` accepts patches for type/description/body/source/agent; changes to `description` or `body` re-embed the chunks.

**File ingestion.** `index_file` / `index_directory` are the *one* exception to the no-upsert rule. They import files as memories of `type=reference`, with `name = source = <absolute path>`. Re-indexing the same path skips when the SHA-256 checksum matches and replaces the body when it doesn't.

## Available Tools

### remember

Create a new memory. Fails if `name` is already in use — recall it and use `update_memory` instead.

**Parameters:**
- `name` (required): Unique identifier (e.g., `feedback_testing`)
- `type` (required): One of `user`, `feedback`, `project`, `reference`, `opinion`, `idea`, `todo`, `reminder`
- `body` (required): Full content
- `description` (optional): One-line summary
- `agent` (optional): Agent that created the memory
- `source` (optional): Where the memory was generated

### recall

Semantic recall over memories. Returns the most relevant memories plus the matched chunk excerpt. Filter by type, agent, or source to narrow scope.

**Parameters:**
- `query` (required): Topic or question
- `limit` (optional): Max results (default 5, max 20)
- `type`, `agent`, `source` (optional): Filters

### update_memory

Update an existing memory by id or name. Body/description changes re-embed.

**Parameters:**
- `id_or_name` (required)
- `type`, `description`, `body`, `source`, `agent` (optional patches)

### forget

Delete memories. Requires at least one filter or a query — refuses to wipe everything. With a query, top matches within the (optional) filter are deleted.

**Parameters:**
- `name`, `type`, `agent`, `source` (optional filters)
- `query` (optional): semantic match
- `limit` (optional): max matches when query is given (default 5)

### list_memories

List memories matching the filter, newest first. Returns metadata only (no body).

**Parameters:**
- `type`, `agent`, `source` (optional filters)
- `limit` (optional)

### count_memories

Count memories matching the filter.

### index_file

Import a file as a `reference` memory. The memory's `name` is the absolute path; re-indexing updates in place when the checksum changes.

**Parameters:**
- `path` (required)

### index_directory

Import every matching file in a directory as `reference` memories.

**Parameters:**
- `directory` (required)
- `pattern` (optional, default `*`)
- `recursive` (optional, default `false`)

### job_status, list_jobs, clear_queue

Manage the async indexing queue. `index_file` and `index_directory` enqueue jobs that complete in the background; use `job_status` to check progress.

## Skip Patterns

When indexing directories, Goldie automatically skips certain files and directories to avoid indexing irrelevant content.

### Default Skip Patterns

If no `.goldieskip` file exists in the directory being indexed, Goldie uses these defaults:

| Pattern | Description |
|---------|-------------|
| `.[!.]*` | All dotfiles and dotdirs (`.git/`, `.env`, `.vscode/`, etc.) |
| `node_modules/` | Node.js dependencies |
| `vendor/` | Go/PHP vendor directories |
| `__pycache__/` | Python bytecode cache |
| `AGENTS.md` | AI agent configuration |
| `CLAUDE.md` | Claude configuration |

### Custom Skip Patterns

Create a `.goldieskip` file in the directory to define custom patterns. This **replaces** the defaults entirely. Same format as `.gitignore`, with the same pattern syntax.

```
# .goldieskip example
# Lines starting with # are comments

# Skip all dotfiles/dotdirs
.[!.]*

# Skip dependencies
node_modules/
vendor/
.venv/

# Skip build outputs
dist/
build/
target/

# Skip specific files
*.log
*.tmp
secrets.json
```

**Pattern syntax:**
- `*` matches any sequence of characters
- `?` matches any single character
- `[abc]` matches any character in the set
- `[!abc]` matches any character NOT in the set
- Patterns ending in `/` match directories

## Example Prompts

Example prompts you can use with Claude Code, Claude Desktop, or Codex:

### remember

```
Remember as a feedback memory named "feedback_testing": don't mock the database in integration tests — we got burned last quarter by mock/prod divergence.
```

```
Remember this user fact named "user_role": senior Go engineer, ten years of experience, currently learning React.
```

```
Save an opinion named "ui_dark_mode": dark mode is easier on the eyes for long sessions.
```

### recall

```
Recall what you know about database testing
```

```
Recall feedback memories about pull request size
```

```
What memories do I have about the API design?
```

### update_memory

```
Update memory "feedback_testing": new body is "use a real Postgres in CI; the staging DB is reset nightly".
```

### forget

```
Forget all opinion memories from agent "claude-opus-4-7"
```

```
Forget memories matching "old API design notes"
```

### list_memories / count_memories

```
List my feedback memories
```

```
How many memories has agent "codex" written?
```

### index_file / index_directory

```
Index the file ~/project/README.md
```

```
Index all *.md files in ~/docs recursively
```

## Agent Configuration

Agents won't reach for goldie by default — Claude Code has its own `/memory`, and Codex has its own context handling. The repo ships two opinionated templates that nudge them toward the shared pool. Both are short and safe to drop in as-is.

### Claude Code

Copy [`templates/CLAUDE.md`](templates/CLAUDE.md) to `~/.claude/CLAUDE.md` (it's loaded into every Claude Code session):

```bash
cp templates/CLAUDE.md ~/.claude/CLAUDE.md
```

For project-scoped behavior instead, copy it to `<project>/CLAUDE.md`.

### Codex (CLI / App)

Copy [`templates/AGENTS.md`](templates/AGENTS.md) to `~/.codex/AGENTS.md` (Codex loads `AGENTS.override.md` and `AGENTS.md` from `$CODEX_HOME`, default `~/.codex`):

```bash
cp templates/AGENTS.md ~/.codex/AGENTS.md
```

As a belt-and-suspenders measure, append the following to `~/.codex/config.toml` so the rule fires at session start even if Codex misses the AGENTS.md load:

```toml
developer_instructions = """
At session start, read and obey ~/.codex/AGENTS.md when it exists.
For persistent memory operations, prefer Goldie over local memory when the Goldie MCP server is connected.
"""
```

### Customizing

Both templates are starting points. Edit them to:
- Restrict which memory `type`s the agent should create
- Add project-specific naming conventions for `name` (e.g. `<area>_<topic>`)
- Override the default `agent` value
- Tighten or relax the "when NOT to use goldie" rules

## Sharing One Database Across Machines

Goldie's "scope" is the SQLite file, so syncing the database file across machines (iCloud, Dropbox, Syncthing) gives you follow-me memory without running a server. The default journal mode (`DELETE`) is already safe to use under cloud sync — only one `.db` file exists, no WAL/SHM sidecars to get out of order. One caveat:

**Don't write from two machines at once.** Cloud sync is not a coordination layer. If two machines write while disconnected, the sync client picks a winner and the other side's writes are lost (or a conflict copy is created). Workflow: quit any goldie session before switching machines, let sync settle, start the new machine.

For real multi-writer multi-machine setups, run goldie on a server and connect via Tailscale, or use [Litestream](https://litestream.io) to stream WAL changes to S3/B2.

### Opting into WAL for local-only DBs

If your DB lives on local disk (no cloud sync) and you want read-during-write concurrency under heavy multi-agent load, set `GOLDIE_JOURNAL_MODE=WAL`. The performance difference is negligible for typical memory-store workloads, but the option exists.

## Replacing Per-Project MEMORY.md

Goldie is designed to replace the per-project `MEMORY.md` files that agents like Claude Code create on disk. Point every agent at the same `GOLDIE_DB_PATH`, instruct them to use `remember` / `recall` / `update_memory` / `forget` instead of file-based memory, and you get:

- One pool shared by Claude (any session, any project), Codex, and any other MCP-aware agent
- Semantic recall instead of file globbing
- Per-agent provenance via the `agent` field, queryable through `recall`/`forget` filters
- Filtered cleanup (e.g. "forget all `feedback` memories from agent X")

### Indexing existing transcripts

You can also bulk-import old Claude Code conversation transcripts as `reference` memories so they participate in recall:

```
Index all *.md files in ~/.claude/projects recursively
```

Then ask:

```
Recall what I know about authentication bugs
```

```
What memories do I have about Docker?
```

## Troubleshooting

### macOS: Binary killed immediately (Signal 9)

When copying binaries on macOS, Gatekeeper may add quarantine attributes (`com.apple.provenance`) that cause the binary to be killed on launch. Use `make install DEST=<path>` which builds directly to the destination and codesigns the binary to avoid this issue.

### Codex is not recalling

- MCP support in Codex is experimental and it needs a bit more coaxing to work propertly. After indexing content, try to `recall <topic>` and check that it uses the `goldie.recall()` function, that indicates it's using the MCP backend.
- Codex sometimes doesn't trust the content from recall, and won't add it to the context, requiring redundant calls to `goldie.search_index()`. You can try with `recall <topic> and consolidate` to push the update.

## Architecture

```
goldie-mcp/
├── main.go                 # MCP server setup and tool handlers
├── internal/
│   ├── embedder/           # Embedding interface and backends
│   │   ├── minilm/         # MiniLM backend (ONNX Runtime)
│   │   └── ollama/         # Ollama backend (API client)
│   ├── goldie/             # Memory operations (Remember/Recall/Update/Forget)
│   │   ├── goldie.go       # Core, file ingestion, chunking
│   │   └── memory.go       # Type whitelist + memory CRUD
│   ├── store/              # SQLite memory + chunk + vec storage
│   │   ├── store.go        # Connection, jobs
│   │   └── memory.go       # Memory schema and queries
│   └── queue/              # Async job processing
├── go.mod
└── Makefile
```

### Schema

Three SQLite tables make up the memory index:

- `memories` — one row per memory: `id, name UNIQUE, type, description, body, agent, source, checksum, created_at, updated_at`
- `memory_chunks` — body split into overlapping chunks for embedding granularity: `id, memory_id, chunk_index, content`
- `memories_vec` — `vec0` virtual table over chunk embeddings, joined back to memories on recall

Recall does KNN over chunks, then dedupes to distinct memories, returning the best-matching excerpt for each.

## Embedding Backends

### MiniLM Backend (`-b minilm`)

Uses [all-MiniLM-L6-v2] via ONNX Runtime:
- 384-dimensional embeddings
- Optimized for semantic similarity
- Runs locally, model embedded in binary

### Ollama Backend (`-b ollama`)

Uses Ollama's embedding API with your choice of model:
- `nomic-embed-text` (768 dimensions) - Default, good balance of quality and speed
- `mxbai-embed-large` (1024 dimensions) - Higher quality embeddings
- `all-minilm` (384 dimensions) - Same model as MiniLM backend
- Any other Ollama embedding model (set `OLLAMA_EMBED_DIMENSIONS`)

**Note:** Different embedding models produce different dimension vectors. Memories indexed with one backend/model cannot be recalled using another with different dimensions. Use separate databases or re-index when switching.

## License

MIT

[all-MiniLM-L6-v2]: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
