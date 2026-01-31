# Goldie 🐕

A Retrieval-Augmented Generation (RAG) MCP server written in Go that runs locally in your machine.

## Features

- **Multiple embedding backends**: Choose between MiniLM (local, via ONNX Runtime) or Ollama
- **Local embeddings**: Uses [all-MiniLM-L6-v2] model for high-quality semantic embeddings (384 dimensions)
- **Ollama support**: Use any Ollama embedding model (nomic-embed-text, mxbai-embed-large, etc.)
- **SQLite vector storage**: Persistent storage using sqlite-vec extension
- **Document chunking**: Automatically chunks large documents with overlap
- **Semantic search**: Find relevant documents using vector similarity
- **Directory indexing**: Batch index files with glob patterns, supports recursive search

## Requirements

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

Or add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "goldie": {
      "type": "stdio",
      "command": "/path/to/goldie-mcp",
      "env": {
        "GOLDIE_DB_PATH": "/home/user/.local/share/goldie/index.db",
        "ONNXRUNTIME_LIB_PATH": "/path/to/libonnxruntime.so"
      }
    }
  }
}
```

Note: `ONNXRUNTIME_LIB_PATH` is optional if the library is in a standard location.

### With Ollama

```bash
claude mcp add -s user -e GOLDIE_DB_PATH=~/.local/share/goldie/index.db goldie /path/to/goldie-mcp -- -b ollama
```

Or add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "goldie": {
      "type": "stdio",
      "command": "/path/to/goldie-mcp",
      "args": ["-b", "ollama"],
      "env": {
        "GOLDIE_DB_PATH": "/home/user/.local/share/goldie/index.db",
        "OLLAMA_HOST": "http://localhost:11434",
        "OLLAMA_EMBED_MODEL": "nomic-embed-text"
      }
    }
  }
}
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

## Available Tools

### index_content

Index text content for semantic search. Use this for web pages, API responses, notes, or any text that doesn't come from a local file. For local files, use `index_file` instead.

**Parameters:**
- `content` (required): The text content to index
- `metadata` (optional): JSON string with metadata (e.g., `{"source": "https://example.com", "title": "Page Title"}`)

### index_file

Index a file from the filesystem.

**Parameters:**
- `path` (required): Path to the file to index

### index_directory

Index all files matching a pattern in a directory.

**Parameters:**
- `directory` (required): The directory path to index
- `pattern` (optional): File pattern to match (e.g., `*.md`, `*.txt`). Default: `*`
- `recursive` (optional): Whether to search subdirectories. Default: `false`

### search_index

Search for documents using semantic similarity.

**Parameters:**
- `query` (required): Search query text
- `limit` (optional): Maximum results (default: 5)

### recall

Recall knowledge from indexed documents about a topic. Returns a consolidated summary with source attribution, designed for natural conversation flow.

**Parameters:**
- `topic` (required): The topic to recall information about
- `depth` (optional): How many sources to consult (default: 5, max: 20)

### delete_document

Delete a document from the index.

**Parameters:**
- `id` (required): Document ID

### count_documents

Get the total number of indexed documents.

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

Here are example prompts you can use with Claude Code or Claude Desktop:

### index_content

Use for content that doesn't come from local files:

**Web content:**
```
Index this content from the React docs: "useState is a Hook that lets you add state to function components..."
```

**API responses:**
```
Index this API documentation: "POST /api/users - Creates a new user. Required fields: email, password"
```

**Notes and knowledge:**
```
Index this note: "Team decided to use PostgreSQL for the main database, Redis for caching"
```

**With metadata:**
```
Index this with source metadata: "OAuth2 flow requires client_id and redirect_uri" from "https://docs.example.com/auth"
```

### index_file

```
Index the file ~/project/README.md
```

```
Index ~/docs/architecture.md
```

### index_directory

```
Index all markdown files in ~/docs
```

```
Index all *.txt files in ~/notes
```

```
Index all *.md files in ~/projects recursively
```

```
Index everything in ~/config with pattern *.json recursively
```

### search_index

```
Search for authentication implementation
```

```
Search for "database migrations" and show me 10 results
```

### recall

```
Recall what you know about authentication
```

```
What do you remember about the API design?
```

```
Summarize your knowledge about error handling
```

```
Find documents about error handling
```

```
What do I have indexed about Docker?
```

### delete_document

```
Delete document abc123
```

```
Remove document xyz789 from the index
```

### count_documents

```
How many documents are in the Goldie index?
```

```
Count all indexed documents
```

## Indexing Claude Code Conversations

Goldie can index your Claude Code conversation history, making it searchable with semantic search. This lets you find past solutions, code snippets, and discussions across all your sessions.

### Where Claude Code Stores Conversations

Claude Code stores conversation transcripts in:

```
~/.claude/projects/<project-hash>/
```

Each project directory contains markdown files with your conversation history.

### Index All Your Conversations

```
Index all *.md files in ~/.claude/projects recursively
```

### Search Your Past Conversations

```
Search for "how did I fix the authentication bug"
```

```
Find conversations about Docker configuration
```

```
What regex patterns have I used before?
```

### Use Cases

- **Find past solutions**: "How did I solve that memory leak?"
- **Retrieve code snippets**: "Find the SQL migration I wrote"
- **Track project history**: "What changes did I make to the API?"
- **Learn from patterns**: "Show me examples of error handling"

### Tips

- Index conversations periodically to keep your knowledge base current
- Use metadata to tag conversations by project or topic
- Exclude sensitive conversations containing credentials or secrets

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
│   ├── goldie/             # RAG core logic
│   ├── store/              # SQLite vector storage
│   └── queue/              # Async job processing
├── go.mod
└── Makefile
```

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

**Note:** Different embedding models produce different dimension vectors. Documents indexed with one backend/model cannot be searched using another with different dimensions. Use separate databases or re-index when switching.

## License

MIT

[all-MiniLM-L6-v2]: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
