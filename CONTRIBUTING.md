# Contributing to Goldie

Thank you for your interest in contributing to Goldie.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Go 1.22+
- CGO enabled
- Git LFS (for the MiniLM model)
- ONNX Runtime (for MiniLM backend) or Ollama (for Ollama backend)

### Setup

```bash
git lfs install
git clone https://github.com/srfrog/goldie-mcp
cd goldie-mcp
make build
```

### Running Tests

```bash
make test
```

### Linting

```bash
make lint
```

## How to Contribute

### Reporting Bugs

- Check existing issues first
- Use the bug report template
- Include reproduction steps and environment details

### Suggesting Features

- Open an issue using the feature request template
- Explain the use case and motivation

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests (`make test`)
5. Run linter (`make lint`)
6. Commit with sign-off (`git commit -s`)
7. Push to your fork
8. Open a Pull Request

### Commit Messages

Use clear, descriptive commit messages:

```
type: short description

Longer explanation if needed.

Signed-off-by: Your Name <your@email.com>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

### Developer Certificate of Origin

All commits must be signed off to certify you have the right to submit the code:

```bash
git commit -s -m "feat: add new feature"
```

This adds a `Signed-off-by` line to your commit, indicating you agree to the [DCO](https://developercertificate.org/).

## Code Style

- Follow standard Go conventions (`gofmt`, `go vet`)
- Keep functions focused and small
- Add tests for new functionality
- Document exported functions and types

## Project Structure

```
goldie-mcp/
├── main.go                 # MCP server and tool handlers
├── internal/
│   ├── embedder/           # Embedding backends
│   │   ├── minilm/         # MiniLM (ONNX Runtime)
│   │   └── ollama/         # Ollama API client
│   ├── goldie/             # RAG core logic
│   ├── store/              # SQLite vector storage
│   └── queue/              # Async job processing
```

## Questions?

Open an issue or start a discussion.
