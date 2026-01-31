# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Goldie, please report it privately.

**Do not open a public issue for security vulnerabilities.**

### How to Report

Email: [8219721+srfrog@users.noreply.github.com](mailto:8219721+srfrog@users.noreply.github.com)

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Resolution timeline**: Depends on severity, typically 30-90 days

### Scope

This policy applies to:
- The Goldie MCP server (`goldie-mcp`)
- Official release binaries
- Code in this repository

Out of scope:
- Third-party dependencies (report to their maintainers)
- Ollama or ONNX Runtime issues (report upstream)

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| Latest  | Yes                |
| < 1.0   | Best effort        |

## Security Considerations

Goldie runs locally and:
- Stores data in a local SQLite database
- Does not transmit data externally (except Ollama API if configured)
- Has filesystem access for indexing files

Users should:
- Avoid indexing sensitive files (credentials, secrets)
- Use appropriate file permissions on the database
- Keep the binary and dependencies updated
