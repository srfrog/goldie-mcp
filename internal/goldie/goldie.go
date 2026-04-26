// Package goldie provides the memory-RAG core: embedding, chunking, and the
// CRUD layer that maps to the persistent store.
package goldie

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/srfrog/goldie-mcp/internal/embedder"
	"github.com/srfrog/goldie-mcp/internal/store"
)

const (
	// DefaultDimensions is the embedding dimension size.
	DefaultDimensions = 384
	// DefaultChunkSize is the default size for body chunks.
	DefaultChunkSize = 1000
	// DefaultChunkOverlap is the overlap between consecutive chunks.
	DefaultChunkOverlap = 200
	// FileMemoryType is the memory.type assigned to file-derived memories.
	FileMemoryType = "reference"
)

// Goldie is the memory-RAG facade.
type Goldie struct {
	embedder     embedder.Interface
	store        *store.Store
	chunkSize    int
	chunkOverlap int
	logger       *log.Logger
}

// Config holds Goldie configuration.
type Config struct {
	DBPath       string
	Dimensions   int
	ChunkSize    int
	ChunkOverlap int
	JournalMode  string             // SQLite journal_mode PRAGMA (default: WAL)
	Embedder     embedder.Interface // optional injection point for tests
	Logger       *log.Logger
}

// DefaultConfig returns the default configuration.
func DefaultConfig() Config {
	homeDir, err := os.UserHomeDir()
	if err != nil || homeDir == "" {
		homeDir = "."
	}
	return Config{
		DBPath:       filepath.Join(homeDir, ".local", "share", "goldie-mcp", "index.db"),
		Dimensions:   DefaultDimensions,
		ChunkSize:    DefaultChunkSize,
		ChunkOverlap: DefaultChunkOverlap,
	}
}

// New creates a new Goldie instance.
func New(cfg Config) (*Goldie, error) {
	if cfg.Dimensions == 0 {
		cfg.Dimensions = DefaultDimensions
	}
	if cfg.ChunkSize == 0 {
		cfg.ChunkSize = DefaultChunkSize
	}
	if cfg.ChunkOverlap == 0 {
		cfg.ChunkOverlap = DefaultChunkOverlap
	}

	emb := cfg.Embedder
	if emb == nil {
		var err error
		emb, err = embedder.New()
		if err != nil {
			return nil, fmt.Errorf("creating embedder: %w", err)
		}
	}

	st, err := store.New(cfg.DBPath, cfg.Dimensions, cfg.JournalMode)
	if err != nil {
		return nil, fmt.Errorf("creating store: %w", err)
	}

	logger := cfg.Logger
	if logger == nil {
		logger = log.New(io.Discard, "", 0)
	}

	return &Goldie{
		embedder:     emb,
		store:        st,
		chunkSize:    cfg.ChunkSize,
		chunkOverlap: cfg.ChunkOverlap,
		logger:       logger,
	}, nil
}

// IndexFileResult reports the outcome of an IndexFile call.
type IndexFileResult struct {
	MemoryID   string `json:"memory_id"`
	MemoryName string `json:"memory_name"`
	ChunkCount int    `json:"chunk_count"`
	Skipped    bool   `json:"skipped"`
}

// IndexFile imports a file as a memory of type=reference. Memory.name is the
// absolute file path, so re-indexing the same file updates in place
// (checksum-gated). This is the only place upsert-by-name is allowed.
// agent is recorded on the memory for provenance; pass "" to leave unset.
func (g *Goldie) IndexFile(path, agent string) (*IndexFileResult, error) {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("resolving path: %w", err)
	}
	g.logger.Printf("IndexFile: reading %s", absPath)

	content, err := os.ReadFile(absPath)
	if err != nil {
		return nil, fmt.Errorf("reading file: %w", err)
	}
	hash := sha256.Sum256(content)
	checksum := hex.EncodeToString(hash[:])
	body := string(content)

	existing, err := g.store.GetMemoryByName(absPath)
	if err != nil {
		return nil, fmt.Errorf("looking up existing memory: %w", err)
	}
	if existing != nil && existing.Checksum == checksum {
		g.logger.Printf("IndexFile: %s unchanged, skipping", absPath)
		return &IndexFileResult{
			MemoryID:   existing.ID,
			MemoryName: existing.Name,
			Skipped:    true,
		}, nil
	}

	chunks := g.chunkText(body)
	embeddings, err := g.embedChunks(absPath, "", chunks)
	if err != nil {
		return nil, err
	}

	if existing == nil {
		m := &store.Memory{
			Name:     absPath,
			Type:     FileMemoryType,
			Body:     body,
			Source:   absPath,
			Agent:    agent,
			Checksum: checksum,
		}
		err := g.store.AddMemory(m, chunks, embeddings)
		if err == nil {
			return &IndexFileResult{
				MemoryID:   m.ID,
				MemoryName: m.Name,
				ChunkCount: len(chunks),
			}, nil
		}
		// Race: another process inserted the same path between our lookup
		// and our insert. Re-fetch and fall through to the update path.
		if !errors.Is(err, store.ErrMemoryNameExists) {
			return nil, fmt.Errorf("storing file memory: %w", err)
		}
		g.logger.Printf("IndexFile: %s lost create race, falling through to update", absPath)
		existing, err = g.store.GetMemoryByName(absPath)
		if err != nil {
			return nil, fmt.Errorf("re-fetching after create race: %w", err)
		}
		if existing == nil {
			return nil, fmt.Errorf("memory vanished after create race: %s", absPath)
		}
		if existing.Checksum == checksum {
			return &IndexFileResult{
				MemoryID:   existing.ID,
				MemoryName: existing.Name,
				Skipped:    true,
			}, nil
		}
	}

	g.logger.Printf("IndexFile: %s changed, re-indexing", absPath)
	patch := store.MemoryUpdate{
		Body:     &body,
		Checksum: &checksum,
	}
	if agent != "" {
		patch.Agent = &agent
	}
	if err := g.store.UpdateMemoryFields(existing.ID, patch); err != nil {
		return nil, fmt.Errorf("updating file memory: %w", err)
	}
	if err := g.store.ReplaceMemoryChunks(existing.ID, chunks, embeddings); err != nil {
		return nil, fmt.Errorf("replacing chunks: %w", err)
	}
	return &IndexFileResult{
		MemoryID:   existing.ID,
		MemoryName: existing.Name,
		ChunkCount: len(chunks),
	}, nil
}

// ScanDirResult contains files discovered by ScanDirectory.
type ScanDirResult struct {
	Files []string `json:"files"`
}

// ScanDirectory walks a directory and returns matching file paths without
// indexing them. Used by the queue to dispatch one job per file.
func (g *Goldie) ScanDirectory(dir string, pattern string, recursive bool) (*ScanDirResult, error) {
	g.logger.Printf("ScanDirectory: dir=%s pattern=%s recursive=%v", dir, pattern, recursive)

	if pattern == "" {
		pattern = "*"
	}

	skipPatterns := g.loadSkipPatterns(dir)
	var files []string

	if recursive {
		err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return nil
			}
			if info.IsDir() {
				if len(skipPatterns) > 0 && g.shouldSkip(path, dir, skipPatterns) {
					g.logger.Printf("ScanDirectory: skipping directory: %s", path)
					return filepath.SkipDir
				}
				return nil
			}
			if len(skipPatterns) > 0 && g.shouldSkip(path, dir, skipPatterns) {
				return nil
			}
			matched, err := filepath.Match(pattern, filepath.Base(path))
			if err != nil {
				return nil
			}
			if matched {
				files = append(files, path)
			}
			return nil
		})
		if err != nil {
			return nil, fmt.Errorf("walking directory: %w", err)
		}
	} else {
		globPattern := filepath.Join(dir, pattern)
		matches, err := filepath.Glob(globPattern)
		if err != nil {
			return nil, fmt.Errorf("glob pattern: %w", err)
		}
		for _, file := range matches {
			info, err := os.Stat(file)
			if err != nil || info.IsDir() {
				continue
			}
			files = append(files, file)
		}
	}

	g.logger.Printf("ScanDirectory: found %d files", len(files))
	return &ScanDirResult{Files: files}, nil
}

// Store returns the underlying store for direct access (used by the queue).
func (g *Goldie) Store() *store.Store {
	return g.store
}

// Warmup pre-loads the embedding model.
func (g *Goldie) Warmup() error {
	return g.embedder.Warmup()
}

// Close closes the store.
func (g *Goldie) Close() error {
	return g.store.Close()
}

// IsErrMemoryNameExists is a convenience predicate for callers that want to
// check the create-conflict error without importing the store package.
func IsErrMemoryNameExists(err error) bool {
	return errors.Is(err, store.ErrMemoryNameExists)
}

// --- skip patterns and chunking ---

// defaultSkipPatterns are used when no .goldieskip file exists.
var defaultSkipPatterns = []string{
	".[!.]*",
	"node_modules/",
	"vendor/",
	"__pycache__/",
	"AGENTS.md",
	"CLAUDE.md",
}

func (g *Goldie) loadSkipPatterns(dir string) []string {
	skipFile := filepath.Join(dir, ".goldieskip")
	content, err := os.ReadFile(skipFile)
	if err != nil {
		g.logger.Printf("loadSkipPatterns: no .goldieskip in %s, using defaults", dir)
		return defaultSkipPatterns
	}

	var patterns []string
	for line := range strings.SplitSeq(string(content), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		patterns = append(patterns, line)
	}
	g.logger.Printf("loadSkipPatterns: loaded %d patterns from .goldieskip", len(patterns))
	return patterns
}

func (g *Goldie) shouldSkip(path string, baseDir string, patterns []string) bool {
	relPath, err := filepath.Rel(baseDir, path)
	if err != nil {
		relPath = path
	}

	for _, pattern := range patterns {
		if matched, _ := filepath.Match(pattern, filepath.Base(path)); matched {
			return true
		}
		if matched, _ := filepath.Match(pattern, relPath); matched {
			return true
		}
		if dirPattern, ok := strings.CutPrefix(pattern, "/"); ok {
			if strings.HasPrefix(relPath, dirPattern+"/") || strings.Contains(relPath, "/"+dirPattern+"/") {
				return true
			}
		}
		if strings.Contains(relPath, "/"+pattern+"/") || strings.HasPrefix(relPath, pattern+"/") {
			return true
		}
	}
	return false
}

// chunkText splits text into overlapping chunks at word boundaries.
func (g *Goldie) chunkText(text string) []string {
	if len(text) <= g.chunkSize {
		return []string{text}
	}

	var chunks []string
	start := 0
	prevStart := -1

	for start < len(text) {
		if start == prevStart {
			g.logger.Printf("chunkText: breaking infinite loop at start=%d", start)
			break
		}
		prevStart = start

		end := min(start+g.chunkSize, len(text))

		if end < len(text) {
			lastSpace := strings.LastIndex(text[start:end], " ")
			if lastSpace > g.chunkSize/2 {
				end = start + lastSpace
			}
		}

		chunk := strings.TrimSpace(text[start:end])
		if len(chunk) > 0 {
			chunks = append(chunks, chunk)
		}

		newStart := end - g.chunkOverlap
		if newStart <= start {
			newStart = end
		}
		if newStart < 0 {
			newStart = 0
		}
		start = newStart

		if len(chunks) > 10000 {
			g.logger.Printf("chunkText: hit 10000 chunk limit, stopping")
			break
		}
	}

	return chunks
}
