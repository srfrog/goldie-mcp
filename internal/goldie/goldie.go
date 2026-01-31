// Package goldie
package goldie

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"log"
	"maps"
	"os"
	"path/filepath"
	"strings"

	"github.com/google/uuid"

	"github.com/srfrog/goldie-mcp/internal/embedder"
	"github.com/srfrog/goldie-mcp/internal/store"
)

const (
	// DefaultDimensions is the embedding dimension size
	DefaultDimensions = 384
	// DefaultChunkSize is the default size for document chunks
	DefaultChunkSize = 1000
	// DefaultChunkOverlap is the overlap between chunks
	DefaultChunkOverlap = 200
)

// Goldie provides retrieval-augmented generation functionality
type Goldie struct {
	embedder     embedder.Interface
	store        *store.Store
	chunkSize    int
	chunkOverlap int
	logger       *log.Logger
}

// Config holds RAG configuration
type Config struct {
	DBPath       string
	Dimensions   int
	ChunkSize    int
	ChunkOverlap int
	Embedder     embedder.Interface // Optional: inject custom embedder (for testing)
	Logger       *log.Logger        // Optional: logger for debug output
}

// DefaultConfig returns default configuration
func DefaultConfig() Config {
	homeDir, err := os.UserHomeDir()
	if err != nil || homeDir == "" {
		// Fallback to current directory if home dir not available
		homeDir = "."
	}
	return Config{
		DBPath:       filepath.Join(homeDir, ".local", "share", "goldie-mcp", "index.db"),
		Dimensions:   DefaultDimensions,
		ChunkSize:    DefaultChunkSize,
		ChunkOverlap: DefaultChunkOverlap,
	}
}

// New creates a new Goldie instance
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

	// Use injected embedder or create default ONNX embedder
	var emb embedder.Interface
	if cfg.Embedder != nil {
		emb = cfg.Embedder
	} else {
		var err error
		emb, err = embedder.New()
		if err != nil {
			return nil, fmt.Errorf("creating embedder: %w", err)
		}
	}

	st, err := store.New(cfg.DBPath, cfg.Dimensions)
	if err != nil {
		return nil, fmt.Errorf("creating store: %w", err)
	}

	// Use provided logger or create a discard logger
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

// IndexResult contains information about an indexed document
type IndexResult struct {
	ID         string `json:"id"`
	ChunkCount int    `json:"chunk_count"`
}

// Index indexes a document, optionally chunking large content
func (r *Goldie) Index(content string, metadata map[string]string, id string) (*IndexResult, error) {
	r.logger.Printf("Index: starting, id=%s, content_len=%d, chunkSize=%d", id, len(content), r.chunkSize)

	if content == "" {
		return nil, fmt.Errorf("empty content")
	}
	r.logger.Printf("Index: content not empty")

	if id == "" {
		id = uuid.New().String()
	}
	r.logger.Printf("Index: id=%s, checking size", id)

	// For small documents, index directly
	if len(content) <= r.chunkSize {
		r.logger.Printf("Index: small document, generating single embedding")
		embedding, err := r.embedder.Embed(content)
		if err != nil {
			return nil, fmt.Errorf("generating embedding: %w", err)
		}
		r.logger.Printf("Index: embedding generated, storing document")

		if err := r.store.AddDocument(id, content, metadata, embedding); err != nil {
			return nil, fmt.Errorf("storing document: %w", err)
		}
		r.logger.Printf("Index: document stored successfully")

		return &IndexResult{ID: id, ChunkCount: 1}, nil
	}

	// Chunk large documents
	r.logger.Printf("Index: large document, calling chunkText...")
	chunks := r.chunkText(content)
	r.logger.Printf("Index: split into %d chunks", len(chunks))

	for i, chunk := range chunks {
		chunkID := fmt.Sprintf("%s_chunk_%d", id, i)

		chunkMeta := make(map[string]string)
		maps.Copy(chunkMeta, metadata)
		chunkMeta["parent_id"] = id
		chunkMeta["chunk_index"] = fmt.Sprintf("%d", i)
		chunkMeta["total_chunks"] = fmt.Sprintf("%d", len(chunks))

		// r.logger.Printf("Index: generating embedding for chunk %d/%d (len=%d)", i+1, len(chunks), len(chunk))
		embedding, err := r.embedder.Embed(chunk)
		if err != nil {
			return nil, fmt.Errorf("generating embedding for chunk %d: %w", i, err)
		}
		// r.logger.Printf("Index: storing chunk %d/%d", i+1, len(chunks))

		if err := r.store.AddDocument(chunkID, chunk, chunkMeta, embedding); err != nil {
			return nil, fmt.Errorf("storing chunk %d: %w", i, err)
		}
	}

	return &IndexResult{ID: id, ChunkCount: len(chunks)}, nil
}

// IndexFile indexes a file from the filesystem
func (r *Goldie) IndexFile(path string) (*IndexResult, error) {
	r.logger.Printf("IndexFile: reading file %s", path)

	// Use filename as base ID
	id := filepath.Base(path)

	// Read file content first to compute checksum
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading file: %w", err)
	}
	r.logger.Printf("IndexFile: read %d bytes from %s", len(content), path)

	// Compute checksum
	hash := sha256.Sum256(content)
	checksum := hex.EncodeToString(hash[:])

	// Check if already indexed with same checksum (check both direct ID and chunked ID)
	existing, _ := r.store.GetDocument(id)
	if existing != nil && existing.Metadata != nil && existing.Metadata["checksum"] == checksum {
		r.logger.Printf("IndexFile: %s unchanged (checksum match), skipping", path)
		return &IndexResult{ID: id, ChunkCount: 0}, nil
	}
	existingChunk, _ := r.store.GetDocument(id + "_chunk_0")
	if existingChunk != nil && existingChunk.Metadata != nil && existingChunk.Metadata["checksum"] == checksum {
		r.logger.Printf("IndexFile: %s unchanged (chunked, checksum match), skipping", path)
		return &IndexResult{ID: id, ChunkCount: 0}, nil
	}

	// If document exists but checksum differs, delete old version first
	if existing != nil || existingChunk != nil {
		r.logger.Printf("IndexFile: %s changed, re-indexing", path)
		r.DeleteDocumentAndChunks(id)
	}

	metadata := map[string]string{
		"source":   path,
		"filename": filepath.Base(path),
		"checksum": checksum,
	}

	r.logger.Printf("IndexFile: calling Index with id=%s, content_len=%d", id, len(content))
	return r.Index(string(content), metadata, id)
}

// DeleteDocumentAndChunks removes a document and all its chunks, returns count of deleted
func (r *Goldie) DeleteDocumentAndChunks(id string) int {
	deleted := 0

	// Delete main document
	if doc, _ := r.store.GetDocument(id); doc != nil {
		r.store.DeleteDocument(id)
		deleted++
	}

	// Delete chunks (up to 10000)
	for i := range 10000 {
		chunkID := fmt.Sprintf("%s_chunk_%d", id, i)
		doc, _ := r.store.GetDocument(chunkID)
		if doc == nil {
			break
		}
		r.store.DeleteDocument(chunkID)
		deleted++
	}

	return deleted
}

// IndexDirResult contains results from indexing a directory
type IndexDirResult struct {
	IndexedFiles []string `json:"indexed_files"`
	SkippedFiles []string `json:"skipped_files,omitempty"`
	FailedFiles  []string `json:"failed_files,omitempty"`
	TotalChunks  int      `json:"total_chunks"`
}

// defaultSkipPatterns are used when no .goldieskip file exists
var defaultSkipPatterns = []string{
	".[!.]*", // All dotfiles/dotdirs except "." and ".."
	"node_modules/",
	"vendor/",
	"__pycache__/",
	"AGENTS.md",
	"CLAUDE.md",
}

// loadSkipPatterns loads patterns from .goldieskip file, or returns defaults if not found
func (r *Goldie) loadSkipPatterns(dir string) []string {
	skipFile := filepath.Join(dir, ".goldieskip")
	content, err := os.ReadFile(skipFile)
	if err != nil {
		// No skip file, use defaults
		r.logger.Printf("IndexDirectory: no .goldieskip found, using %d default skip patterns", len(defaultSkipPatterns))
		return defaultSkipPatterns
	}

	var patterns []string
	for line := range strings.SplitSeq(string(content), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue // Skip empty lines and comments
		}
		patterns = append(patterns, line)
	}
	r.logger.Printf("IndexDirectory: loaded %d skip patterns from .goldieskip", len(patterns))
	return patterns
}

// shouldSkip checks if a path matches any skip pattern
func (r *Goldie) shouldSkip(path string, baseDir string, patterns []string) bool {
	relPath, err := filepath.Rel(baseDir, path)
	if err != nil {
		relPath = path
	}

	for _, pattern := range patterns {
		// Check against filename
		if matched, _ := filepath.Match(pattern, filepath.Base(path)); matched {
			return true
		}
		// Check against relative path
		if matched, _ := filepath.Match(pattern, relPath); matched {
			return true
		}
		// Check if pattern is a directory prefix
		if dirPattern, ok := strings.CutPrefix(pattern, "/"); ok {
			if strings.HasPrefix(relPath, dirPattern+"/") || strings.Contains(relPath, "/"+dirPattern+"/") {
				return true
			}
		}
		// Check if path contains the pattern as a directory component
		if strings.Contains(relPath, "/"+pattern+"/") || strings.HasPrefix(relPath, pattern+"/") {
			return true
		}
	}
	return false
}

// IndexDirectory indexes all files matching a pattern in a directory
func (r *Goldie) IndexDirectory(dir string, pattern string, recursive bool) (*IndexDirResult, error) {
	r.logger.Printf("IndexDirectory: dir=%s pattern=%s recursive=%v", dir, pattern, recursive)

	if pattern == "" {
		pattern = "*"
	}

	// Load skip patterns
	skipPatterns := r.loadSkipPatterns(dir)

	var files []string

	if recursive {
		r.logger.Printf("IndexDirectory: walking directory recursively...")
		err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return nil // Skip files we can't access
			}
			if info.IsDir() {
				// Skip directories matching skip patterns
				if len(skipPatterns) > 0 && r.shouldSkip(path, dir, skipPatterns) {
					r.logger.Printf("IndexDirectory: skipping directory: %s", path)
					return filepath.SkipDir
				}
				return nil
			}
			// Skip files matching skip patterns
			if len(skipPatterns) > 0 && r.shouldSkip(path, dir, skipPatterns) {
				return nil
			}
			matched, err := filepath.Match(pattern, filepath.Base(path))
			if err != nil {
				return nil // Invalid pattern, skip silently
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
		var err error
		files, err = filepath.Glob(globPattern)
		if err != nil {
			return nil, fmt.Errorf("glob pattern: %w", err)
		}
	}

	r.logger.Printf("IndexDirectory: found %d files to index", len(files))
	result := &IndexDirResult{}

	for i, file := range files {
		info, err := os.Stat(file)
		if err != nil || info.IsDir() {
			continue
		}
		r.logger.Printf("IndexDirectory: indexing file %d/%d: %s", i+1, len(files), file)

		indexResult, err := r.IndexFile(file)
		if err != nil {
			result.FailedFiles = append(result.FailedFiles, file)
			continue
		}

		if indexResult.ChunkCount == 0 {
			// File was already indexed, skipped
			result.SkippedFiles = append(result.SkippedFiles, file)
		} else {
			result.IndexedFiles = append(result.IndexedFiles, file)
			result.TotalChunks += indexResult.ChunkCount
		}
	}

	return result, nil
}

// ScanDirResult contains results from scanning a directory (without indexing)
type ScanDirResult struct {
	Files []string `json:"files"`
}

// ScanDirectory scans a directory for files matching a pattern without indexing them
func (r *Goldie) ScanDirectory(dir string, pattern string, recursive bool) (*ScanDirResult, error) {
	r.logger.Printf("ScanDirectory: dir=%s pattern=%s recursive=%v", dir, pattern, recursive)

	if pattern == "" {
		pattern = "*"
	}

	// Load skip patterns
	skipPatterns := r.loadSkipPatterns(dir)

	var files []string

	if recursive {
		r.logger.Printf("ScanDirectory: walking directory recursively...")
		err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return nil // Skip files we can't access
			}
			if info.IsDir() {
				// Skip directories matching skip patterns
				if len(skipPatterns) > 0 && r.shouldSkip(path, dir, skipPatterns) {
					r.logger.Printf("ScanDirectory: skipping directory: %s", path)
					return filepath.SkipDir
				}
				return nil
			}
			// Skip files matching skip patterns
			if len(skipPatterns) > 0 && r.shouldSkip(path, dir, skipPatterns) {
				return nil
			}
			matched, err := filepath.Match(pattern, filepath.Base(path))
			if err != nil {
				return nil // Invalid pattern, skip silently
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
		var err error
		files, err = filepath.Glob(globPattern)
		if err != nil {
			return nil, fmt.Errorf("glob pattern: %w", err)
		}
		// Filter out directories
		var regularFiles []string
		for _, file := range files {
			info, err := os.Stat(file)
			if err != nil || info.IsDir() {
				continue
			}
			regularFiles = append(regularFiles, file)
		}
		files = regularFiles
	}

	r.logger.Printf("ScanDirectory: found %d files", len(files))
	return &ScanDirResult{Files: files}, nil
}

// QueryResult contains search results
type QueryResult struct {
	Results []store.SearchResult `json:"results"`
	Query   string               `json:"query"`
}

// Query searches for relevant documents
func (r *Goldie) Query(query string, limit int) (*QueryResult, error) {
	if query == "" {
		return nil, fmt.Errorf("empty query")
	}

	if limit <= 0 {
		limit = 5
	}

	embedding, err := r.embedder.Embed(query)
	if err != nil {
		return nil, fmt.Errorf("generating query embedding: %w", err)
	}

	results, err := r.store.Search(embedding, limit)
	if err != nil {
		return nil, fmt.Errorf("searching: %w", err)
	}

	return &QueryResult{
		Results: results,
		Query:   query,
	}, nil
}

// GetDocument retrieves a document by ID
func (r *Goldie) GetDocument(id string) (*store.Document, error) {
	return r.store.GetDocument(id)
}

// ListDocuments returns all documents
func (r *Goldie) ListDocuments() ([]store.Document, error) {
	return r.store.ListDocuments()
}

// DeleteDocument removes a document
func (r *Goldie) DeleteDocument(id string) error {
	return r.store.DeleteDocument(id)
}

// Count returns the number of indexed documents
func (r *Goldie) Count() (int, error) {
	return r.store.Count()
}

// Store returns the underlying store for direct access
func (r *Goldie) Store() *store.Store {
	return r.store
}

// Warmup pre-loads the embedding model
func (r *Goldie) Warmup() error {
	return r.embedder.Warmup()
}

// Close closes the RAG instance
func (r *Goldie) Close() error {
	return r.store.Close()
}

// chunkText splits text into overlapping chunks
func (r *Goldie) chunkText(text string) []string {
	if len(text) <= r.chunkSize {
		return []string{text}
	}

	var chunks []string
	start := 0
	prevStart := -1

	for start < len(text) {
		// Safety: prevent infinite loop
		if start == prevStart {
			r.logger.Printf("chunkText: breaking infinite loop at start=%d", start)
			break
		}
		prevStart = start

		end := min(start+r.chunkSize, len(text))

		// Try to break at word boundary
		if end < len(text) {
			// Look for last space within chunk
			lastSpace := strings.LastIndex(text[start:end], " ")
			if lastSpace > r.chunkSize/2 {
				end = start + lastSpace
			}
		}

		chunk := strings.TrimSpace(text[start:end])
		if len(chunk) > 0 {
			chunks = append(chunks, chunk)
		}

		// Move start forward, ensuring progress
		newStart := end - r.chunkOverlap
		if newStart <= start {
			newStart = end // Force progress if overlap would cause no movement
		}
		if newStart < 0 {
			newStart = 0
		}
		start = newStart

		// Safety limit
		if len(chunks) > 10000 {
			r.logger.Printf("chunkText: hit 10000 chunk limit, stopping")
			break
		}
	}

	return chunks
}
