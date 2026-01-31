// Package embedder
package embedder

import (
	"fmt"
	"math"
	"os"
	"sync"

	"github.com/srfrog/goldie-mcp/internal/embedder/minilm"
)

// Interface defines the embedding operations
type Interface interface {
	Embed(text string) ([]float32, error)
	EmbedBatch(texts []string) ([][]float32, error)
	GetDimensions() int
	Warmup() error
	Close() error
}

// Embedder generates text embeddings using all-MiniLM-L6-v2
type Embedder struct {
	model *minilm.MiniLM
	mu    sync.Mutex
}

// Ensure Embedder implements Interface
var _ Interface = (*Embedder)(nil)

// New creates a new embedder with the all-MiniLM-L6-v2 model.
func New() (*Embedder, error) {
	model, err := minilm.New(os.Getenv("ONNXRUNTIME_LIB_PATH"))
	if err != nil {
		return nil, fmt.Errorf("loading model: %w", err)
	}

	return &Embedder{
		model: model,
	}, nil
}

// Embed generates an embedding vector for the given text
func (e *Embedder) Embed(text string) ([]float32, error) {
	if text == "" {
		return nil, fmt.Errorf("empty text")
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	embedding, err := e.model.Embed(text)
	if err != nil {
		return nil, fmt.Errorf("generating embedding: %w", err)
	}

	return embedding, nil
}

// EmbedBatch generates embeddings for multiple texts
func (e *Embedder) EmbedBatch(texts []string) ([][]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	embeddings, err := e.model.EmbedBatch(texts)
	if err != nil {
		return nil, fmt.Errorf("generating batch embeddings: %w", err)
	}

	return embeddings, nil
}

// GetDimensions returns the embedding dimension size
func (e *Embedder) GetDimensions() int {
	return minilm.Dimensions
}

// Warmup pre-loads the model by running a test embedding
func (e *Embedder) Warmup() error {
	_, err := e.Embed("warmup")
	return err
}

// Close releases model resources
func (e *Embedder) Close() error {
	if e.model != nil {
		return e.model.Close()
	}
	return nil
}

// CosineSimilarity computes cosine similarity between two vectors
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return float32(dot / (math.Sqrt(normA) * math.Sqrt(normB)))
}
