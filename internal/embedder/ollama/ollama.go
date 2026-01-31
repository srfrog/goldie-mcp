// Package ollama provides text embeddings using the Ollama API.
package ollama

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// Common embedding model dimensions
const (
	DimensionsNomicEmbedText  = 768  // nomic-embed-text
	DimensionsMxbaiEmbedLarge = 1024 // mxbai-embed-large
	DimensionsAllMiniLM       = 384  // all-minilm
)

// Config holds Ollama embedder configuration
type Config struct {
	BaseURL    string // Ollama API base URL (default: http://localhost:11434)
	Model      string // Model name (default: nomic-embed-text)
	Dimensions int    // Embedding dimensions (must match model output)
}

// Ollama generates text embeddings using the Ollama API.
type Ollama struct {
	client     *http.Client
	baseURL    string
	model      string
	dimensions int
}

type embedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type embedResponse struct {
	Embedding []float32 `json:"embedding"`
}

// New creates a new Ollama embedder with the given configuration.
func New(cfg Config) (*Ollama, error) {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "http://localhost:11434"
	}
	if cfg.Model == "" {
		cfg.Model = "nomic-embed-text"
	}
	if cfg.Dimensions == 0 {
		cfg.Dimensions = DimensionsNomicEmbedText
	}

	return &Ollama{
		client:     &http.Client{Timeout: 60 * time.Second},
		baseURL:    cfg.BaseURL,
		model:      cfg.Model,
		dimensions: cfg.Dimensions,
	}, nil
}

// Embed generates an embedding vector for a single text.
func (o *Ollama) Embed(text string) ([]float32, error) {
	if text == "" {
		return nil, fmt.Errorf("empty text")
	}

	reqBody, err := json.Marshal(embedRequest{Model: o.model, Prompt: text})
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", err)
	}

	resp, err := o.client.Post(o.baseURL+"/api/embeddings", "application/json", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("ollama request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama returned status %d", resp.StatusCode)
	}

	var result embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	if len(result.Embedding) == 0 {
		return nil, fmt.Errorf("ollama returned empty embedding")
	}

	return result.Embedding, nil
}

// EmbedBatch generates embedding vectors for multiple texts.
// Note: Ollama doesn't have a native batch API, so this processes texts sequentially.
func (o *Ollama) EmbedBatch(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	results := make([][]float32, len(texts))
	for i, text := range texts {
		emb, err := o.Embed(text)
		if err != nil {
			return nil, fmt.Errorf("embedding text %d: %w", i, err)
		}
		results[i] = emb
	}
	return results, nil
}

// GetDimensions returns the embedding dimension size.
func (o *Ollama) GetDimensions() int {
	return o.dimensions
}

// Warmup pre-loads the model by running a test embedding.
func (o *Ollama) Warmup() error {
	_, err := o.Embed("warmup")
	return err
}

// Close releases resources (no-op for Ollama).
func (o *Ollama) Close() error {
	return nil
}
