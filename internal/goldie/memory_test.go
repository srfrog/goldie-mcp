package goldie

import (
	"fmt"
	"path/filepath"
	"testing"

	"github.com/srfrog/goldie-mcp/internal/store"
)

type failingEmbedder struct{ fail bool }

func (e *failingEmbedder) Embed(string) ([]float32, error) {
	if e.fail {
		return nil, fmt.Errorf("embedding backend unavailable")
	}
	return []float32{1, 0, 0}, nil
}
func (e *failingEmbedder) EmbedBatch(texts []string) ([][]float32, error) {
	var vectors [][]float32
	for _, text := range texts {
		v, err := e.Embed(text)
		if err != nil {
			return nil, err
		}
		vectors = append(vectors, v)
	}
	return vectors, nil
}
func (*failingEmbedder) GetDimensions() int { return 3 }
func (*failingEmbedder) Warmup() error      { return nil }
func (*failingEmbedder) Close() error       { return nil }

func TestUpdateMemoryEmbeddingFailurePreservesMemory(t *testing.T) {
	emb := &failingEmbedder{}
	g, err := New(Config{DBPath: filepath.Join(t.TempDir(), "test.db"), Dimensions: 3, Embedder: emb})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	m, err := g.Remember(RememberInput{Name: "decision", Type: "project", Body: "old body", Description: "old description"})
	if err != nil {
		t.Fatal(err)
	}
	emb.fail = true
	body, description := "new body", "new description"
	for _, patch := range []UpdateMemoryInput{{Body: &body}, {Description: &description}} {
		if _, err := g.UpdateMemory(m.ID, patch); err == nil {
			t.Fatal("expected embedding failure")
		}
		got, err := g.GetMemory(m.ID)
		if err != nil {
			t.Fatal(err)
		}
		if got.Body != m.Body || got.Description != m.Description {
			t.Fatalf("failed update changed memory: %+v", got)
		}
	}
	emb.fail = false
	results, err := g.RecallMemory("decision", 5, store.MemoryFilter{})
	if err != nil || len(results) != 1 || results[0].Excerpt != m.Body {
		t.Fatalf("original memory no longer searchable: %+v, %v", results, err)
	}
}
