package store

import (
	"fmt"
	"testing"
)

func TestSearchMemoriesFiltersBeforeRanking(t *testing.T) {
	st := newTestStore(t)
	for i := 0; i < 30; i++ {
		m := &Memory{Name: fmt.Sprintf("near_%d", i), Type: "user", Body: "near"}
		if err := st.AddMemory(m, []string{m.Body}, [][]float32{{0, 0, 0}}); err != nil {
			t.Fatal(err)
		}
	}
	m := &Memory{Name: "target", Type: "project", Agent: "codex", Source: "conversation", Body: "target"}
	if err := st.AddMemory(m, []string{m.Body}, [][]float32{{1, 0, 0}}); err != nil {
		t.Fatal(err)
	}
	for _, filter := range []MemoryFilter{{Type: m.Type}, {Agent: m.Agent}, {Source: m.Source}, {Name: m.Name}} {
		results, err := st.SearchMemories([]float32{0, 0, 0}, 5, filter)
		if err != nil {
			t.Fatal(err)
		}
		if len(results) != 1 || results[0].Memory.ID != m.ID {
			t.Fatalf("filter %+v: expected target, got %+v", filter, results)
		}
	}
}

func TestSearchMemoriesFillsLimitWithDistinctMemories(t *testing.T) {
	st := newTestStore(t)
	chunks := make([]string, 30)
	vectors := make([][]float32, 30)
	for i := range chunks {
		chunks[i], vectors[i] = "near", []float32{0, 0, 0}
	}
	if err := st.AddMemory(&Memory{Name: "large", Type: "project", Body: "near"}, chunks, vectors); err != nil {
		t.Fatal(err)
	}
	for i := 1; i < 5; i++ {
		m := &Memory{Name: fmt.Sprintf("other_%d", i), Type: "project", Body: "farther"}
		if err := st.AddMemory(m, []string{m.Body}, [][]float32{{float32(i), 0, 0}}); err != nil {
			t.Fatal(err)
		}
	}
	results, err := st.SearchMemories([]float32{0, 0, 0}, 5, MemoryFilter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 5 || results[0].Memory.Name != "large" || results[4].Distance != 4 {
		t.Fatalf("expected five distinct memories ordered by distance, got %+v", results)
	}
}

func TestUpdateMemoryWithChunksRollsBackOnVectorFailure(t *testing.T) {
	st := newTestStore(t)
	m := addTestMemory(t, st, "original")
	body, checksum := "replacement", "new-checksum"
	err := st.UpdateMemoryWithChunks(m.ID, MemoryUpdate{Body: &body, Checksum: &checksum}, []string{body}, [][]float32{{1, 2}})
	if err == nil {
		t.Fatal("expected wrong embedding dimensions to fail")
	}
	got, err := st.GetMemory(m.ID)
	if err != nil {
		t.Fatal(err)
	}
	results, err := st.SearchMemories([]float32{0.1, 0.2, 0.3}, 5, MemoryFilter{})
	if err != nil {
		t.Fatal(err)
	}
	if got.Body != m.Body || got.Checksum != "" || len(results) != 1 || results[0].Excerpt != m.Body {
		t.Fatalf("failed update changed memory or chunks: %+v, %+v", got, results)
	}
}
