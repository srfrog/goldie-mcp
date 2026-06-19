package goldie

import (
	"fmt"
	"strings"

	"github.com/srfrog/goldie-mcp/internal/store"
)

// MemoryTypes is the closed set of allowed memory.type values.
var MemoryTypes = []string{
	"user",
	"feedback",
	"project",
	"reference",
	"opinion",
	"idea",
	"todo",
	"reminder",
}

var memoryTypeSet = func() map[string]struct{} {
	m := make(map[string]struct{}, len(MemoryTypes))
	for _, t := range MemoryTypes {
		m[t] = struct{}{}
	}
	return m
}()

// ValidateMemoryType returns an error if t is not a recognized memory type.
func ValidateMemoryType(t string) error {
	if _, ok := memoryTypeSet[t]; !ok {
		return fmt.Errorf("invalid memory type %q (allowed: %s)", t, strings.Join(MemoryTypes, ", "))
	}
	return nil
}

// RememberInput is the payload for creating a new memory.
type RememberInput struct {
	Name        string
	Type        string
	Body        string
	Description string
	Agent       string
	Source      string
	Hints       store.MemoryHints
}

// UpdateMemoryInput patches an existing memory. Nil fields are left unchanged;
// non-nil empty strings clear optional columns. Name is immutable.
type UpdateMemoryInput struct {
	Type        string
	Description *string
	Body        *string
	Source      *string
	Agent       *string
	Hints       store.MemoryHints
}

// Remember creates a new memory. Returns store.ErrMemoryNameExists if the
// name is already taken — callers should recall + UpdateMemory in that case.
func (g *Goldie) Remember(in RememberInput) (*store.Memory, error) {
	if in.Name == "" {
		return nil, fmt.Errorf("name is required")
	}
	if in.Body == "" {
		return nil, fmt.Errorf("body is required")
	}
	if err := ValidateMemoryType(in.Type); err != nil {
		return nil, err
	}

	chunks := g.chunkText(in.Body)
	embeddings, err := g.embedChunks(in.Name, in.Description, chunks)
	if err != nil {
		return nil, err
	}

	m := &store.Memory{
		Name:        in.Name,
		Type:        in.Type,
		Description: in.Description,
		Body:        in.Body,
		Agent:       in.Agent,
		Source:      in.Source,
	}
	if err := g.store.AddMemory(m, chunks, embeddings); err != nil {
		return nil, err
	}
	if err := g.applyHints(m.ID, in.Hints); err != nil {
		return nil, err
	}
	return g.store.GetMemoryByName(in.Name)
}

// UpdateMemory patches an existing memory by id or name. When body or
// description change, chunks are re-embedded.
func (g *Goldie) UpdateMemory(idOrName string, in UpdateMemoryInput) (*store.Memory, error) {
	existing, err := g.findMemory(idOrName)
	if err != nil {
		return nil, err
	}
	if existing == nil {
		return nil, fmt.Errorf("memory not found: %s", idOrName)
	}

	if in.Type != "" {
		if err := ValidateMemoryType(in.Type); err != nil {
			return nil, err
		}
	}

	patch := store.MemoryUpdate{
		Type:        in.Type,
		Description: in.Description,
		Body:        in.Body,
		Source:      in.Source,
		Agent:       in.Agent,
	}
	if err := g.store.UpdateMemoryFields(existing.ID, patch); err != nil {
		return nil, fmt.Errorf("updating memory: %w", err)
	}

	if in.Description != nil || in.Body != nil {
		updated, err := g.store.GetMemory(existing.ID)
		if err != nil {
			return nil, err
		}
		chunks := g.chunkText(updated.Body)
		embeddings, err := g.embedChunks(updated.Name, updated.Description, chunks)
		if err != nil {
			return nil, err
		}
		if err := g.store.ReplaceMemoryChunks(updated.ID, chunks, embeddings); err != nil {
			return nil, err
		}
	}
	if err := g.applyHints(existing.ID, in.Hints); err != nil {
		return nil, err
	}

	return g.store.GetMemory(existing.ID)
}

func (g *Goldie) applyHints(memoryID string, hints store.MemoryHints) error {
	if len(hints.About) == 0 && len(hints.Aliases) == 0 && len(hints.Links) == 0 {
		return nil
	}
	refreshIDs, err := g.store.ApplyMemoryHints(memoryID, hints)
	if err != nil {
		return err
	}
	return g.RefreshNodeEmbeddingsByID(refreshIDs)
}

// RecallMemory runs semantic search over memories, optionally filtered.
func (g *Goldie) RecallMemory(query string, limit int, filter store.MemoryFilter) ([]store.MemorySearchResult, error) {
	if query == "" {
		return nil, fmt.Errorf("empty query")
	}
	emb, err := g.embedder.Embed(query)
	if err != nil {
		return nil, fmt.Errorf("generating query embedding: %w", err)
	}
	return g.store.SearchMemories(emb, limit, filter)
}

// RecallConcept returns graph-grouped results when query exactly matches a concept.
func (g *Goldie) RecallConcept(query string, limit int, filter store.MemoryFilter) (*store.ConceptRecall, error) {
	if query == "" {
		return nil, fmt.Errorf("empty query")
	}
	return g.store.RecallConcept(query, limit, filter)
}

// RecallAutomatic returns fallback-safe automatic graph recall metadata.
func (g *Goldie) RecallAutomatic(query string, limit int, filter store.MemoryFilter) (*store.AutomaticRecall, error) {
	if query == "" {
		return nil, fmt.Errorf("empty query")
	}
	emb, err := g.embedder.Embed(query)
	if err != nil {
		return nil, fmt.Errorf("generating query embedding: %w", err)
	}
	return g.store.RecallAutomatic(query, emb, limit, filter)
}

// RefreshNodeEmbeddings rebuilds concept node embeddings for fuzzy graph recall.
func (g *Goldie) RefreshNodeEmbeddings() error {
	items, err := g.store.ListConceptNodeEmbeddingTexts()
	if err != nil {
		return err
	}
	if len(items) == 0 {
		return g.store.ReplaceNodeEmbeddings(nil)
	}
	texts := make([]string, 0, len(items))
	for _, item := range items {
		texts = append(texts, item.Text)
	}
	vectors, err := g.embedder.EmbedBatch(texts)
	if err != nil {
		return fmt.Errorf("embedding concept nodes: %w", err)
	}
	embeddings := make([]store.NodeEmbedding, 0, len(items))
	for i, item := range items {
		embeddings = append(embeddings, store.NodeEmbedding{
			ID:        item.ID,
			Embedding: vectors[i],
		})
	}
	return g.store.ReplaceNodeEmbeddings(embeddings)
}

// RefreshNodeEmbeddingsByID rebuilds concept node embeddings for selected nodes.
func (g *Goldie) RefreshNodeEmbeddingsByID(ids []string) error {
	items, err := g.store.ListConceptNodeEmbeddingTextsByID(ids)
	if err != nil {
		return err
	}
	if len(items) == 0 {
		return nil
	}
	texts := make([]string, 0, len(items))
	for _, item := range items {
		texts = append(texts, item.Text)
	}
	vectors, err := g.embedder.EmbedBatch(texts)
	if err != nil {
		return fmt.Errorf("embedding concept nodes: %w", err)
	}
	embeddings := make([]store.NodeEmbedding, 0, len(items))
	for i, item := range items {
		embeddings = append(embeddings, store.NodeEmbedding{
			ID:        item.ID,
			Embedding: vectors[i],
		})
	}
	return g.store.UpsertNodeEmbeddings(embeddings)
}

// ForgetMemory deletes memories. If query is non-empty, semantic search
// (constrained by filter) selects up to `limit` candidates and they are
// deleted. Otherwise every memory matching the filter is deleted. Refuses to
// run with both an empty filter and an empty query.
func (g *Goldie) ForgetMemory(filter store.MemoryFilter, query string, limit int) ([]store.Memory, error) {
	if filter.IsEmpty() && query == "" {
		return nil, fmt.Errorf("forget requires at least one filter (name, type, agent, source) or a query")
	}

	if query != "" {
		results, err := g.RecallMemory(query, limit, filter)
		if err != nil {
			return nil, err
		}
		var deleted []store.Memory
		for _, r := range results {
			ok, err := g.store.DeleteMemoryByID(r.Memory.ID)
			if err != nil {
				return deleted, fmt.Errorf("deleting %s: %w", r.Memory.ID, err)
			}
			if ok {
				deleted = append(deleted, r.Memory)
			}
		}
		return deleted, nil
	}

	return g.store.DeleteMemoriesByFilter(filter)
}

// ListMemories returns memories matching the filter, newest first.
func (g *Goldie) ListMemories(filter store.MemoryFilter, limit int) ([]store.Memory, error) {
	return g.store.ListMemories(filter, limit)
}

// CountMemories returns the count of memories matching the filter.
func (g *Goldie) CountMemories(filter store.MemoryFilter) (int, error) {
	return g.store.CountMemories(filter)
}

// ListNodes returns graph nodes for inspection.
func (g *Goldie) ListNodes(filter store.NodeFilter, limit int) ([]store.Node, error) {
	return g.store.ListNodes(filter, limit)
}

// GetNodeDetails returns aliases and edges for a graph node.
func (g *Goldie) GetNodeDetails(id, kind, label string) (*store.NodeDetails, error) {
	return g.store.GetNodeDetails(id, kind, label)
}

// MergeNodes merges a duplicate concept into a target concept.
func (g *Goldie) MergeNodes(sourceRef, targetRef string) (*store.MergeNodesResult, error) {
	result, err := g.store.MergeConceptNodes(sourceRef, targetRef)
	if err != nil {
		return nil, err
	}
	if err := g.store.DeleteNodeEmbeddings(result.DeleteEmbeddingIDs); err != nil {
		return nil, err
	}
	if err := g.RefreshNodeEmbeddingsByID(result.RefreshConceptIDs); err != nil {
		return nil, err
	}
	return result, nil
}

// LinkMemory manually attaches a memory to a concept.
func (g *Goldie) LinkMemory(memoryRef, conceptRef, relation string) (*store.LinkMemoryResult, error) {
	memory, err := g.findMemory(memoryRef)
	if err != nil {
		return nil, err
	}
	if memory == nil {
		return nil, fmt.Errorf("memory not found: %s", memoryRef)
	}
	result, err := g.store.LinkMemoryToConcept(memory.ID, conceptRef, relation, "manual")
	if err != nil {
		return nil, err
	}
	if err := g.RefreshNodeEmbeddingsByID(result.RefreshConceptIDs); err != nil {
		return nil, err
	}
	return result, nil
}

// GetMemory looks up a memory by id, falling back to name lookup.
func (g *Goldie) GetMemory(idOrName string) (*store.Memory, error) {
	return g.findMemory(idOrName)
}

func (g *Goldie) findMemory(idOrName string) (*store.Memory, error) {
	m, err := g.store.GetMemory(idOrName)
	if err != nil {
		return nil, err
	}
	if m != nil {
		return m, nil
	}
	return g.store.GetMemoryByName(idOrName)
}

// embedChunks generates per-chunk embeddings, prefixing each chunk text with
// the memory's name and description so semantic recall can hit on those
// fields too — not just raw body content.
func (g *Goldie) embedChunks(name, description string, chunks []string) ([][]float32, error) {
	out := make([][]float32, len(chunks))
	for i, chunk := range chunks {
		text := composeEmbedText(name, description, chunk)
		emb, err := g.embedder.Embed(text)
		if err != nil {
			return nil, fmt.Errorf("embedding chunk %d: %w", i, err)
		}
		out[i] = emb
	}
	return out, nil
}

func composeEmbedText(name, description, chunk string) string {
	var parts []string
	if name != "" {
		parts = append(parts, name)
	}
	if description != "" {
		parts = append(parts, description)
	}
	parts = append(parts, chunk)
	return strings.Join(parts, "\n\n")
}
