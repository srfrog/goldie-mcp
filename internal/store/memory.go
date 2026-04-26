package store

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
)

// ErrMemoryNameExists is returned by AddMemory when a memory with the given name already exists.
var ErrMemoryNameExists = errors.New("memory with that name already exists")

// Memory is the canonical entity stored in the index. Each memory has a unique
// human-readable name and may be backed by one or more embedded chunks for
// semantic recall.
type Memory struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Type        string    `json:"type"`
	Description string    `json:"description,omitempty"`
	Body        string    `json:"body"`
	Agent       string    `json:"agent,omitempty"`
	Source      string    `json:"source,omitempty"`
	Checksum    string    `json:"checksum,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// MemoryFilter narrows memory queries. Empty fields are ignored.
type MemoryFilter struct {
	Name   string
	Type   string
	Agent  string
	Source string
}

// IsEmpty reports whether the filter has no constraints set.
func (f MemoryFilter) IsEmpty() bool {
	return f.Name == "" && f.Type == "" && f.Agent == "" && f.Source == ""
}

func (f MemoryFilter) where(prefix string) (string, []any) {
	var clauses []string
	var args []any
	if f.Name != "" {
		clauses = append(clauses, prefix+"name = ?")
		args = append(args, f.Name)
	}
	if f.Type != "" {
		clauses = append(clauses, prefix+"type = ?")
		args = append(args, f.Type)
	}
	if f.Agent != "" {
		clauses = append(clauses, prefix+"agent = ?")
		args = append(args, f.Agent)
	}
	if f.Source != "" {
		clauses = append(clauses, prefix+"source = ?")
		args = append(args, f.Source)
	}
	return strings.Join(clauses, " AND "), args
}

// MemorySearchResult is a memory returned by semantic search, with the matched
// chunk excerpt and the underlying vector distance/score.
type MemorySearchResult struct {
	Memory   Memory  `json:"memory"`
	Excerpt  string  `json:"excerpt"`
	Score    float32 `json:"score"`
	Distance float32 `json:"distance"`
}

func (s *Store) initMemorySchema() error {
	_, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS memories (
			id TEXT PRIMARY KEY,
			name TEXT NOT NULL UNIQUE,
			type TEXT NOT NULL,
			description TEXT,
			body TEXT NOT NULL,
			agent TEXT,
			source TEXT,
			checksum TEXT,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
		)
	`)
	if err != nil {
		return fmt.Errorf("creating memories table: %w", err)
	}

	_, err = s.db.Exec(`
		CREATE TABLE IF NOT EXISTS memory_chunks (
			id TEXT PRIMARY KEY,
			memory_id TEXT NOT NULL,
			chunk_index INTEGER NOT NULL,
			content TEXT NOT NULL,
			UNIQUE(memory_id, chunk_index)
		)
	`)
	if err != nil {
		return fmt.Errorf("creating memory_chunks table: %w", err)
	}

	_, err = s.db.Exec(`CREATE INDEX IF NOT EXISTS idx_memory_chunks_memory_id ON memory_chunks(memory_id)`)
	if err != nil {
		return fmt.Errorf("creating memory_chunks index: %w", err)
	}

	query := fmt.Sprintf(`
		CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
			id TEXT PRIMARY KEY,
			embedding FLOAT[%d]
		)
	`, s.dimensions)
	if _, err := s.db.Exec(query); err != nil {
		return fmt.Errorf("creating memories_vec table: %w", err)
	}

	return nil
}

// AddMemory inserts a memory and its chunks (with embeddings) atomically.
// Returns ErrMemoryNameExists if a memory with the same name is already stored.
// chunkContents and chunkEmbeddings must have equal length.
func (s *Store) AddMemory(m *Memory, chunkContents []string, chunkEmbeddings [][]float32) error {
	if len(chunkContents) != len(chunkEmbeddings) {
		return fmt.Errorf("chunk contents (%d) and embeddings (%d) length mismatch", len(chunkContents), len(chunkEmbeddings))
	}
	if len(chunkContents) == 0 {
		return fmt.Errorf("at least one chunk is required")
	}

	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	if m.ID == "" {
		m.ID = uuid.New().String()
	}

	_, err = tx.Exec(`
		INSERT INTO memories (id, name, type, description, body, agent, source, checksum)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	`, m.ID, m.Name, m.Type, nullableString(m.Description), m.Body,
		nullableString(m.Agent), nullableString(m.Source), nullableString(m.Checksum))
	if err != nil {
		if isUniqueConstraintErr(err) {
			return ErrMemoryNameExists
		}
		return fmt.Errorf("inserting memory: %w", err)
	}

	if err := insertChunks(tx, m.ID, chunkContents, chunkEmbeddings); err != nil {
		return err
	}

	return tx.Commit()
}

// ReplaceMemoryChunks deletes all existing chunks for the given memory and
// inserts the provided chunks/embeddings. Used when a memory's body is rewritten.
func (s *Store) ReplaceMemoryChunks(memoryID string, chunkContents []string, chunkEmbeddings [][]float32) error {
	if len(chunkContents) != len(chunkEmbeddings) {
		return fmt.Errorf("chunk contents (%d) and embeddings (%d) length mismatch", len(chunkContents), len(chunkEmbeddings))
	}

	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	if err := deleteChunksTx(tx, memoryID); err != nil {
		return err
	}
	if len(chunkContents) > 0 {
		if err := insertChunks(tx, memoryID, chunkContents, chunkEmbeddings); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// UpdateMemoryFields updates non-chunk memory fields. Pass empty strings to
// leave a field untouched; pass a single space to clear an optional field.
// (Type is treated as required: empty string leaves it.)
func (s *Store) UpdateMemoryFields(id string, fields MemoryUpdate) error {
	var sets []string
	var args []any
	if fields.Type != "" {
		sets = append(sets, "type = ?")
		args = append(args, fields.Type)
	}
	if fields.Description != nil {
		sets = append(sets, "description = ?")
		args = append(args, nullableString(*fields.Description))
	}
	if fields.Body != nil {
		sets = append(sets, "body = ?")
		args = append(args, *fields.Body)
	}
	if fields.Source != nil {
		sets = append(sets, "source = ?")
		args = append(args, nullableString(*fields.Source))
	}
	if fields.Agent != nil {
		sets = append(sets, "agent = ?")
		args = append(args, nullableString(*fields.Agent))
	}
	if fields.Checksum != nil {
		sets = append(sets, "checksum = ?")
		args = append(args, nullableString(*fields.Checksum))
	}
	if len(sets) == 0 {
		return nil
	}
	sets = append(sets, "updated_at = CURRENT_TIMESTAMP")
	args = append(args, id)

	query := fmt.Sprintf("UPDATE memories SET %s WHERE id = ?", strings.Join(sets, ", "))
	res, err := s.db.Exec(query, args...)
	if err != nil {
		return fmt.Errorf("updating memory: %w", err)
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		return sql.ErrNoRows
	}
	return nil
}

// MemoryUpdate carries optional patch values for UpdateMemoryFields. A nil
// pointer means "do not change"; a non-nil empty string clears the column.
type MemoryUpdate struct {
	Type        string
	Description *string
	Body        *string
	Source      *string
	Agent       *string
	Checksum    *string
}

// GetMemory fetches a memory by id. Returns nil, nil if not found.
func (s *Store) GetMemory(id string) (*Memory, error) {
	return s.queryMemory("WHERE id = ?", id)
}

// GetMemoryByName fetches a memory by its unique name. Returns nil, nil if not found.
func (s *Store) GetMemoryByName(name string) (*Memory, error) {
	return s.queryMemory("WHERE name = ?", name)
}

func (s *Store) queryMemory(where string, args ...any) (*Memory, error) {
	row := s.db.QueryRow(`
		SELECT id, name, type, description, body, agent, source, checksum, created_at, updated_at
		FROM memories `+where, args...)
	m, err := scanMemoryRow(row)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("querying memory: %w", err)
	}
	return m, nil
}

// ListMemories returns memories matching the filter, newest first.
func (s *Store) ListMemories(filter MemoryFilter, limit int) ([]Memory, error) {
	query := `
		SELECT id, name, type, description, body, agent, source, checksum, created_at, updated_at
		FROM memories`
	var args []any
	if !filter.IsEmpty() {
		clause, fargs := filter.where("")
		query += " WHERE " + clause
		args = append(args, fargs...)
	}
	query += " ORDER BY updated_at DESC"
	if limit > 0 {
		query += " LIMIT ?"
		args = append(args, limit)
	}

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("listing memories: %w", err)
	}
	defer rows.Close()

	var out []Memory
	for rows.Next() {
		m, err := scanMemoryRow(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, *m)
	}
	return out, rows.Err()
}

// CountMemories returns the number of memories matching the filter.
func (s *Store) CountMemories(filter MemoryFilter) (int, error) {
	query := "SELECT COUNT(*) FROM memories"
	var args []any
	if !filter.IsEmpty() {
		clause, fargs := filter.where("")
		query += " WHERE " + clause
		args = append(args, fargs...)
	}
	var n int
	err := s.db.QueryRow(query, args...).Scan(&n)
	return n, err
}

// SearchMemories runs filtered KNN over the chunk vector index and returns up
// to `limit` distinct memories ordered by best chunk distance.
func (s *Store) SearchMemories(embedding []float32, limit int, filter MemoryFilter) ([]MemorySearchResult, error) {
	if limit <= 0 {
		limit = 5
	}

	embJSON, err := json.Marshal(embedding)
	if err != nil {
		return nil, fmt.Errorf("marshaling query embedding: %w", err)
	}

	// Over-fetch from vec to give the filter+dedup pass room to work.
	probeK := max(limit*5, 25)

	query := `
		SELECT
			v.distance,
			c.content,
			m.id, m.name, m.type, m.description, m.body, m.agent, m.source, m.checksum,
			m.created_at, m.updated_at
		FROM memories_vec v
		JOIN memory_chunks c ON v.id = c.id
		JOIN memories m ON c.memory_id = m.id
		WHERE v.embedding MATCH ? AND k = ?`

	args := []any{string(embJSON), probeK}
	if !filter.IsEmpty() {
		clause, fargs := filter.where("m.")
		query += " AND " + clause
		args = append(args, fargs...)
	}
	query += " ORDER BY v.distance"

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("searching memories: %w", err)
	}
	defer rows.Close()

	seen := make(map[string]struct{})
	var out []MemorySearchResult
	for rows.Next() {
		var (
			distance       float32
			excerpt        string
			id, name, typ  string
			descNS, bodyNS sql.NullString
			agentNS        sql.NullString
			sourceNS       sql.NullString
			checksumNS     sql.NullString
			createdAt      time.Time
			updatedAt      time.Time
		)
		if err := rows.Scan(
			&distance, &excerpt,
			&id, &name, &typ, &descNS, &bodyNS, &agentNS, &sourceNS, &checksumNS,
			&createdAt, &updatedAt,
		); err != nil {
			return nil, fmt.Errorf("scanning memory search row: %w", err)
		}
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}

		body := ""
		if bodyNS.Valid {
			body = bodyNS.String
		}
		out = append(out, MemorySearchResult{
			Memory: Memory{
				ID:          id,
				Name:        name,
				Type:        typ,
				Description: descNS.String,
				Body:        body,
				Agent:       agentNS.String,
				Source:      sourceNS.String,
				Checksum:    checksumNS.String,
				CreatedAt:   createdAt,
				UpdatedAt:   updatedAt,
			},
			Excerpt:  excerpt,
			Score:    1 - distance,
			Distance: distance,
		})
		if len(out) >= limit {
			break
		}
	}
	return out, rows.Err()
}

// DeleteMemoryByID removes a memory and all its chunks/vectors. Returns false
// if no memory with that id existed.
func (s *Store) DeleteMemoryByID(id string) (bool, error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	if err := deleteChunksTx(tx, id); err != nil {
		return false, err
	}
	res, err := tx.Exec("DELETE FROM memories WHERE id = ?", id)
	if err != nil {
		return false, fmt.Errorf("deleting memory: %w", err)
	}
	n, _ := res.RowsAffected()
	if err := tx.Commit(); err != nil {
		return false, fmt.Errorf("committing delete: %w", err)
	}
	return n > 0, nil
}

// DeleteMemoriesByFilter removes every memory matching the (non-empty) filter
// and returns the deleted memory rows (for caller verification).
func (s *Store) DeleteMemoriesByFilter(filter MemoryFilter) ([]Memory, error) {
	if filter.IsEmpty() {
		return nil, fmt.Errorf("refusing to delete with empty filter")
	}
	matches, err := s.ListMemories(filter, 0)
	if err != nil {
		return nil, err
	}
	for _, m := range matches {
		if _, err := s.DeleteMemoryByID(m.ID); err != nil {
			return nil, fmt.Errorf("deleting %s: %w", m.ID, err)
		}
	}
	return matches, nil
}

// --- helpers ---

func insertChunks(tx *sql.Tx, memoryID string, contents []string, embeddings [][]float32) error {
	for i, content := range contents {
		chunkID := uuid.New().String()
		if _, err := tx.Exec(
			"INSERT INTO memory_chunks (id, memory_id, chunk_index, content) VALUES (?, ?, ?, ?)",
			chunkID, memoryID, i, content,
		); err != nil {
			return fmt.Errorf("inserting chunk %d: %w", i, err)
		}
		embJSON, err := json.Marshal(embeddings[i])
		if err != nil {
			return fmt.Errorf("marshaling embedding %d: %w", i, err)
		}
		if _, err := tx.Exec(
			"INSERT INTO memories_vec (id, embedding) VALUES (?, ?)",
			chunkID, string(embJSON),
		); err != nil {
			return fmt.Errorf("inserting vector %d: %w", i, err)
		}
	}
	return nil
}

func deleteChunksTx(tx *sql.Tx, memoryID string) error {
	rows, err := tx.Query("SELECT id FROM memory_chunks WHERE memory_id = ?", memoryID)
	if err != nil {
		return fmt.Errorf("listing chunks: %w", err)
	}
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			rows.Close()
			return fmt.Errorf("scanning chunk id: %w", err)
		}
		ids = append(ids, id)
	}
	rows.Close()

	for _, id := range ids {
		if _, err := tx.Exec("DELETE FROM memories_vec WHERE id = ?", id); err != nil {
			return fmt.Errorf("deleting vec row: %w", err)
		}
	}
	if _, err := tx.Exec("DELETE FROM memory_chunks WHERE memory_id = ?", memoryID); err != nil {
		return fmt.Errorf("deleting chunks: %w", err)
	}
	return nil
}

type rowScanner interface {
	Scan(dest ...any) error
}

func scanMemoryRow(r rowScanner) (*Memory, error) {
	var (
		m                          Memory
		desc, agent, source, csum  sql.NullString
		createdAt, updatedAt       time.Time
	)
	if err := r.Scan(
		&m.ID, &m.Name, &m.Type, &desc, &m.Body, &agent, &source, &csum,
		&createdAt, &updatedAt,
	); err != nil {
		return nil, err
	}
	m.Description = desc.String
	m.Agent = agent.String
	m.Source = source.String
	m.Checksum = csum.String
	m.CreatedAt = createdAt
	m.UpdatedAt = updatedAt
	return &m, nil
}

func nullableString(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func isUniqueConstraintErr(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "UNIQUE constraint failed") || strings.Contains(msg, "constraint failed")
}
