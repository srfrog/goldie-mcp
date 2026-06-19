package store

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"unicode"

	"github.com/google/uuid"
)

const (
	NodeKindMemory        = "memory"
	NodeKindConcept       = "concept"
	JobTypeGraphBackfill  = "graph_backfill"
	JobTypeGraphHarvest   = "graph_harvest"
	RelationAbout         = "about"
	RelationMentions      = "mentions"
	RelationCanonicalFor  = "canonical_for"
	RelationPartOf        = "part_of"
	defaultFacetThreshold = 2
)

type Node struct {
	ID              string
	Kind            string
	Label           string
	NormalizedLabel string
	AliasCount      int
	OutgoingCount   int
	IncomingCount   int
}

type NodeFilter struct {
	Kind  string
	Query string
}

type ConceptRecall struct {
	Concept Node
	Groups  []ConceptRecallGroup
	Direct  []Memory
}

type ConceptRecallGroup struct {
	Concept  Node
	Memories []Memory
}

type AutomaticRecall struct {
	Vantage    Node
	MatchedBy  string
	Path       []AutomaticRecallPathItem
	Candidates []AutomaticRecallCandidate
	Memories   []Memory
	Converged  bool
	StopReason string
}

type AutomaticRecallPathItem struct {
	Concept Node
	Score   float64
	Reason  string
}

type AutomaticRecallCandidate struct {
	Concept     Node
	Score       float64
	MemoryCount int
}

type NodeEmbeddingText struct {
	ID   string
	Text string
}

type NodeEmbedding struct {
	ID        string
	Embedding []float32
}

type NodeSearchResult struct {
	Node     Node
	Score    float32
	Distance float32
}

type NodeAlias struct {
	Alias           string
	NormalizedAlias string
	Source          string
}

type EdgeDetail struct {
	Relation         string
	Origin           string
	Confidence       float64
	EvidenceMemoryID string
	Node             Node
}

type NodeDetails struct {
	Node     Node
	Aliases  []NodeAlias
	Outgoing []EdgeDetail
	Incoming []EdgeDetail
}

type MergeNodesResult struct {
	Target       Node
	Source       Node
	AliasesMoved int
	EdgesMoved   int
}

type LinkMemoryResult struct {
	Memory   Memory
	Concept  Node
	Relation string
	Origin   string
}

func (s *Store) initGraphSchema() error {
	_, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS nodes (
			id TEXT PRIMARY KEY,
			kind TEXT NOT NULL,
			label TEXT NOT NULL,
			normalized_label TEXT NOT NULL,
			description TEXT NOT NULL DEFAULT '',
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			UNIQUE(kind, normalized_label)
		)
	`)
	if err != nil {
		return fmt.Errorf("creating nodes table: %w", err)
	}

	_, err = s.db.Exec(`
		CREATE TABLE IF NOT EXISTS node_aliases (
			id TEXT PRIMARY KEY,
			node_id TEXT NOT NULL REFERENCES nodes(id),
			alias TEXT NOT NULL,
			normalized_alias TEXT NOT NULL,
			source TEXT NOT NULL DEFAULT '',
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			UNIQUE(node_id, normalized_alias)
		)
	`)
	if err != nil {
		return fmt.Errorf("creating node_aliases table: %w", err)
	}

	_, err = s.db.Exec(`
		CREATE INDEX IF NOT EXISTS node_aliases_normalized_alias_idx
		ON node_aliases(normalized_alias)
	`)
	if err != nil {
		return fmt.Errorf("creating node aliases index: %w", err)
	}

	_, err = s.db.Exec(`
		CREATE TABLE IF NOT EXISTS edges (
			id TEXT PRIMARY KEY,
			src_node_id TEXT NOT NULL REFERENCES nodes(id),
			relation TEXT NOT NULL,
			dst_node_id TEXT NOT NULL REFERENCES nodes(id),
			confidence REAL NOT NULL DEFAULT 1.0,
			evidence_memory_id TEXT REFERENCES memories(id),
			origin TEXT NOT NULL DEFAULT 'harvest',
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			UNIQUE(src_node_id, relation, dst_node_id)
		)
	`)
	if err != nil {
		return fmt.Errorf("creating edges table: %w", err)
	}

	_, err = s.db.Exec(`
		CREATE INDEX IF NOT EXISTS edges_src_relation_idx
		ON edges(src_node_id, relation)
	`)
	if err != nil {
		return fmt.Errorf("creating edge source index: %w", err)
	}

	_, err = s.db.Exec(`
		CREATE INDEX IF NOT EXISTS edges_dst_relation_idx
		ON edges(dst_node_id, relation)
	`)
	if err != nil {
		return fmt.Errorf("creating edge destination index: %w", err)
	}

	if err := s.ensureMemoryNodeIDColumn(); err != nil {
		return err
	}
	if err := s.ensureEdgeOriginColumn(); err != nil {
		return err
	}
	query := fmt.Sprintf(`
		CREATE VIRTUAL TABLE IF NOT EXISTS nodes_vec USING vec0(
			id TEXT PRIMARY KEY,
			embedding FLOAT[%d]
		)
	`, s.dimensions)
	if _, err := s.db.Exec(query); err != nil {
		return fmt.Errorf("creating nodes_vec table: %w", err)
	}
	return nil
}

func (s *Store) ensureMemoryNodeIDColumn() error {
	rows, err := s.db.Query("PRAGMA table_info(memories)")
	if err != nil {
		return fmt.Errorf("reading memories columns: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var cid int
		var name, typ string
		var notNull int
		var defaultValue any
		var pk int
		if err := rows.Scan(&cid, &name, &typ, &notNull, &defaultValue, &pk); err != nil {
			return fmt.Errorf("scanning memories column: %w", err)
		}
		if name == "node_id" {
			return rows.Err()
		}
	}
	if err := rows.Err(); err != nil {
		return err
	}

	_, err = s.db.Exec("ALTER TABLE memories ADD COLUMN node_id TEXT REFERENCES nodes(id)")
	if err != nil {
		return fmt.Errorf("adding memories.node_id: %w", err)
	}
	return nil
}

func (s *Store) ensureEdgeOriginColumn() error {
	rows, err := s.db.Query("PRAGMA table_info(edges)")
	if err != nil {
		return fmt.Errorf("reading edges columns: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var cid int
		var name, typ string
		var notNull int
		var defaultValue any
		var pk int
		if err := rows.Scan(&cid, &name, &typ, &notNull, &defaultValue, &pk); err != nil {
			return fmt.Errorf("scanning edges column: %w", err)
		}
		if name == "origin" {
			return rows.Err()
		}
	}
	if err := rows.Err(); err != nil {
		return err
	}

	_, err = s.db.Exec("ALTER TABLE edges ADD COLUMN origin TEXT NOT NULL DEFAULT 'harvest'")
	if err != nil {
		return fmt.Errorf("adding edges.origin: %w", err)
	}
	return nil
}

// NormalizeLabel returns the canonical string used for exact graph lookups.
func NormalizeLabel(s string) string {
	return strings.Join(strings.Fields(strings.ToLower(strings.TrimSpace(s))), " ")
}

// EnsureMemoryNode creates and links the memory graph node if it is missing.
func (s *Store) EnsureMemoryNode(memoryID string) error {
	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	var name string
	var nodeID sql.NullString
	err = tx.QueryRow("SELECT name, node_id FROM memories WHERE id = ?", memoryID).Scan(&name, &nodeID)
	if err == sql.ErrNoRows {
		return sql.ErrNoRows
	}
	if err != nil {
		return fmt.Errorf("querying memory for graph node: %w", err)
	}
	if nodeID.Valid && nodeID.String != "" {
		return tx.Commit()
	}
	if _, err := ensureMemoryNodeTx(tx, memoryID, name); err != nil {
		return err
	}
	return tx.Commit()
}

func ensureMemoryNodeTx(tx *sql.Tx, memoryID, memoryName string) (string, error) {
	var existing sql.NullString
	err := tx.QueryRow("SELECT node_id FROM memories WHERE id = ?", memoryID).Scan(&existing)
	if err != nil {
		return "", fmt.Errorf("querying memory node id: %w", err)
	}
	if existing.Valid && existing.String != "" {
		return existing.String, nil
	}

	nodeID := uuid.New().String()
	_, err = tx.Exec(`
		INSERT INTO nodes (id, kind, label, normalized_label)
		VALUES (?, ?, ?, ?)
		ON CONFLICT(kind, normalized_label) DO NOTHING
	`, nodeID, NodeKindMemory, memoryName, memoryID)
	if err != nil {
		return "", fmt.Errorf("inserting memory node: %w", err)
	}

	err = tx.QueryRow(
		"SELECT id FROM nodes WHERE kind = ? AND normalized_label = ?",
		NodeKindMemory, memoryID,
	).Scan(&nodeID)
	if err != nil {
		return "", fmt.Errorf("querying memory node: %w", err)
	}

	if _, err := tx.Exec("UPDATE memories SET node_id = ? WHERE id = ?", nodeID, memoryID); err != nil {
		return "", fmt.Errorf("linking memory node: %w", err)
	}
	return nodeID, nil
}

func ensureConceptNodeTx(tx *sql.Tx, label string) (string, error) {
	label = strings.TrimSpace(label)
	if label == "" {
		return "", fmt.Errorf("empty concept label")
	}
	normalized := NormalizeLabel(label)
	nodeID := uuid.New().String()
	_, err := tx.Exec(`
		INSERT INTO nodes (id, kind, label, normalized_label)
		VALUES (?, ?, ?, ?)
		ON CONFLICT(kind, normalized_label) DO NOTHING
	`, nodeID, NodeKindConcept, label, normalized)
	if err != nil {
		return "", fmt.Errorf("inserting concept node: %w", err)
	}
	err = tx.QueryRow(
		"SELECT id FROM nodes WHERE kind = ? AND normalized_label = ?",
		NodeKindConcept, normalized,
	).Scan(&nodeID)
	if err != nil {
		return "", fmt.Errorf("querying concept node: %w", err)
	}
	return nodeID, nil
}

func ensureOrResolveConceptNodeTx(tx *sql.Tx, ref string) (string, error) {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return "", fmt.Errorf("concept reference is required")
	}
	node, err := queryNodeTx(tx, `
		SELECT id, kind, label, normalized_label
		FROM nodes
		WHERE kind = ? AND id = ?
	`, NodeKindConcept, ref)
	if err != nil {
		return "", err
	}
	if node != nil {
		return node.ID, nil
	}

	normalized := NormalizeLabel(ref)
	node, err = queryNodeTx(tx, `
		SELECT id, kind, label, normalized_label
		FROM nodes
		WHERE kind = ? AND normalized_label = ?
	`, NodeKindConcept, normalized)
	if err != nil {
		return "", err
	}
	if node != nil {
		return node.ID, nil
	}

	node, err = queryNodeTx(tx, `
		SELECT n.id, n.kind, n.label, n.normalized_label
		FROM node_aliases a
		JOIN nodes n ON a.node_id = n.id
		WHERE n.kind = ? AND a.normalized_alias = ?
	`, NodeKindConcept, normalized)
	if err != nil {
		return "", err
	}
	if node != nil {
		return node.ID, nil
	}
	return ensureConceptNodeTx(tx, ref)
}

func ensureEdgeTx(tx *sql.Tx, srcNodeID, relation, dstNodeID, evidenceMemoryID string) error {
	return ensureEdgeWithOriginTx(tx, srcNodeID, relation, dstNodeID, evidenceMemoryID, "harvest")
}

func ensureEdgeWithOriginTx(tx *sql.Tx, srcNodeID, relation, dstNodeID, evidenceMemoryID, origin string) error {
	if srcNodeID == "" || dstNodeID == "" {
		return nil
	}
	if origin == "" {
		origin = "harvest"
	}
	edgeID := uuid.New().String()
	var evidence any
	if evidenceMemoryID != "" {
		evidence = evidenceMemoryID
	}
	_, err := tx.Exec(`
		INSERT INTO edges (id, src_node_id, relation, dst_node_id, evidence_memory_id, origin)
		VALUES (?, ?, ?, ?, ?, ?)
		ON CONFLICT(src_node_id, relation, dst_node_id) DO NOTHING
	`, edgeID, srcNodeID, relation, dstNodeID, evidence, origin)
	if err != nil {
		return fmt.Errorf("inserting graph edge: %w", err)
	}
	return nil
}

func ensureAliasTx(tx *sql.Tx, nodeID, alias, source string) error {
	normalized := NormalizeLabel(alias)
	if nodeID == "" || normalized == "" {
		return nil
	}
	aliasID := uuid.New().String()
	_, err := tx.Exec(`
		INSERT INTO node_aliases (id, node_id, alias, normalized_alias, source)
		VALUES (?, ?, ?, ?, ?)
		ON CONFLICT(node_id, normalized_alias) DO NOTHING
	`, aliasID, nodeID, alias, normalized, source)
	if err != nil {
		return fmt.Errorf("inserting node alias: %w", err)
	}
	return nil
}

func queryNodeTx(tx *sql.Tx, query string, args ...any) (*Node, error) {
	var node Node
	err := tx.QueryRow(query, args...).Scan(
		&node.ID,
		&node.Kind,
		&node.Label,
		&node.NormalizedLabel,
	)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("querying graph node: %w", err)
	}
	return &node, nil
}

func getMemoryTx(tx *sql.Tx, id string) (*Memory, error) {
	row := tx.QueryRow(`
		SELECT id, name, type, description, body, agent, source, checksum, node_id, created_at, updated_at
		FROM memories
		WHERE id = ?
	`, id)
	memory, err := scanMemoryRow(row)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("querying memory: %w", err)
	}
	return memory, nil
}

func moveNodeAliasesTx(tx *sql.Tx, sourceNodeID, targetNodeID, sourceLabel string) (int, error) {
	moved := 0
	if err := ensureAliasTx(tx, targetNodeID, sourceLabel, "merge"); err != nil {
		return 0, err
	}
	moved++

	rows, err := tx.Query(`
		SELECT alias, source
		FROM node_aliases
		WHERE node_id = ?
	`, sourceNodeID)
	if err != nil {
		return 0, fmt.Errorf("querying source aliases: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var alias, source string
		if err := rows.Scan(&alias, &source); err != nil {
			return 0, fmt.Errorf("scanning source alias: %w", err)
		}
		if err := ensureAliasTx(tx, targetNodeID, alias, source); err != nil {
			return 0, err
		}
		moved++
	}
	return moved, rows.Err()
}

func moveNodeEdgesTx(tx *sql.Tx, sourceNodeID, targetNodeID string) (int, error) {
	moved := 0
	type edgeCopy struct {
		srcNodeID        string
		relation         string
		dstNodeID        string
		confidence       float64
		evidenceMemoryID string
		origin           string
	}
	var edges []edgeCopy

	outgoing, err := tx.Query(`
		SELECT relation, dst_node_id, confidence, COALESCE(evidence_memory_id, ''), origin
		FROM edges
		WHERE src_node_id = ?
	`, sourceNodeID)
	if err != nil {
		return 0, fmt.Errorf("querying outgoing source edges: %w", err)
	}
	for outgoing.Next() {
		var relation, dstNodeID, evidenceMemoryID, origin string
		var confidence float64
		if err := outgoing.Scan(&relation, &dstNodeID, &confidence, &evidenceMemoryID, &origin); err != nil {
			outgoing.Close()
			return 0, fmt.Errorf("scanning outgoing source edge: %w", err)
		}
		if dstNodeID == targetNodeID {
			continue
		}
		edges = append(edges, edgeCopy{
			srcNodeID:        targetNodeID,
			relation:         relation,
			dstNodeID:        dstNodeID,
			confidence:       confidence,
			evidenceMemoryID: evidenceMemoryID,
			origin:           origin,
		})
	}
	if err := outgoing.Close(); err != nil {
		return 0, err
	}
	if err := outgoing.Err(); err != nil {
		return 0, err
	}

	incoming, err := tx.Query(`
		SELECT src_node_id, relation, confidence, COALESCE(evidence_memory_id, ''), origin
		FROM edges
		WHERE dst_node_id = ?
	`, sourceNodeID)
	if err != nil {
		return 0, fmt.Errorf("querying incoming source edges: %w", err)
	}
	for incoming.Next() {
		var srcNodeID, relation, evidenceMemoryID, origin string
		var confidence float64
		if err := incoming.Scan(&srcNodeID, &relation, &confidence, &evidenceMemoryID, &origin); err != nil {
			incoming.Close()
			return 0, fmt.Errorf("scanning incoming source edge: %w", err)
		}
		if srcNodeID == targetNodeID {
			continue
		}
		edges = append(edges, edgeCopy{
			srcNodeID:        srcNodeID,
			relation:         relation,
			dstNodeID:        targetNodeID,
			confidence:       confidence,
			evidenceMemoryID: evidenceMemoryID,
			origin:           origin,
		})
	}
	if err := incoming.Close(); err != nil {
		return 0, err
	}
	if err := incoming.Err(); err != nil {
		return 0, err
	}
	for _, edge := range edges {
		if err := insertEdgeCopyTx(tx, edge.srcNodeID, edge.relation, edge.dstNodeID, edge.confidence, edge.evidenceMemoryID, edge.origin); err != nil {
			return 0, err
		}
		moved++
	}
	return moved, nil
}

func insertEdgeCopyTx(tx *sql.Tx, srcNodeID, relation, dstNodeID string, confidence float64, evidenceMemoryID, origin string) error {
	var evidence any
	if evidenceMemoryID != "" {
		evidence = evidenceMemoryID
	}
	_, err := tx.Exec(`
		INSERT INTO edges (id, src_node_id, relation, dst_node_id, confidence, evidence_memory_id, origin)
		VALUES (?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(src_node_id, relation, dst_node_id) DO NOTHING
	`, uuid.New().String(), srcNodeID, relation, dstNodeID, confidence, evidence, origin)
	if err != nil {
		return fmt.Errorf("copying graph edge: %w", err)
	}
	return nil
}

func validMemoryConceptRelation(relation string) bool {
	switch relation {
	case RelationAbout, RelationMentions, RelationCanonicalFor:
		return true
	default:
		return false
	}
}

func deleteMemoryGraphTx(tx *sql.Tx, memoryID, nodeID string) error {
	if nodeID != "" {
		if _, err := tx.Exec("DELETE FROM edges WHERE src_node_id = ? OR dst_node_id = ?", nodeID, nodeID); err != nil {
			return fmt.Errorf("deleting memory node edges: %w", err)
		}
	}
	if _, err := tx.Exec("DELETE FROM edges WHERE evidence_memory_id = ?", memoryID); err != nil {
		return fmt.Errorf("deleting memory evidence edges: %w", err)
	}
	if nodeID != "" {
		if _, err := tx.Exec("DELETE FROM node_aliases WHERE node_id = ?", nodeID); err != nil {
			return fmt.Errorf("deleting memory node aliases: %w", err)
		}
	}
	return nil
}

func deleteMemoryNodeTx(tx *sql.Tx, nodeID string) error {
	if nodeID == "" {
		return nil
	}
	if _, err := tx.Exec("DELETE FROM nodes WHERE id = ?", nodeID); err != nil {
		return fmt.Errorf("deleting memory node: %w", err)
	}
	return nil
}

// CountNodes returns the number of graph nodes, optionally filtered by kind.
func (s *Store) CountNodes(kind string) (int, error) {
	query := "SELECT COUNT(*) FROM nodes"
	var args []any
	if kind != "" {
		query += " WHERE kind = ?"
		args = append(args, kind)
	}
	var n int
	err := s.db.QueryRow(query, args...).Scan(&n)
	return n, err
}

// ListNodes returns graph nodes with lightweight alias/edge counts for inspection.
func (s *Store) ListNodes(filter NodeFilter, limit int) ([]Node, error) {
	if limit <= 0 {
		limit = 50
	}
	if limit > 200 {
		limit = 200
	}

	query := `
		SELECT
			n.id,
			n.kind,
			n.label,
			n.normalized_label,
			(SELECT COUNT(*) FROM node_aliases a WHERE a.node_id = n.id) AS alias_count,
			(SELECT COUNT(*) FROM edges e WHERE e.src_node_id = n.id) AS outgoing_count,
			(SELECT COUNT(*) FROM edges e WHERE e.dst_node_id = n.id) AS incoming_count
		FROM nodes n`
	var clauses []string
	var args []any
	if filter.Kind != "" {
		clauses = append(clauses, "n.kind = ?")
		args = append(args, filter.Kind)
	}
	if filter.Query != "" {
		clauses = append(clauses, `(n.normalized_label LIKE ? OR EXISTS (
			SELECT 1 FROM node_aliases a
			WHERE a.node_id = n.id AND a.normalized_alias LIKE ?
		))`)
		q := "%" + NormalizeLabel(filter.Query) + "%"
		args = append(args, q, q)
	}
	if len(clauses) > 0 {
		query += " WHERE " + strings.Join(clauses, " AND ")
	}
	query += " ORDER BY n.kind, n.label LIMIT ?"
	args = append(args, limit)

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("listing nodes: %w", err)
	}
	defer rows.Close()

	var out []Node
	for rows.Next() {
		var n Node
		if err := rows.Scan(
			&n.ID,
			&n.Kind,
			&n.Label,
			&n.NormalizedLabel,
			&n.AliasCount,
			&n.OutgoingCount,
			&n.IncomingCount,
		); err != nil {
			return nil, fmt.Errorf("scanning node: %w", err)
		}
		out = append(out, n)
	}
	return out, rows.Err()
}

func (s *Store) ListConceptNodeEmbeddingTexts() ([]NodeEmbeddingText, error) {
	rows, err := s.db.Query(`
		SELECT
			n.id,
			n.label || COALESCE(char(10) || GROUP_CONCAT(a.alias, char(10)), '')
		FROM nodes n
		LEFT JOIN node_aliases a ON a.node_id = n.id
		WHERE n.kind = ?
		GROUP BY n.id, n.label
		ORDER BY n.label
	`, NodeKindConcept)
	if err != nil {
		return nil, fmt.Errorf("listing concept node texts: %w", err)
	}
	defer rows.Close()

	var out []NodeEmbeddingText
	for rows.Next() {
		var item NodeEmbeddingText
		if err := rows.Scan(&item.ID, &item.Text); err != nil {
			return nil, fmt.Errorf("scanning concept node text: %w", err)
		}
		out = append(out, item)
	}
	return out, rows.Err()
}

func (s *Store) ReplaceNodeEmbeddings(embeddings []NodeEmbedding) error {
	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("beginning node embedding transaction: %w", err)
	}
	defer tx.Rollback()

	if _, err := tx.Exec("DELETE FROM nodes_vec"); err != nil {
		return fmt.Errorf("clearing node vectors: %w", err)
	}
	for _, item := range embeddings {
		embJSON, err := json.Marshal(item.Embedding)
		if err != nil {
			return fmt.Errorf("marshaling node embedding: %w", err)
		}
		if _, err := tx.Exec("INSERT INTO nodes_vec (id, embedding) VALUES (?, ?)", item.ID, string(embJSON)); err != nil {
			return fmt.Errorf("inserting node vector: %w", err)
		}
	}
	return tx.Commit()
}

func (s *Store) SearchConceptNodes(embedding []float32, limit int) ([]NodeSearchResult, error) {
	if limit <= 0 {
		limit = 5
	}
	embJSON, err := json.Marshal(embedding)
	if err != nil {
		return nil, fmt.Errorf("marshaling node query embedding: %w", err)
	}
	rows, err := s.db.Query(`
		SELECT
			v.distance,
			n.id, n.kind, n.label, n.normalized_label
		FROM nodes_vec v
		JOIN nodes n ON n.id = v.id
		WHERE v.embedding MATCH ? AND k = ? AND n.kind = ?
		ORDER BY v.distance
	`, string(embJSON), limit, NodeKindConcept)
	if err != nil {
		return nil, fmt.Errorf("searching concept nodes: %w", err)
	}
	defer rows.Close()

	var out []NodeSearchResult
	for rows.Next() {
		var result NodeSearchResult
		if err := rows.Scan(
			&result.Distance,
			&result.Node.ID,
			&result.Node.Kind,
			&result.Node.Label,
			&result.Node.NormalizedLabel,
		); err != nil {
			return nil, fmt.Errorf("scanning concept node search row: %w", err)
		}
		result.Score = 1 - result.Distance
		out = append(out, result)
	}
	return out, rows.Err()
}

func (s *Store) ResolveConceptNode(ref string) (*Node, error) {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return nil, fmt.Errorf("concept reference is required")
	}
	node, err := s.queryNode(`
		SELECT id, kind, label, normalized_label
		FROM nodes
		WHERE kind = ? AND id = ?
	`, NodeKindConcept, ref)
	if err != nil || node != nil {
		return node, err
	}
	return s.findConcept(ref)
}

func (s *Store) GetNodeDetails(id, kind, label string) (*NodeDetails, error) {
	node, err := s.resolveNode(id, kind, label)
	if err != nil || node == nil {
		return nil, err
	}
	aliases, err := s.nodeAliases(node.ID)
	if err != nil {
		return nil, err
	}
	outgoing, err := s.nodeEdges(node.ID, true)
	if err != nil {
		return nil, err
	}
	incoming, err := s.nodeEdges(node.ID, false)
	if err != nil {
		return nil, err
	}
	return &NodeDetails{
		Node:     *node,
		Aliases:  aliases,
		Outgoing: outgoing,
		Incoming: incoming,
	}, nil
}

func (s *Store) MergeConceptNodes(sourceRef, targetRef string) (*MergeNodesResult, error) {
	source, err := s.ResolveConceptNode(sourceRef)
	if err != nil {
		return nil, err
	}
	if source == nil {
		return nil, fmt.Errorf("source concept not found: %s", sourceRef)
	}
	target, err := s.ResolveConceptNode(targetRef)
	if err != nil {
		return nil, err
	}
	if target == nil {
		return nil, fmt.Errorf("target concept not found: %s", targetRef)
	}
	if source.ID == target.ID {
		return nil, fmt.Errorf("source and target are the same concept")
	}

	tx, err := s.db.Begin()
	if err != nil {
		return nil, fmt.Errorf("beginning merge transaction: %w", err)
	}
	defer tx.Rollback()

	aliasesMoved, err := moveNodeAliasesTx(tx, source.ID, target.ID, source.Label)
	if err != nil {
		return nil, err
	}
	edgesMoved, err := moveNodeEdgesTx(tx, source.ID, target.ID)
	if err != nil {
		return nil, err
	}
	if _, err := tx.Exec("DELETE FROM nodes_vec WHERE id = ?", source.ID); err != nil {
		return nil, fmt.Errorf("deleting source node vector: %w", err)
	}
	if _, err := tx.Exec("DELETE FROM node_aliases WHERE node_id = ?", source.ID); err != nil {
		return nil, fmt.Errorf("deleting source aliases: %w", err)
	}
	if _, err := tx.Exec("DELETE FROM edges WHERE src_node_id = ? OR dst_node_id = ?", source.ID, source.ID); err != nil {
		return nil, fmt.Errorf("deleting source edges: %w", err)
	}
	if _, err := tx.Exec("DELETE FROM nodes WHERE id = ?", source.ID); err != nil {
		return nil, fmt.Errorf("deleting source node: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("committing node merge: %w", err)
	}

	return &MergeNodesResult{
		Target:       *target,
		Source:       *source,
		AliasesMoved: aliasesMoved,
		EdgesMoved:   edgesMoved,
	}, nil
}

func (s *Store) LinkMemoryToConcept(memoryID, conceptRef, relation, origin string) (*LinkMemoryResult, error) {
	if relation == "" {
		relation = RelationAbout
	}
	if origin == "" {
		origin = "manual"
	}
	if !validMemoryConceptRelation(relation) {
		return nil, fmt.Errorf("invalid memory concept relation %q", relation)
	}

	tx, err := s.db.Begin()
	if err != nil {
		return nil, fmt.Errorf("beginning link transaction: %w", err)
	}
	defer tx.Rollback()

	memory, err := getMemoryTx(tx, memoryID)
	if err != nil {
		return nil, err
	}
	if memory == nil {
		return nil, fmt.Errorf("memory not found: %s", memoryID)
	}
	if memory.NodeID == "" {
		nodeID, err := ensureMemoryNodeTx(tx, memory.ID, memory.Name)
		if err != nil {
			return nil, err
		}
		memory.NodeID = nodeID
	}

	conceptID, err := ensureOrResolveConceptNodeTx(tx, conceptRef)
	if err != nil {
		return nil, err
	}
	if err := ensureEdgeWithOriginTx(tx, memory.NodeID, relation, conceptID, memory.ID, origin); err != nil {
		return nil, err
	}
	concept, err := queryNodeTx(tx, `
		SELECT id, kind, label, normalized_label
		FROM nodes
		WHERE id = ?
	`, conceptID)
	if err != nil {
		return nil, err
	}
	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("committing memory link: %w", err)
	}

	return &LinkMemoryResult{
		Memory:   *memory,
		Concept:  *concept,
		Relation: relation,
		Origin:   origin,
	}, nil
}

func (s *Store) resolveNode(id, kind, label string) (*Node, error) {
	id = strings.TrimSpace(id)
	kind = strings.TrimSpace(kind)
	label = strings.TrimSpace(label)
	if id != "" {
		query := `
			SELECT id, kind, label, normalized_label
			FROM nodes
			WHERE id = ?`
		args := []any{id}
		if kind != "" {
			query += " AND kind = ?"
			args = append(args, kind)
		}
		return s.queryNode(query, args...)
	}
	if label == "" {
		return nil, fmt.Errorf("id or label is required")
	}
	if kind == "" {
		kind = NodeKindConcept
	}
	return s.queryNode(`
		SELECT id, kind, label, normalized_label
		FROM nodes
		WHERE kind = ? AND normalized_label = ?
	`, kind, NormalizeLabel(label))
}

func (s *Store) nodeAliases(nodeID string) ([]NodeAlias, error) {
	rows, err := s.db.Query(`
		SELECT alias, normalized_alias, source
		FROM node_aliases
		WHERE node_id = ?
		ORDER BY alias
	`, nodeID)
	if err != nil {
		return nil, fmt.Errorf("querying node aliases: %w", err)
	}
	defer rows.Close()

	var out []NodeAlias
	for rows.Next() {
		var alias NodeAlias
		if err := rows.Scan(&alias.Alias, &alias.NormalizedAlias, &alias.Source); err != nil {
			return nil, fmt.Errorf("scanning node alias: %w", err)
		}
		out = append(out, alias)
	}
	return out, rows.Err()
}

func (s *Store) nodeEdges(nodeID string, outgoing bool) ([]EdgeDetail, error) {
	selectNode := "e.dst_node_id"
	where := "e.src_node_id = ?"
	if !outgoing {
		selectNode = "e.src_node_id"
		where = "e.dst_node_id = ?"
	}
	rows, err := s.db.Query(fmt.Sprintf(`
		SELECT
			e.relation,
			e.origin,
			e.confidence,
			COALESCE(e.evidence_memory_id, ''),
			n.id,
			n.kind,
			n.label,
			n.normalized_label
		FROM edges e
		JOIN nodes n ON n.id = %s
		WHERE %s
		ORDER BY e.relation, n.kind, n.label
	`, selectNode, where), nodeID)
	if err != nil {
		return nil, fmt.Errorf("querying node edges: %w", err)
	}
	defer rows.Close()

	var out []EdgeDetail
	for rows.Next() {
		var edge EdgeDetail
		if err := rows.Scan(
			&edge.Relation,
			&edge.Origin,
			&edge.Confidence,
			&edge.EvidenceMemoryID,
			&edge.Node.ID,
			&edge.Node.Kind,
			&edge.Node.Label,
			&edge.Node.NormalizedLabel,
		); err != nil {
			return nil, fmt.Errorf("scanning node edge: %w", err)
		}
		out = append(out, edge)
	}
	return out, rows.Err()
}

func (s *Store) CountConceptNodesMissingEmbeddings() (int, error) {
	var n int
	err := s.db.QueryRow(`
		SELECT COUNT(*)
		FROM nodes n
		LEFT JOIN nodes_vec v ON v.id = n.id
		WHERE n.kind = ? AND v.id IS NULL
	`, NodeKindConcept).Scan(&n)
	return n, err
}

func (s *Store) CountMemoriesMissingNodes() (int, error) {
	var n int
	err := s.db.QueryRow("SELECT COUNT(*) FROM memories WHERE node_id IS NULL OR node_id = ''").Scan(&n)
	return n, err
}

// BackfillMemoryNodesBatch creates graph nodes for at most limit existing memories.
func (s *Store) BackfillMemoryNodesBatch(limit int) (int, error) {
	if limit <= 0 {
		limit = 100
	}

	tx, err := s.db.Begin()
	if err != nil {
		return 0, fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	rows, err := tx.Query(`
		SELECT id, name
		FROM memories
		WHERE node_id IS NULL OR node_id = ''
		ORDER BY created_at ASC
		LIMIT ?
	`, limit)
	if err != nil {
		return 0, fmt.Errorf("listing memories without nodes: %w", err)
	}
	var memories []Memory
	for rows.Next() {
		var m Memory
		if err := rows.Scan(&m.ID, &m.Name); err != nil {
			rows.Close()
			return 0, fmt.Errorf("scanning memory without node: %w", err)
		}
		memories = append(memories, m)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return 0, err
	}

	for _, m := range memories {
		if _, err := ensureMemoryNodeTx(tx, m.ID, m.Name); err != nil {
			return 0, err
		}
	}
	if err := tx.Commit(); err != nil {
		return 0, fmt.Errorf("committing graph backfill: %w", err)
	}
	return len(memories), nil
}

// RefreshHarvestedConcepts rebuilds deterministic concept/facet links from
// existing memory names and sources. It only inserts nodes and edges; cleanup and
// merge tools come later.
func (s *Store) RefreshHarvestedConcepts() error {
	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	if err := refreshHarvestedConceptsTx(tx); err != nil {
		return err
	}
	return tx.Commit()
}

func refreshHarvestedConceptsTx(tx *sql.Tx) error {
	memories, err := listGraphHarvestMemoriesTx(tx)
	if err != nil {
		return err
	}
	if len(memories) == 0 {
		return nil
	}

	seqsByMemory := make(map[string][][]string)
	leadingCounts := map[string]int{}
	rootLabels := map[string]string{}

	for _, m := range memories {
		seqs := harvestSequences(m)
		seqsByMemory[m.ID] = seqs
		seenLeading := map[string]struct{}{}
		for _, seq := range seqs {
			if len(seq) == 0 {
				continue
			}
			lead := seq[0]
			if _, seen := seenLeading[lead]; !seen {
				leadingCounts[lead]++
				seenLeading[lead] = struct{}{}
			}
		}
		for _, label := range pathRootLabels(m.Name, m.Source) {
			rootLabels[NormalizeLabel(label)] = label
		}
	}

	for lead, count := range leadingCounts {
		if count >= defaultFacetThreshold {
			rootLabels[lead] = labelFromNormalized(lead)
		}
	}

	prefixCounts := map[string]int{}
	for _, seqs := range seqsByMemory {
		seenPrefixes := map[string]struct{}{}
		for _, seq := range seqs {
			for _, suffix := range rootedSuffixes(seq, rootLabels) {
				for n := 1; n <= len(suffix); n++ {
					key := strings.Join(suffix[:n], " ")
					if _, seen := seenPrefixes[key]; !seen {
						prefixCounts[key]++
						seenPrefixes[key] = struct{}{}
					}
				}
			}
		}
	}

	promoted := map[string]string{}
	for normalized, label := range rootLabels {
		promoted[normalized] = label
	}
	for prefix, count := range prefixCounts {
		if count < defaultFacetThreshold {
			continue
		}
		if _, isRoot := rootLabels[prefix]; isRoot {
			continue
		}
		if hasPromotedParent(prefix, rootLabels) {
			promoted[prefix] = labelFromNormalized(prefix)
		}
	}

	nodeIDs := map[string]string{}
	keys := make([]string, 0, len(promoted))
	for key := range promoted {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		return tokenCount(keys[i]) < tokenCount(keys[j])
	})
	for _, key := range keys {
		nodeID, err := ensureConceptNodeTx(tx, promoted[key])
		if err != nil {
			return err
		}
		if err := ensureAliasTx(tx, nodeID, strings.ReplaceAll(key, " ", "_"), "harvested"); err != nil {
			return err
		}
		if err := ensureAliasTx(tx, nodeID, strings.ReplaceAll(key, " ", "-"), "harvested"); err != nil {
			return err
		}
		nodeIDs[key] = nodeID
		if parent := longestParent(key, nodeIDs); parent != "" {
			if err := ensureEdgeTx(tx, nodeID, RelationPartOf, nodeIDs[parent], ""); err != nil {
				return err
			}
		}
	}

	for _, m := range memories {
		if m.NodeID == "" {
			nodeID, err := ensureMemoryNodeTx(tx, m.ID, m.Name)
			if err != nil {
				return err
			}
			m.NodeID = nodeID
		}
		if err := deleteHarvestedAboutEdgesTx(tx, m.ID, m.NodeID); err != nil {
			return err
		}
		for _, seq := range seqsByMemory[m.ID] {
			key := bestPrefixForSeq(seq, nodeIDs)
			if key == "" {
				continue
			}
			if err := ensureEdgeTx(tx, m.NodeID, RelationAbout, nodeIDs[key], m.ID); err != nil {
				return err
			}
			break
		}
	}
	return nil
}

func deleteHarvestedAboutEdgesTx(tx *sql.Tx, memoryID, memoryNodeID string) error {
	if memoryNodeID == "" {
		return nil
	}
	_, err := tx.Exec(`
		DELETE FROM edges
		WHERE src_node_id = ? AND relation = ? AND evidence_memory_id = ? AND origin = ?
	`, memoryNodeID, RelationAbout, memoryID, "harvest")
	if err != nil {
		return fmt.Errorf("deleting harvested about edges: %w", err)
	}
	return nil
}

func listGraphHarvestMemoriesTx(tx *sql.Tx) ([]Memory, error) {
	rows, err := tx.Query(`
		SELECT id, name, type, source, node_id, created_at, updated_at
		FROM memories
		ORDER BY created_at ASC
	`)
	if err != nil {
		return nil, fmt.Errorf("listing memories for concept harvest: %w", err)
	}
	defer rows.Close()

	var out []Memory
	for rows.Next() {
		var m Memory
		var source, nodeID sql.NullString
		if err := rows.Scan(&m.ID, &m.Name, &m.Type, &source, &nodeID, &m.CreatedAt, &m.UpdatedAt); err != nil {
			return nil, fmt.Errorf("scanning harvest memory: %w", err)
		}
		m.Source = source.String
		m.NodeID = nodeID.String
		out = append(out, m)
	}
	return out, rows.Err()
}

func harvestSequences(m Memory) [][]string {
	var seqs [][]string
	if seq := stripMemoryTypePrefix(splitConceptTokens(m.Name), m.Type); len(seq) > 0 {
		seqs = append(seqs, seq)
	}
	if m.Source != "" && m.Source != m.Name {
		if seq := stripMemoryTypePrefix(splitConceptTokens(m.Source), m.Type); len(seq) > 0 {
			seqs = append(seqs, seq)
		}
	}
	return seqs
}

func splitConceptTokens(s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	base := s
	if strings.ContainsAny(s, `/\`) {
		base = filepath.Base(s)
		base = strings.TrimSuffix(base, filepath.Ext(base))
	}
	var b strings.Builder
	var prevLower bool
	for _, r := range base {
		if unicode.IsUpper(r) && prevLower {
			b.WriteRune(' ')
		}
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			b.WriteRune(unicode.ToLower(r))
			prevLower = unicode.IsLower(r)
		default:
			b.WriteRune(' ')
			prevLower = false
		}
	}
	return strings.Fields(b.String())
}

func stripMemoryTypePrefix(tokens []string, typ string) []string {
	if len(tokens) == 0 {
		return tokens
	}
	if typ != "" && tokens[0] == NormalizeLabel(typ) {
		return tokens[1:]
	}
	switch tokens[0] {
	case "user", "feedback", "project", "reference", "opinion", "idea", "todo", "reminder":
		return tokens[1:]
	default:
		return tokens
	}
}

func pathRootLabels(values ...string) []string {
	var labels []string
	for _, value := range values {
		if value == "" || !strings.ContainsAny(value, `/\`) {
			continue
		}
		dir := filepath.Dir(value)
		if dir == "." || dir == "/" {
			continue
		}
		base := filepath.Base(dir)
		if tokens := splitConceptTokens(base); len(tokens) > 0 {
			labels = append(labels, labelFromTokens(tokens))
		}
	}
	return labels
}

func hasPromotedParent(prefix string, roots map[string]string) bool {
	for root := range roots {
		if prefix != root && strings.HasPrefix(prefix, root+" ") {
			return true
		}
	}
	return false
}

func longestParent(key string, nodeIDs map[string]string) string {
	parts := strings.Fields(key)
	for n := len(parts) - 1; n >= 1; n-- {
		candidate := strings.Join(parts[:n], " ")
		if _, ok := nodeIDs[candidate]; ok {
			return candidate
		}
	}
	return ""
}

func longestPrefix(tokens []string, nodeIDs map[string]string) string {
	for n := len(tokens); n >= 1; n-- {
		candidate := strings.Join(tokens[:n], " ")
		if _, ok := nodeIDs[candidate]; ok {
			return candidate
		}
	}
	return ""
}

func rootedSuffixes(tokens []string, roots map[string]string) [][]string {
	var out [][]string
	for i := range tokens {
		suffix := tokens[i:]
		for root := range roots {
			rootTokens := strings.Fields(root)
			if len(rootTokens) > len(suffix) {
				continue
			}
			if equalTokens(suffix[:len(rootTokens)], rootTokens) {
				out = append(out, suffix)
				break
			}
		}
	}
	return out
}

func bestPrefixForSeq(tokens []string, nodeIDs map[string]string) string {
	best := ""
	for i := range tokens {
		if candidate := longestPrefix(tokens[i:], nodeIDs); tokenCount(candidate) > tokenCount(best) {
			best = candidate
		}
	}
	return best
}

func equalTokens(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func containsTokenSubsequence(tokens, candidate []string) bool {
	if len(candidate) == 0 || len(candidate) > len(tokens) {
		return false
	}
	idx := 0
	for _, token := range tokens {
		if token == candidate[idx] {
			idx++
			if idx == len(candidate) {
				return true
			}
		}
	}
	return false
}

func conceptQueryScore(queryTokens []string, concept Node) float64 {
	return 3 * tokenOverlapScore(queryTokens, strings.Fields(concept.NormalizedLabel))
}

func bestMemoryQueryScore(queryTokens []string, memories []Memory) float64 {
	best := 0.0
	for _, memory := range memories {
		if score := memoryQueryScore(queryTokens, memory); score > best {
			best = score
		}
	}
	return best
}

func memoryQueryScore(queryTokens []string, memory Memory) float64 {
	var memoryTokens []string
	memoryTokens = append(memoryTokens, splitConceptTokens(memory.Name)...)
	memoryTokens = append(memoryTokens, splitConceptTokens(memory.Description)...)
	memoryTokens = append(memoryTokens, splitConceptTokens(memory.Source)...)
	return tokenOverlapScore(queryTokens, memoryTokens)
}

func tokenOverlapScore(queryTokens, targetTokens []string) float64 {
	if len(queryTokens) == 0 || len(targetTokens) == 0 {
		return 0
	}
	target := make(map[string]struct{}, len(targetTokens))
	for _, token := range targetTokens {
		target[token] = struct{}{}
	}
	score := 0.0
	seen := map[string]struct{}{}
	for _, token := range queryTokens {
		if _, ok := seen[token]; ok {
			continue
		}
		seen[token] = struct{}{}
		if _, ok := target[token]; ok {
			score++
		}
	}
	return score
}

func tokenCount(s string) int {
	return len(strings.Fields(s))
}

func labelFromNormalized(s string) string {
	return labelFromTokens(strings.Fields(s))
}

func labelFromTokens(tokens []string) string {
	words := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if token == "" {
			continue
		}
		words = append(words, strings.ToUpper(token[:1])+token[1:])
	}
	return strings.Join(words, " ")
}

func (s *Store) EnqueueGraphBackfillIfNeeded() error {
	missing, err := s.CountMemoriesMissingNodes()
	if err != nil {
		return err
	}
	if missing == 0 {
		return nil
	}

	var existing int
	err = s.db.QueryRow(`
		SELECT COUNT(*)
		FROM jobs
		WHERE type = ? AND status IN (?, ?)
	`, JobTypeGraphBackfill, JobStatusQueued, JobStatusProcessing).Scan(&existing)
	if err != nil {
		return fmt.Errorf("checking graph backfill jobs: %w", err)
	}
	if existing > 0 {
		return nil
	}

	return s.CreateJob(uuid.New().String(), JobTypeGraphBackfill, `{"batch_size":100}`)
}

func (s *Store) EnqueueGraphHarvestIfNeeded() error {
	missing, err := s.CountConceptNodesMissingEmbeddings()
	if err != nil {
		return err
	}
	if missing == 0 {
		return nil
	}

	var existing int
	err = s.db.QueryRow(`
		SELECT COUNT(*)
		FROM jobs
		WHERE type = ? AND status IN (?, ?)
	`, JobTypeGraphHarvest, JobStatusQueued, JobStatusProcessing).Scan(&existing)
	if err != nil {
		return fmt.Errorf("checking graph harvest jobs: %w", err)
	}
	if existing > 0 {
		return nil
	}

	return s.CreateJob(uuid.New().String(), JobTypeGraphHarvest, `{}`)
}

func enqueueGraphHarvestTx(tx *sql.Tx) error {
	var existing int
	err := tx.QueryRow(`
		SELECT COUNT(*)
		FROM jobs
		WHERE type = ? AND status IN (?, ?)
	`, JobTypeGraphHarvest, JobStatusQueued, JobStatusProcessing).Scan(&existing)
	if err != nil {
		return fmt.Errorf("checking graph harvest jobs: %w", err)
	}
	if existing > 0 {
		return nil
	}

	_, err = tx.Exec(`
		INSERT INTO jobs (id, type, params)
		VALUES (?, ?, ?)
	`, uuid.New().String(), JobTypeGraphHarvest, `{}`)
	if err != nil {
		return fmt.Errorf("creating graph harvest job: %w", err)
	}
	return nil
}

// RecallConcept returns grouped graph results when query exactly resolves to a concept.
func (s *Store) RecallConcept(query string, limit int, filter MemoryFilter) (*ConceptRecall, error) {
	if limit <= 0 {
		limit = 5
	}
	concept, err := s.findConcept(query)
	if err != nil {
		return nil, err
	}
	if concept == nil {
		return nil, nil
	}

	children, err := s.childConcepts(concept.ID)
	if err != nil {
		return nil, err
	}
	recall := &ConceptRecall{Concept: *concept}
	seen := map[string]struct{}{}
	for _, child := range children {
		memories, err := s.memoriesForConcept(child.ID, limit, filter)
		if err != nil {
			return nil, err
		}
		if len(memories) == 0 {
			continue
		}
		for _, m := range memories {
			seen[m.ID] = struct{}{}
		}
		recall.Groups = append(recall.Groups, ConceptRecallGroup{
			Concept:  child,
			Memories: memories,
		})
	}

	direct, err := s.memoriesForConcept(concept.ID, limit, filter)
	if err != nil {
		return nil, err
	}
	for _, m := range direct {
		if _, ok := seen[m.ID]; ok {
			continue
		}
		recall.Direct = append(recall.Direct, m)
	}
	return recall, nil
}

// RecallAutomatic follows the strongest graph branch for a query and returns
// inspectable metadata. It is fallback-safe: ambiguous or weak gradients return a
// stop reason instead of pretending to have found a precise memory.
func (s *Store) RecallAutomatic(query string, queryEmbedding []float32, limit int, filter MemoryFilter) (*AutomaticRecall, error) {
	if limit <= 0 {
		limit = 5
	}
	queryTokens := splitConceptTokens(query)
	if len(queryTokens) == 0 {
		return nil, nil
	}

	vantage, matchedBy, err := s.resolveAutomaticVantage(query, queryTokens, queryEmbedding)
	if err != nil || vantage == nil {
		return nil, err
	}

	recall := &AutomaticRecall{
		Vantage:   *vantage,
		MatchedBy: matchedBy,
		Path: []AutomaticRecallPathItem{{
			Concept: *vantage,
			Reason:  matchedBy,
		}},
	}

	current := *vantage
	const (
		maxDepth  = 3
		minScore  = 1.0
		minMargin = 1.0
	)
	for depth := 0; depth < maxDepth; depth++ {
		candidates, err := s.automaticChildCandidates(current, queryTokens, limit, filter)
		if err != nil {
			return nil, err
		}
		if depth == 0 {
			recall.Candidates = candidates
		}
		if len(candidates) == 0 {
			memories, err := s.rankedMemoriesForConcept(current.ID, queryTokens, limit, filter)
			if err != nil {
				return nil, err
			}
			recall.Memories = memories
			recall.Converged = len(memories) > 0
			recall.StopReason = "leaf"
			if !recall.Converged {
				recall.StopReason = "no_memories"
			}
			return recall, nil
		}

		best := candidates[0]
		if best.Score < minScore {
			recall.StopReason = "no_signal"
			return recall, nil
		}
		if len(candidates) > 1 && best.Score-candidates[1].Score < minMargin {
			recall.StopReason = "flat_gradient"
			return recall, nil
		}

		current = best.Concept
		recall.Path = append(recall.Path, AutomaticRecallPathItem{
			Concept: current,
			Score:   best.Score,
			Reason:  "strongest_graph_branch",
		})
	}

	memories, err := s.rankedMemoriesForConcept(current.ID, queryTokens, limit, filter)
	if err != nil {
		return nil, err
	}
	recall.Memories = memories
	recall.Converged = len(memories) > 0
	recall.StopReason = "max_depth"
	if recall.Converged {
		recall.StopReason = "converged"
	}
	return recall, nil
}

func (s *Store) resolveAutomaticVantage(query string, queryTokens []string, queryEmbedding []float32) (*Node, string, error) {
	concept, err := s.findConcept(query)
	if err != nil || concept != nil {
		return concept, "exact", err
	}
	concept, matchedBy, err := s.findConceptContainedInQuery(queryTokens)
	if err != nil || concept != nil {
		return concept, matchedBy, err
	}
	return s.findFuzzyConcept(queryEmbedding)
}

func (s *Store) findConcept(query string) (*Node, error) {
	normalized := NormalizeLabel(query)
	if normalized == "" {
		return nil, nil
	}

	node, err := s.queryNode(`
		SELECT id, kind, label, normalized_label
		FROM nodes
		WHERE kind = ? AND normalized_label = ?
	`, NodeKindConcept, normalized)
	if err != nil || node != nil {
		return node, err
	}

	return s.queryNode(`
		SELECT n.id, n.kind, n.label, n.normalized_label
		FROM node_aliases a
		JOIN nodes n ON a.node_id = n.id
		WHERE n.kind = ? AND a.normalized_alias = ?
	`, NodeKindConcept, normalized)
}

func (s *Store) findConceptContainedInQuery(queryTokens []string) (*Node, string, error) {
	rows, err := s.db.Query(`
		SELECT id, kind, label, normalized_label
		FROM nodes
		WHERE kind = ?
		ORDER BY label
	`, NodeKindConcept)
	if err != nil {
		return nil, "", fmt.Errorf("querying concept candidates: %w", err)
	}
	defer rows.Close()

	var best *Node
	bestSize := 0
	for rows.Next() {
		var node Node
		if err := rows.Scan(&node.ID, &node.Kind, &node.Label, &node.NormalizedLabel); err != nil {
			return nil, "", fmt.Errorf("scanning concept candidate: %w", err)
		}
		tokens := strings.Fields(node.NormalizedLabel)
		if len(tokens) <= bestSize || !containsTokenSubsequence(queryTokens, tokens) {
			continue
		}
		best = &node
		bestSize = len(tokens)
	}
	if err := rows.Err(); err != nil {
		return nil, "", err
	}
	if best != nil {
		return best, "contained_label", nil
	}

	aliasRows, err := s.db.Query(`
		SELECT n.id, n.kind, n.label, n.normalized_label, a.normalized_alias
		FROM node_aliases a
		JOIN nodes n ON a.node_id = n.id
		WHERE n.kind = ?
		ORDER BY n.label
	`, NodeKindConcept)
	if err != nil {
		return nil, "", fmt.Errorf("querying alias candidates: %w", err)
	}
	defer aliasRows.Close()

	bestSize = 0
	for aliasRows.Next() {
		var node Node
		var alias string
		if err := aliasRows.Scan(&node.ID, &node.Kind, &node.Label, &node.NormalizedLabel, &alias); err != nil {
			return nil, "", fmt.Errorf("scanning alias candidate: %w", err)
		}
		tokens := strings.Fields(alias)
		if len(tokens) <= bestSize || !containsTokenSubsequence(queryTokens, tokens) {
			continue
		}
		best = &node
		bestSize = len(tokens)
	}
	if err := aliasRows.Err(); err != nil {
		return nil, "", err
	}
	if best != nil {
		return best, "contained_alias", nil
	}
	return nil, "", nil
}

func (s *Store) findFuzzyConcept(queryEmbedding []float32) (*Node, string, error) {
	if len(queryEmbedding) == 0 {
		return nil, "", nil
	}
	results, err := s.SearchConceptNodes(queryEmbedding, 2)
	if err != nil || len(results) == 0 {
		return nil, "", err
	}

	const (
		maxDistance = 0.45
		minMargin   = 0.05
	)
	best := results[0]
	if best.Distance > maxDistance {
		return nil, "", nil
	}
	if len(results) > 1 && results[1].Distance-best.Distance < minMargin {
		return nil, "", nil
	}
	return &best.Node, "fuzzy_vector", nil
}

func (s *Store) queryNode(query string, args ...any) (*Node, error) {
	var node Node
	err := s.db.QueryRow(query, args...).Scan(
		&node.ID,
		&node.Kind,
		&node.Label,
		&node.NormalizedLabel,
	)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("querying graph node: %w", err)
	}
	return &node, nil
}

func (s *Store) automaticChildCandidates(parent Node, queryTokens []string, limit int, filter MemoryFilter) ([]AutomaticRecallCandidate, error) {
	children, err := s.childConcepts(parent.ID)
	if err != nil {
		return nil, err
	}
	candidates := make([]AutomaticRecallCandidate, 0, len(children))
	for _, child := range children {
		memories, err := s.memoriesForConcept(child.ID, limit, filter)
		if err != nil {
			return nil, err
		}
		score := conceptQueryScore(queryTokens, child) + bestMemoryQueryScore(queryTokens, memories)
		candidates = append(candidates, AutomaticRecallCandidate{
			Concept:     child,
			Score:       score,
			MemoryCount: len(memories),
		})
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].Score == candidates[j].Score {
			return candidates[i].Concept.Label < candidates[j].Concept.Label
		}
		return candidates[i].Score > candidates[j].Score
	})
	if len(candidates) > 5 {
		candidates = candidates[:5]
	}
	return candidates, nil
}

func (s *Store) rankedMemoriesForConcept(conceptID string, queryTokens []string, limit int, filter MemoryFilter) ([]Memory, error) {
	memories, err := s.memoriesForConcept(conceptID, 0, filter)
	if err != nil {
		return nil, err
	}
	sort.SliceStable(memories, func(i, j int) bool {
		left := memoryQueryScore(queryTokens, memories[i])
		right := memoryQueryScore(queryTokens, memories[j])
		if left == right {
			return memories[i].UpdatedAt.After(memories[j].UpdatedAt)
		}
		return left > right
	})
	if limit > 0 && len(memories) > limit {
		memories = memories[:limit]
	}
	return memories, nil
}

func (s *Store) childConcepts(parentID string) ([]Node, error) {
	rows, err := s.db.Query(`
		SELECT n.id, n.kind, n.label, n.normalized_label
		FROM edges e
		JOIN nodes n ON e.src_node_id = n.id
		WHERE e.relation = ? AND e.dst_node_id = ? AND n.kind = ?
		ORDER BY n.label
	`, RelationPartOf, parentID, NodeKindConcept)
	if err != nil {
		return nil, fmt.Errorf("querying child concepts: %w", err)
	}
	defer rows.Close()

	var out []Node
	for rows.Next() {
		var node Node
		if err := rows.Scan(&node.ID, &node.Kind, &node.Label, &node.NormalizedLabel); err != nil {
			return nil, fmt.Errorf("scanning child concept: %w", err)
		}
		out = append(out, node)
	}
	return out, rows.Err()
}

func (s *Store) memoriesForConcept(conceptID string, limit int, filter MemoryFilter) ([]Memory, error) {
	query := `
		SELECT m.id, m.name, m.type, m.description, m.body, m.agent, m.source, m.checksum, m.node_id, m.created_at, m.updated_at
		FROM edges e
		JOIN memories m ON e.src_node_id = m.node_id
		WHERE e.relation = ? AND e.dst_node_id = ?`
	args := []any{RelationAbout, conceptID}
	if !filter.IsEmpty() {
		clause, fargs := filter.where("m.")
		query += " AND " + clause
		args = append(args, fargs...)
	}
	query += " ORDER BY m.updated_at DESC"
	if limit > 0 {
		query += " LIMIT ?"
		args = append(args, limit)
	}

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("querying concept memories: %w", err)
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
