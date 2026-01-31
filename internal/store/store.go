// Package store
package store

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	sqlite_vec "github.com/asg017/sqlite-vec-go-bindings/cgo"
	_ "github.com/mattn/go-sqlite3"
)

func init() {
	sqlite_vec.Auto()
}

// Document represents a stored document
type Document struct {
	ID        string            `json:"id"`
	Content   string            `json:"content"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	CreatedAt time.Time         `json:"created_at"`
}

// SearchResult represents a search result with similarity score
type SearchResult struct {
	Document Document `json:"document"`
	Score    float32  `json:"score"`
	Distance float32  `json:"distance"`
}

// Job represents an async indexing job
type Job struct {
	ID        string    `json:"id"`
	Type      string    `json:"type"`
	Status    string    `json:"status"`
	Params    string    `json:"params"`
	Result    string    `json:"result,omitempty"`
	Error     string    `json:"error,omitempty"`
	Progress  int       `json:"progress"`
	Total     int       `json:"total"`
	ParentID  string    `json:"parent_id,omitempty"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// Job status constants
const (
	JobStatusQueued     = "queued"
	JobStatusProcessing = "processing"
	JobStatusCompleted  = "completed"
	JobStatusFailed     = "failed"
)

// Job type constants
const (
	JobTypeIndexFile = "index_file"
	JobTypeIndexDir  = "index_directory"
)

// Store manages document storage and vector search
type Store struct {
	db         *sql.DB
	dimensions int
}

// New creates a new Store with the given database path
func New(dbPath string, dimensions int) (*Store, error) {
	// Ensure directory exists
	dir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("creating database directory: %w", err)
	}

	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("opening database: %w", err)
	}

	store := &Store{
		db:         db,
		dimensions: dimensions,
	}

	if err := store.initSchema(); err != nil {
		db.Close()
		return nil, fmt.Errorf("initializing schema: %w", err)
	}

	return store, nil
}

func (s *Store) initSchema() error {
	// Create documents table
	_, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS documents (
			id TEXT PRIMARY KEY,
			content TEXT NOT NULL,
			metadata TEXT,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		)
	`)
	if err != nil {
		return fmt.Errorf("creating documents table: %w", err)
	}

	// Create vector virtual table
	query := fmt.Sprintf(`
		CREATE VIRTUAL TABLE IF NOT EXISTS documents_vec USING vec0(
			id TEXT PRIMARY KEY,
			embedding FLOAT[%d]
		)
	`, s.dimensions)
	_, err = s.db.Exec(query)
	if err != nil {
		return fmt.Errorf("creating vector table: %w", err)
	}

	// Create jobs table
	_, err = s.db.Exec(`
		CREATE TABLE IF NOT EXISTS jobs (
			id TEXT PRIMARY KEY,
			type TEXT NOT NULL,
			status TEXT DEFAULT 'queued',
			params TEXT NOT NULL,
			result TEXT,
			error TEXT,
			progress INTEGER DEFAULT 0,
			total INTEGER DEFAULT 0,
			parent_id TEXT,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
		)
	`)
	if err != nil {
		return fmt.Errorf("creating jobs table: %w", err)
	}

	// Add parent_id column if it doesn't exist (migration for existing databases)
	// Ignore error if column already exists
	s.db.Exec(`ALTER TABLE jobs ADD COLUMN parent_id TEXT`)

	return nil
}

// AddDocument adds a document with its embedding to the store
func (s *Store) AddDocument(id, content string, metadata map[string]string, embedding []float32) error {
	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	// Serialize metadata
	var metadataJSON []byte
	if metadata != nil {
		metadataJSON, err = json.Marshal(metadata)
		if err != nil {
			return fmt.Errorf("marshaling metadata: %w", err)
		}
	}

	// Insert document
	_, err = tx.Exec(
		"INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?, ?, ?)",
		id, content, string(metadataJSON),
	)
	if err != nil {
		return fmt.Errorf("inserting document: %w", err)
	}

	// Serialize embedding to JSON for sqlite-vec
	embeddingJSON, err := json.Marshal(embedding)
	if err != nil {
		return fmt.Errorf("marshaling embedding: %w", err)
	}

	// Insert vector
	_, err = tx.Exec(
		"INSERT OR REPLACE INTO documents_vec (id, embedding) VALUES (?, ?)",
		id, string(embeddingJSON),
	)
	if err != nil {
		return fmt.Errorf("inserting vector: %w", err)
	}

	return tx.Commit()
}

// Search finds similar documents using vector similarity
func (s *Store) Search(embedding []float32, limit int) ([]SearchResult, error) {
	if limit <= 0 {
		limit = 5
	}

	// Serialize query embedding
	embeddingJSON, err := json.Marshal(embedding)
	if err != nil {
		return nil, fmt.Errorf("marshaling query embedding: %w", err)
	}

	// Query using sqlite-vec KNN syntax (requires k = ? in WHERE clause)
	query := `
		SELECT
			v.id,
			v.distance,
			d.content,
			d.metadata,
			d.created_at
		FROM documents_vec v
		JOIN documents d ON v.id = d.id
		WHERE v.embedding MATCH ? AND k = ?
		ORDER BY v.distance
	`

	rows, err := s.db.Query(query, string(embeddingJSON), limit)
	if err != nil {
		return nil, fmt.Errorf("querying vectors: %w", err)
	}
	defer rows.Close()

	var results []SearchResult
	for rows.Next() {
		var (
			id          string
			distance    float32
			content     string
			metadataStr sql.NullString
			createdAt   time.Time
		)

		if err := rows.Scan(&id, &distance, &content, &metadataStr, &createdAt); err != nil {
			return nil, fmt.Errorf("scanning row: %w", err)
		}

		var metadata map[string]string
		if metadataStr.Valid && metadataStr.String != "" {
			if err := json.Unmarshal([]byte(metadataStr.String), &metadata); err != nil {
				// Ignore metadata parse errors
				metadata = nil
			}
		}

		// Convert distance to similarity score (1 - distance for cosine)
		score := 1 - distance

		results = append(results, SearchResult{
			Document: Document{
				ID:        id,
				Content:   content,
				Metadata:  metadata,
				CreatedAt: createdAt,
			},
			Score:    score,
			Distance: distance,
		})
	}

	return results, rows.Err()
}

// GetDocument retrieves a document by ID
func (s *Store) GetDocument(id string) (*Document, error) {
	var (
		content     string
		metadataStr sql.NullString
		createdAt   time.Time
	)

	err := s.db.QueryRow(
		"SELECT content, metadata, created_at FROM documents WHERE id = ?",
		id,
	).Scan(&content, &metadataStr, &createdAt)

	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("querying document: %w", err)
	}

	var metadata map[string]string
	if metadataStr.Valid && metadataStr.String != "" {
		if err := json.Unmarshal([]byte(metadataStr.String), &metadata); err != nil {
			metadata = nil
		}
	}

	return &Document{
		ID:        id,
		Content:   content,
		Metadata:  metadata,
		CreatedAt: createdAt,
	}, nil
}

// ListDocuments returns all documents (without embeddings)
func (s *Store) ListDocuments() ([]Document, error) {
	rows, err := s.db.Query("SELECT id, content, metadata, created_at FROM documents ORDER BY created_at DESC")
	if err != nil {
		return nil, fmt.Errorf("querying documents: %w", err)
	}
	defer rows.Close()

	var docs []Document
	for rows.Next() {
		var (
			id          string
			content     string
			metadataStr sql.NullString
			createdAt   time.Time
		)

		if err := rows.Scan(&id, &content, &metadataStr, &createdAt); err != nil {
			return nil, fmt.Errorf("scanning row: %w", err)
		}

		var metadata map[string]string
		if metadataStr.Valid && metadataStr.String != "" {
			if err := json.Unmarshal([]byte(metadataStr.String), &metadata); err != nil {
				// Log but continue - corrupted metadata shouldn't break listing
				metadata = nil
			}
		}

		docs = append(docs, Document{
			ID:        id,
			Content:   content,
			Metadata:  metadata,
			CreatedAt: createdAt,
		})
	}

	return docs, rows.Err()
}

// DeleteDocument removes a document from the store
func (s *Store) DeleteDocument(id string) error {
	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	_, err = tx.Exec("DELETE FROM documents WHERE id = ?", id)
	if err != nil {
		return fmt.Errorf("deleting document: %w", err)
	}

	_, err = tx.Exec("DELETE FROM documents_vec WHERE id = ?", id)
	if err != nil {
		return fmt.Errorf("deleting vector: %w", err)
	}

	return tx.Commit()
}

// Count returns the number of documents in the store
func (s *Store) Count() (int, error) {
	var count int
	err := s.db.QueryRow("SELECT COUNT(*) FROM documents").Scan(&count)
	return count, err
}

// CreateJob creates a new job in the queue
func (s *Store) CreateJob(id, jobType, params string) error {
	_, err := s.db.Exec(
		"INSERT INTO jobs (id, type, params) VALUES (?, ?, ?)",
		id, jobType, params,
	)
	if err != nil {
		return fmt.Errorf("creating job: %w", err)
	}
	return nil
}

// GetJob retrieves a job by ID
func (s *Store) GetJob(id string) (*Job, error) {
	var job Job
	var result, errMsg, parentID sql.NullString

	err := s.db.QueryRow(`
		SELECT id, type, status, params, result, error, progress, total, parent_id, created_at, updated_at
		FROM jobs WHERE id = ?
	`, id).Scan(
		&job.ID, &job.Type, &job.Status, &job.Params,
		&result, &errMsg, &job.Progress, &job.Total, &parentID,
		&job.CreatedAt, &job.UpdatedAt,
	)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("querying job: %w", err)
	}

	if result.Valid {
		job.Result = result.String
	}
	if errMsg.Valid {
		job.Error = errMsg.String
	}
	if parentID.Valid {
		job.ParentID = parentID.String
	}

	return &job, nil
}

// WaitForJob waits for a job to reach a terminal state (completed or failed)
func (s *Store) WaitForJob(id string, timeout time.Duration) (*Job, error) {
	deadline := time.Now().Add(timeout)
	pollInterval := 100 * time.Millisecond

	for time.Now().Before(deadline) {
		job, err := s.GetJob(id)
		if err != nil {
			return nil, err
		}
		if job == nil {
			return nil, fmt.Errorf("job not found: %s", id)
		}

		// Check if job is in terminal state
		if job.Status == JobStatusCompleted || job.Status == JobStatusFailed {
			return job, nil
		}

		time.Sleep(pollInterval)
	}

	// Return the last state even if timeout
	return s.GetJob(id)
}

// ListJobs returns jobs, optionally filtered by status
func (s *Store) ListJobs(status string) ([]Job, error) {
	var rows *sql.Rows
	var err error

	if status == "" {
		rows, err = s.db.Query(`
			SELECT id, type, status, params, result, error, progress, total, parent_id, created_at, updated_at
			FROM jobs ORDER BY created_at DESC
		`)
	} else {
		rows, err = s.db.Query(`
			SELECT id, type, status, params, result, error, progress, total, parent_id, created_at, updated_at
			FROM jobs WHERE status = ? ORDER BY created_at DESC
		`, status)
	}
	if err != nil {
		return nil, fmt.Errorf("querying jobs: %w", err)
	}
	defer rows.Close()

	var jobs []Job
	for rows.Next() {
		var job Job
		var result, errMsg, parentID sql.NullString

		if err := rows.Scan(
			&job.ID, &job.Type, &job.Status, &job.Params,
			&result, &errMsg, &job.Progress, &job.Total, &parentID,
			&job.CreatedAt, &job.UpdatedAt,
		); err != nil {
			return nil, fmt.Errorf("scanning job: %w", err)
		}

		if result.Valid {
			job.Result = result.String
		}
		if errMsg.Valid {
			job.Error = errMsg.String
		}
		if parentID.Valid {
			job.ParentID = parentID.String
		}

		jobs = append(jobs, job)
	}

	return jobs, rows.Err()
}

// UpdateJobStatus updates the status of a job
func (s *Store) UpdateJobStatus(id, status string) error {
	_, err := s.db.Exec(
		"UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
		status, id,
	)
	return err
}

// UpdateJobProgress updates the progress of a job
func (s *Store) UpdateJobProgress(id string, progress, total int) error {
	_, err := s.db.Exec(
		"UPDATE jobs SET progress = ?, total = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
		progress, total, id,
	)
	return err
}

// UpdateJobResult updates the result of a completed job
func (s *Store) UpdateJobResult(id, result string) error {
	_, err := s.db.Exec(
		"UPDATE jobs SET result = ?, status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
		result, JobStatusCompleted, id,
	)
	return err
}

// UpdateJobError marks a job as failed with an error message
func (s *Store) UpdateJobError(id, errMsg string) error {
	_, err := s.db.Exec(
		"UPDATE jobs SET error = ?, status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
		errMsg, JobStatusFailed, id,
	)
	return err
}

// CreateJobWithParent creates a new job with a parent ID
func (s *Store) CreateJobWithParent(id, jobType, params, parentID string) error {
	_, err := s.db.Exec(
		"INSERT INTO jobs (id, type, params, parent_id) VALUES (?, ?, ?, ?)",
		id, jobType, params, parentID,
	)
	if err != nil {
		return fmt.Errorf("creating job: %w", err)
	}
	return nil
}

// ChildJobStats contains aggregated statistics for child jobs
type ChildJobStats struct {
	Total      int `json:"total"`
	Queued     int `json:"queued"`
	Processing int `json:"processing"`
	Completed  int `json:"completed"`
	Failed     int `json:"failed"`
}

// GetChildJobStats returns aggregated statistics for child jobs of a parent
func (s *Store) GetChildJobStats(parentID string) (*ChildJobStats, error) {
	rows, err := s.db.Query(`
		SELECT status, COUNT(*) as count
		FROM jobs
		WHERE parent_id = ?
		GROUP BY status
	`, parentID)
	if err != nil {
		return nil, fmt.Errorf("querying child job stats: %w", err)
	}
	defer rows.Close()

	stats := &ChildJobStats{}
	for rows.Next() {
		var status string
		var count int
		if err := rows.Scan(&status, &count); err != nil {
			return nil, fmt.Errorf("scanning child job stats: %w", err)
		}
		stats.Total += count
		switch status {
		case JobStatusQueued:
			stats.Queued = count
		case JobStatusProcessing:
			stats.Processing = count
		case JobStatusCompleted:
			stats.Completed = count
		case JobStatusFailed:
			stats.Failed = count
		}
	}

	return stats, rows.Err()
}

// GetNextPendingJob retrieves and claims the next queued job
func (s *Store) GetNextPendingJob() (*Job, error) {
	tx, err := s.db.Begin()
	if err != nil {
		return nil, fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	var job Job
	var result, errMsg, parentID sql.NullString

	err = tx.QueryRow(`
		SELECT id, type, status, params, result, error, progress, total, parent_id, created_at, updated_at
		FROM jobs WHERE status = ? ORDER BY created_at ASC LIMIT 1
	`, JobStatusQueued).Scan(
		&job.ID, &job.Type, &job.Status, &job.Params,
		&result, &errMsg, &job.Progress, &job.Total, &parentID,
		&job.CreatedAt, &job.UpdatedAt,
	)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("querying pending job: %w", err)
	}

	if result.Valid {
		job.Result = result.String
	}
	if errMsg.Valid {
		job.Error = errMsg.String
	}
	if parentID.Valid {
		job.ParentID = parentID.String
	}

	// Claim the job by setting status to processing
	_, err = tx.Exec(
		"UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
		JobStatusProcessing, job.ID,
	)
	if err != nil {
		return nil, fmt.Errorf("claiming job: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("committing transaction: %w", err)
	}

	job.Status = JobStatusProcessing
	return &job, nil
}

// DeleteJobs removes jobs by status, or all jobs if status is "all"
func (s *Store) DeleteJobs(status string) (int, error) {
	var result sql.Result
	var err error

	if status == "all" {
		result, err = s.db.Exec("DELETE FROM jobs")
	} else {
		result, err = s.db.Exec("DELETE FROM jobs WHERE status = ?", status)
	}
	if err != nil {
		return 0, fmt.Errorf("deleting jobs: %w", err)
	}

	count, _ := result.RowsAffected()
	return int(count), nil
}

// Close closes the database connection
func (s *Store) Close() error {
	return s.db.Close()
}
