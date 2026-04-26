// Package store provides SQLite-backed memory and job storage.
package store

import (
	"database/sql"
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

// Job represents an async indexing job.
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

const (
	JobStatusQueued     = "queued"
	JobStatusProcessing = "processing"
	JobStatusCompleted  = "completed"
	JobStatusFailed     = "failed"
)

const (
	JobTypeIndexFile = "index_file"
	JobTypeIndexDir  = "index_directory"
)

// Store manages memory storage, vector search, and the indexing job queue.
type Store struct {
	db         *sql.DB
	dimensions int
}

// New creates a new Store with the given database path and embedding dimensions.
// New creates a new Store. journalMode is the SQLite journal_mode PRAGMA;
// empty defaults to "DELETE" (rollback journal — safe to use under cloud
// sync). Set "WAL" explicitly for local-only DBs that benefit from
// read-during-write concurrency.
func New(dbPath string, dimensions int, journalMode string) (*Store, error) {
	dir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("creating database directory: %w", err)
	}

	if journalMode == "" {
		journalMode = "DELETE"
	}
	// busy_timeout makes contended writes wait for the lock instead of
	// failing with SQLITE_BUSY.
	dsn := fmt.Sprintf("%s?_journal_mode=%s&_busy_timeout=5000&_foreign_keys=on", dbPath, journalMode)
	db, err := sql.Open("sqlite3", dsn)
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
	if err := s.initMemorySchema(); err != nil {
		return err
	}
	return s.initJobSchema()
}

func (s *Store) initJobSchema() error {
	_, err := s.db.Exec(`
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
	return nil
}

// CreateJob creates a new job in the queue.
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

// CreateJobWithParent creates a new job linked to a parent job.
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

// GetJob retrieves a job by ID. Returns nil, nil if not found.
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

// WaitForJob blocks until the job reaches a terminal state or the timeout elapses.
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
		if job.Status == JobStatusCompleted || job.Status == JobStatusFailed {
			return job, nil
		}
		time.Sleep(pollInterval)
	}
	return s.GetJob(id)
}

// ListJobs returns jobs, optionally filtered by status.
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

// UpdateJobStatus updates the status of a job.
func (s *Store) UpdateJobStatus(id, status string) error {
	_, err := s.db.Exec(
		"UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
		status, id,
	)
	return err
}

// UpdateJobProgress updates the progress of a job.
func (s *Store) UpdateJobProgress(id string, progress, total int) error {
	_, err := s.db.Exec(
		"UPDATE jobs SET progress = ?, total = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
		progress, total, id,
	)
	return err
}

// UpdateJobResult marks the job completed with a serialized result.
func (s *Store) UpdateJobResult(id, result string) error {
	_, err := s.db.Exec(
		"UPDATE jobs SET result = ?, status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
		result, JobStatusCompleted, id,
	)
	return err
}

// UpdateJobError marks a job as failed with an error message.
func (s *Store) UpdateJobError(id, errMsg string) error {
	_, err := s.db.Exec(
		"UPDATE jobs SET error = ?, status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
		errMsg, JobStatusFailed, id,
	)
	return err
}

// ChildJobStats contains aggregated statistics for child jobs of a parent job.
type ChildJobStats struct {
	Total      int `json:"total"`
	Queued     int `json:"queued"`
	Processing int `json:"processing"`
	Completed  int `json:"completed"`
	Failed     int `json:"failed"`
}

// GetChildJobStats returns aggregated statistics for child jobs of a parent.
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

// GetNextPendingJob retrieves and claims the next queued job atomically.
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

// DeleteJobs removes jobs by status, or all jobs if status is "all".
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

// Close closes the database connection.
func (s *Store) Close() error {
	return s.db.Close()
}
