// Package queue
package queue

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/srfrog/goldie-mcp/internal/goldie"
	"github.com/srfrog/goldie-mcp/internal/store"
)

// IndexFileParams represents parameters for an index_file job
type IndexFileParams struct {
	Path string `json:"path"`
}

// IndexDirParams represents parameters for an index_directory job
type IndexDirParams struct {
	Directory string `json:"directory"`
	Pattern   string `json:"pattern"`
	Recursive bool   `json:"recursive"`
}

// Queue manages background job processing
type Queue struct {
	store   *store.Store
	goldie  *goldie.Goldie
	logger  *log.Logger
	stop    chan struct{}
	wg      sync.WaitGroup
	polling time.Duration
}

// New creates a new Queue
func New(st *store.Store, g *goldie.Goldie, logger *log.Logger) *Queue {
	// Use a discard logger if none provided
	if logger == nil {
		logger = log.New(io.Discard, "", 0)
	}
	return &Queue{
		store:   st,
		goldie:  g,
		logger:  logger,
		stop:    make(chan struct{}),
		polling: 500 * time.Millisecond,
	}
}

// Start begins the background worker
func (q *Queue) Start() {
	q.wg.Add(1)
	go q.worker()
}

// Stop gracefully stops the queue worker
func (q *Queue) Stop() {
	close(q.stop)
	q.wg.Wait()
}

// EnqueueIndexFile creates a job to index a file
func (q *Queue) EnqueueIndexFile(path string) (string, error) {
	id := uuid.New().String()

	params, err := json.Marshal(IndexFileParams{Path: path})
	if err != nil {
		return "", fmt.Errorf("marshaling params: %w", err)
	}

	if err := q.store.CreateJob(id, store.JobTypeIndexFile, string(params)); err != nil {
		return "", fmt.Errorf("creating job: %w", err)
	}

	return id, nil
}

// EnqueueIndexFileWithParent creates a job to index a file as a child of a parent job
func (q *Queue) EnqueueIndexFileWithParent(path string, parentID string) (string, error) {
	id := uuid.New().String()

	params, err := json.Marshal(IndexFileParams{Path: path})
	if err != nil {
		return "", fmt.Errorf("marshaling params: %w", err)
	}

	if err := q.store.CreateJobWithParent(id, store.JobTypeIndexFile, string(params), parentID); err != nil {
		return "", fmt.Errorf("creating job: %w", err)
	}

	return id, nil
}

// EnqueueIndexDirectory creates a job to index a directory
func (q *Queue) EnqueueIndexDirectory(directory, pattern string, recursive bool) (string, error) {
	id := uuid.New().String()

	params, err := json.Marshal(IndexDirParams{
		Directory: directory,
		Pattern:   pattern,
		Recursive: recursive,
	})
	if err != nil {
		return "", fmt.Errorf("marshaling params: %w", err)
	}

	if err := q.store.CreateJob(id, store.JobTypeIndexDir, string(params)); err != nil {
		return "", fmt.Errorf("creating job: %w", err)
	}

	return id, nil
}

// worker is the background goroutine that processes jobs
func (q *Queue) worker() {
	defer q.wg.Done()
	defer func() {
		if r := recover(); r != nil {
			q.logger.Printf("Queue worker panic recovered: %v", r)
			// Restart the worker after a panic
			q.wg.Add(1)
			go q.worker()
		}
	}()

	ticker := time.NewTicker(q.polling)
	defer ticker.Stop()

	for {
		select {
		case <-q.stop:
			return
		case <-ticker.C:
			q.processNextJob()
		}
	}
}

// processNextJob fetches and processes the next pending job
func (q *Queue) processNextJob() {
	job, err := q.store.GetNextPendingJob()
	if err != nil {
		q.logger.Printf("Error getting next job: %v", err)
		return
	}
	if job == nil {
		return // No pending jobs
	}

	q.logger.Printf("Processing job %s (type: %s)", job.ID, job.Type)

	switch job.Type {
	case store.JobTypeIndexFile:
		q.processIndexFile(job)
	case store.JobTypeIndexDir:
		q.processIndexDirectory(job)
	default:
		q.logger.Printf("Unknown job type: %s", job.Type)
		q.store.UpdateJobError(job.ID, fmt.Sprintf("unknown job type: %s", job.Type))
	}
}

// processIndexFile handles an index_file job
func (q *Queue) processIndexFile(job *store.Job) {
	q.logger.Printf("Job %s: processIndexFile started", job.ID)

	var params IndexFileParams
	if err := json.Unmarshal([]byte(job.Params), &params); err != nil {
		q.logger.Printf("Job %s: invalid params: %v", job.ID, err)
		q.store.UpdateJobError(job.ID, fmt.Sprintf("invalid params: %v", err))
		return
	}
	q.logger.Printf("Job %s: params parsed, path=%s", job.ID, params.Path)

	// Update progress
	if err := q.store.UpdateJobProgress(job.ID, 0, 1); err != nil {
		q.logger.Printf("Job %s: failed to update progress: %v", job.ID, err)
	}
	q.logger.Printf("Job %s: progress updated, calling IndexFile", job.ID)

	// Index the file
	result, err := q.goldie.IndexFile(params.Path)
	if err != nil {
		q.logger.Printf("Job %s: indexing failed: %v", job.ID, err)
		q.store.UpdateJobError(job.ID, fmt.Sprintf("indexing failed: %v", err))
		return
	}
	q.logger.Printf("Job %s: IndexFile returned, chunks=%d", job.ID, result.ChunkCount)

	// Mark complete with result
	resultJSON, err := json.Marshal(map[string]any{
		"id":          result.ID,
		"chunk_count": result.ChunkCount,
		"path":        params.Path,
	})
	if err != nil {
		q.logger.Printf("Job %s: failed to marshal result: %v", job.ID, err)
		q.store.UpdateJobError(job.ID, fmt.Sprintf("failed to marshal result: %v", err))
		return
	}

	q.store.UpdateJobProgress(job.ID, 1, 1)
	if err := q.store.UpdateJobResult(job.ID, string(resultJSON)); err != nil {
		q.logger.Printf("Job %s: failed to update result: %v", job.ID, err)
	}

	q.logger.Printf("Job %s: completed - indexed %s (%d chunks)", job.ID, params.Path, result.ChunkCount)
}

// processIndexDirectory handles an index_directory job
func (q *Queue) processIndexDirectory(job *store.Job) {
	q.logger.Printf("Job %s: processIndexDirectory started", job.ID)

	var params IndexDirParams
	if err := json.Unmarshal([]byte(job.Params), &params); err != nil {
		q.logger.Printf("Job %s: invalid params: %v", job.ID, err)
		q.store.UpdateJobError(job.ID, fmt.Sprintf("invalid params: %v", err))
		return
	}
	q.logger.Printf("Job %s: scanning dir=%s pattern=%s recursive=%v", job.ID, params.Directory, params.Pattern, params.Recursive)

	// First, scan the directory to get the list of files
	scanResult, err := q.goldie.ScanDirectory(params.Directory, params.Pattern, params.Recursive)
	if err != nil {
		q.logger.Printf("Job %s: scanning failed: %v", job.ID, err)
		q.store.UpdateJobError(job.ID, fmt.Sprintf("scanning failed: %v", err))
		return
	}

	fileCount := len(scanResult.Files)
	q.logger.Printf("Job %s: found %d files, creating child jobs", job.ID, fileCount)

	// Update progress to show total files found
	q.store.UpdateJobProgress(job.ID, 0, fileCount)

	// Create a child job for each file
	childJobIDs := make([]string, 0, fileCount)
	for _, file := range scanResult.Files {
		childID, err := q.EnqueueIndexFileWithParent(file, job.ID)
		if err != nil {
			q.logger.Printf("Job %s: failed to create child job for %s: %v", job.ID, file, err)
			continue
		}
		childJobIDs = append(childJobIDs, childID)
		q.logger.Printf("Job %s: created child job %s for %s", job.ID, childID, file)
	}

	// Mark parent job complete with metadata about child jobs
	resultJSON, err := json.Marshal(map[string]any{
		"file_count":    fileCount,
		"child_job_ids": childJobIDs,
		"directory":     params.Directory,
		"pattern":       params.Pattern,
		"recursive":     params.Recursive,
	})
	if err != nil {
		q.logger.Printf("Job %s: failed to marshal result: %v", job.ID, err)
		q.store.UpdateJobError(job.ID, fmt.Sprintf("failed to marshal result: %v", err))
		return
	}

	if err := q.store.UpdateJobResult(job.ID, string(resultJSON)); err != nil {
		q.logger.Printf("Job %s: failed to update result: %v", job.ID, err)
	}

	q.logger.Printf("Job %s: completed - created %d child jobs for indexing", job.ID, len(childJobIDs))
}
