package main

import (
	"context"
	"encoding/json"
	"hash/fnv"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/srfrog/goldie-mcp/internal/embedder"
	"github.com/srfrog/goldie-mcp/internal/goldie"
	"github.com/srfrog/goldie-mcp/internal/queue"
	"github.com/srfrog/goldie-mcp/internal/store"
)

// MockEmbedder generates deterministic embeddings for testing
type MockEmbedder struct {
	dimensions int
	delay      time.Duration // simulate processing time
}

var _ embedder.Interface = (*MockEmbedder)(nil)

func NewMockEmbedder(dimensions int, delay time.Duration) *MockEmbedder {
	return &MockEmbedder{
		dimensions: dimensions,
		delay:      delay,
	}
}

func (m *MockEmbedder) Embed(text string) ([]float32, error) {
	if m.delay > 0 {
		time.Sleep(m.delay)
	}
	return m.hashToEmbedding(text), nil
}

func (m *MockEmbedder) EmbedBatch(texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i, text := range texts {
		if m.delay > 0 {
			time.Sleep(m.delay)
		}
		result[i] = m.hashToEmbedding(text)
	}
	return result, nil
}

func (m *MockEmbedder) GetDimensions() int {
	return m.dimensions
}

func (m *MockEmbedder) Warmup() error {
	return nil
}

func (m *MockEmbedder) Close() error {
	return nil
}

// hashToEmbedding creates a deterministic embedding from text hash
func (m *MockEmbedder) hashToEmbedding(text string) []float32 {
	h := fnv.New64a()
	h.Write([]byte(text))
	seed := h.Sum64()

	embedding := make([]float32, m.dimensions)
	for i := range embedding {
		// Simple deterministic pseudo-random based on seed and index
		seed = seed*6364136223846793005 + 1442695040888963407
		embedding[i] = float32(seed%1000) / 1000.0
	}
	return embedding
}

// TestSetup creates a test environment with mock embedder
type TestSetup struct {
	DBPath  string
	Goldie  *goldie.Goldie
	Store   *store.Store
	Queue   *queue.Queue
	TempDir string
}

func NewTestSetup(t *testing.T) *TestSetup {
	t.Helper()

	tempDir, err := os.MkdirTemp("", "goldie-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}

	dbPath := filepath.Join(tempDir, "test.db")
	mockEmb := NewMockEmbedder(384, 10*time.Millisecond)

	cfg := goldie.Config{
		DBPath:   dbPath,
		Embedder: mockEmb,
	}

	r, err := goldie.New(cfg)
	if err != nil {
		os.RemoveAll(tempDir)
		t.Fatalf("failed to create RAG: %v", err)
	}

	st := r.Store()
	q := queue.New(st, r, nil) // nil logger for tests

	return &TestSetup{
		DBPath:  dbPath,
		Goldie:  r,
		Store:   st,
		Queue:   q,
		TempDir: tempDir,
	}
}

func (ts *TestSetup) Cleanup() {
	ts.Queue.Stop()
	ts.Goldie.Close()
	os.RemoveAll(ts.TempDir)
}

// SetupGlobals sets up the global variables used by MCP handlers
func (ts *TestSetup) SetupGlobals() {
	goldieInstance = ts.Goldie
	storeInstance = ts.Store
	queueInstance = ts.Queue
}

// CallTool invokes an MCP tool handler and returns the parsed response
func (ts *TestSetup) CallTool(t *testing.T, toolName string, args map[string]interface{}) map[string]interface{} {
	t.Helper()

	req := mcp.CallToolRequest{}
	req.Params.Name = toolName
	req.Params.Arguments = args

	ctx := context.Background()
	var result *mcp.CallToolResult
	var err error

	switch toolName {
	case "index_file":
		result, err = handleIndexFile(ctx, req)
	case "index_directory":
		result, err = handleIndexDirectory(ctx, req)
	case "index_content":
		result, err = handleIndexContent(ctx, req)
	case "search_index":
		result, err = handleSearch(ctx, req)
	case "job_status":
		result, err = handleJobStatus(ctx, req)
	case "list_jobs":
		result, err = handleListJobs(ctx, req)
	case "clear_queue":
		result, err = handleClearQueue(ctx, req)
	case "count_documents":
		result, err = handleCountDocuments(ctx, req)
	case "delete_document":
		result, err = handleDeleteDocument(ctx, req)
	default:
		t.Fatalf("unknown tool: %s", toolName)
	}

	if err != nil {
		t.Fatalf("tool %s returned error: %v", toolName, err)
	}

	// Extract text content from result
	if len(result.Content) == 0 {
		t.Fatalf("tool %s returned no content", toolName)
	}

	textContent, ok := result.Content[0].(mcp.TextContent)
	if !ok {
		t.Fatalf("tool %s returned non-text content", toolName)
	}

	var response map[string]interface{}
	if err := json.Unmarshal([]byte(textContent.Text), &response); err != nil {
		// If not valid JSON, return a map with the text as "message"
		// This handles plain text responses for empty results
		return map[string]interface{}{
			"message": textContent.Text,
		}
	}

	return response
}

func TestJobQueueBasicFlow(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()

	// Start the queue worker
	ts.Queue.Start()

	// Create a test file
	testFile := filepath.Join(ts.TempDir, "test.txt")
	if err := os.WriteFile(testFile, []byte("This is test content for indexing."), 0o644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	// Enqueue a job
	jobID, err := ts.Queue.EnqueueIndexFile(testFile)
	if err != nil {
		t.Fatalf("failed to enqueue job: %v", err)
	}

	if jobID == "" {
		t.Fatal("expected non-empty job ID")
	}

	// Verify job is queued
	job, err := ts.Store.GetJob(jobID)
	if err != nil {
		t.Fatalf("failed to get job: %v", err)
	}

	if job == nil {
		t.Fatal("job not found")
	}

	// Wait for job to complete (with timeout)
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		job, err = ts.Store.GetJob(jobID)
		if err != nil {
			t.Fatalf("failed to get job: %v", err)
		}
		if job.Status == store.JobStatusCompleted {
			break
		}
		if job.Status == store.JobStatusFailed {
			t.Fatalf("job failed: %s", job.Error)
		}
		time.Sleep(100 * time.Millisecond)
	}

	if job.Status != store.JobStatusCompleted {
		t.Fatalf("job did not complete in time, status: %s", job.Status)
	}

	// Verify result
	var result map[string]interface{}
	if err := json.Unmarshal([]byte(job.Result), &result); err != nil {
		t.Fatalf("failed to parse result: %v", err)
	}

	if result["path"] != testFile {
		t.Errorf("expected path %s, got %v", testFile, result["path"])
	}

	// Verify document was indexed
	count, err := ts.Store.Count()
	if err != nil {
		t.Fatalf("failed to count documents: %v", err)
	}

	if count == 0 {
		t.Error("expected at least one document indexed")
	}
}

func TestJobQueueDirectoryIndexing(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()

	ts.Queue.Start()

	// Create test files
	testDir := filepath.Join(ts.TempDir, "docs")
	if err := os.MkdirAll(testDir, 0o755); err != nil {
		t.Fatalf("failed to create test dir: %v", err)
	}

	files := []string{"file1.txt", "file2.txt", "file3.md"}
	for _, name := range files {
		content := "Content of " + name
		if err := os.WriteFile(filepath.Join(testDir, name), []byte(content), 0o644); err != nil {
			t.Fatalf("failed to create %s: %v", name, err)
		}
	}

	// Enqueue directory indexing job
	jobID, err := ts.Queue.EnqueueIndexDirectory(testDir, "*.txt", false)
	if err != nil {
		t.Fatalf("failed to enqueue job: %v", err)
	}

	// Wait for completion
	deadline := time.Now().Add(10 * time.Second)
	var job *store.Job
	for time.Now().Before(deadline) {
		job, err = ts.Store.GetJob(jobID)
		if err != nil {
			t.Fatalf("failed to get job: %v", err)
		}
		if job.Status == store.JobStatusCompleted || job.Status == store.JobStatusFailed {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	if job.Status != store.JobStatusCompleted {
		t.Fatalf("job did not complete, status: %s, error: %s", job.Status, job.Error)
	}

	// Verify result
	var result map[string]interface{}
	if err := json.Unmarshal([]byte(job.Result), &result); err != nil {
		t.Fatalf("failed to parse result: %v", err)
	}

	fileCount := int(result["file_count"].(float64))
	if fileCount != 2 { // Only .txt files
		t.Errorf("expected 2 files indexed, got %d", fileCount)
	}
}

func TestListJobs(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()

	// Don't start worker - jobs will stay queued

	// Create multiple jobs
	testFile := filepath.Join(ts.TempDir, "test.txt")
	os.WriteFile(testFile, []byte("test"), 0o644)

	_, err := ts.Queue.EnqueueIndexFile(testFile)
	if err != nil {
		t.Fatalf("failed to enqueue job 1: %v", err)
	}

	_, err = ts.Queue.EnqueueIndexFile(testFile)
	if err != nil {
		t.Fatalf("failed to enqueue job 2: %v", err)
	}

	// List all jobs
	jobs, err := ts.Store.ListJobs("")
	if err != nil {
		t.Fatalf("failed to list jobs: %v", err)
	}

	if len(jobs) != 2 {
		t.Errorf("expected 2 jobs, got %d", len(jobs))
	}

	// List only queued jobs
	queuedJobs, err := ts.Store.ListJobs(store.JobStatusQueued)
	if err != nil {
		t.Fatalf("failed to list queued jobs: %v", err)
	}

	if len(queuedJobs) != 2 {
		t.Errorf("expected 2 queued jobs, got %d", len(queuedJobs))
	}
}

func TestClearQueue(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()

	// Create some jobs
	testFile := filepath.Join(ts.TempDir, "test.txt")
	os.WriteFile(testFile, []byte("test"), 0o644)

	for i := 0; i < 3; i++ {
		_, err := ts.Queue.EnqueueIndexFile(testFile)
		if err != nil {
			t.Fatalf("failed to enqueue job: %v", err)
		}
	}

	// Verify jobs exist
	jobs, _ := ts.Store.ListJobs("")
	if len(jobs) != 3 {
		t.Fatalf("expected 3 jobs, got %d", len(jobs))
	}

	// Clear queued jobs
	deleted, err := ts.Store.DeleteJobs(store.JobStatusQueued)
	if err != nil {
		t.Fatalf("failed to delete jobs: %v", err)
	}

	if deleted != 3 {
		t.Errorf("expected 3 deleted, got %d", deleted)
	}

	// Verify queue is empty
	jobs, _ = ts.Store.ListJobs("")
	if len(jobs) != 0 {
		t.Errorf("expected 0 jobs, got %d", len(jobs))
	}
}

func TestJobStatusTransitions(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()

	// Create a test file
	testFile := filepath.Join(ts.TempDir, "test.txt")
	os.WriteFile(testFile, []byte("test content"), 0o644)

	// Enqueue without starting worker
	jobID, _ := ts.Queue.EnqueueIndexFile(testFile)

	// Should be queued
	job, _ := ts.Store.GetJob(jobID)
	if job.Status != store.JobStatusQueued {
		t.Errorf("expected status queued, got %s", job.Status)
	}

	// Start worker
	ts.Queue.Start()

	// Wait for processing to start
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		job, _ = ts.Store.GetJob(jobID)
		if job.Status != store.JobStatusQueued {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}

	// Should transition through processing to completed
	deadline = time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		job, _ = ts.Store.GetJob(jobID)
		if job.Status == store.JobStatusCompleted {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}

	if job.Status != store.JobStatusCompleted {
		t.Errorf("expected status completed, got %s", job.Status)
	}
}

func TestSearchAfterIndexing(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()

	ts.Queue.Start()

	// Create test files with different content
	testDir := ts.TempDir
	files := map[string]string{
		"golang.txt": "Go is a statically typed, compiled programming language designed at Google.",
		"python.txt": "Python is a high-level, interpreted programming language with dynamic semantics.",
		"rust.txt":   "Rust is a multi-paradigm, general-purpose programming language emphasizing safety.",
	}

	for name, content := range files {
		if err := os.WriteFile(filepath.Join(testDir, name), []byte(content), 0o644); err != nil {
			t.Fatalf("failed to create %s: %v", name, err)
		}
	}

	// Index directory
	jobID, _ := ts.Queue.EnqueueIndexDirectory(testDir, "*.txt", false)

	// Wait for parent job to complete (it creates child jobs)
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		job, _ := ts.Store.GetJob(jobID)
		if job.Status == store.JobStatusCompleted {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	// Wait for all child jobs to complete
	for time.Now().Before(deadline) {
		stats, _ := ts.Store.GetChildJobStats(jobID)
		if stats != nil && stats.Total > 0 && stats.Queued == 0 && stats.Processing == 0 {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	// Search for something
	result, err := ts.Goldie.Query("programming language", 5)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(result.Results) == 0 {
		t.Error("expected search results")
	}

	// Verify we can find content
	found := false
	for _, r := range result.Results {
		if r.Document.Content != "" {
			found = true
			break
		}
	}

	if !found {
		t.Error("expected to find documents with content")
	}
}

// ============================================================================
// MCP Tool Handler Tests
// ============================================================================

func TestMCP_IndexFile(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()
	ts.SetupGlobals()
	ts.Queue.Start()

	// Create test file
	testFile := filepath.Join(ts.TempDir, "test.txt")
	os.WriteFile(testFile, []byte("Test content for MCP indexing"), 0o644)

	// Call index_file via MCP handler
	resp := ts.CallTool(t, "index_file", map[string]interface{}{
		"path": testFile,
	})

	// Should return immediately with job_id
	if resp["success"] != true {
		t.Errorf("expected success=true, got %v", resp["success"])
	}
	if resp["status"] != store.JobStatusQueued {
		t.Errorf("expected status=queued, got %v", resp["status"])
	}
	if resp["job_id"] == nil || resp["job_id"] == "" {
		t.Error("expected non-empty job_id")
	}

	jobID := resp["job_id"].(string)

	// Wait for job to complete
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		statusResp := ts.CallTool(t, "job_status", map[string]interface{}{
			"id": jobID,
		})
		if statusResp["status"] == store.JobStatusCompleted {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	// Verify via job_status
	statusResp := ts.CallTool(t, "job_status", map[string]interface{}{
		"id": jobID,
	})
	if statusResp["status"] != store.JobStatusCompleted {
		t.Errorf("expected completed status, got %v", statusResp["status"])
	}
}

func TestMCP_IndexDirectory(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()
	ts.SetupGlobals()
	ts.Queue.Start()

	// Create test directory with files
	testDir := filepath.Join(ts.TempDir, "docs")
	os.MkdirAll(testDir, 0o755)
	os.WriteFile(filepath.Join(testDir, "a.md"), []byte("Document A"), 0o644)
	os.WriteFile(filepath.Join(testDir, "b.md"), []byte("Document B"), 0o644)
	os.WriteFile(filepath.Join(testDir, "c.txt"), []byte("Document C"), 0o644)

	// Call index_directory via MCP handler
	resp := ts.CallTool(t, "index_directory", map[string]interface{}{
		"directory": testDir,
		"pattern":   "*.md",
		"recursive": false,
	})

	if resp["success"] != true {
		t.Errorf("expected success=true, got %v", resp["success"])
	}

	jobID := resp["job_id"].(string)

	// Wait for completion
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		statusResp := ts.CallTool(t, "job_status", map[string]interface{}{
			"id": jobID,
		})
		if statusResp["status"] == store.JobStatusCompleted {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	// Verify result
	statusResp := ts.CallTool(t, "job_status", map[string]interface{}{
		"id": jobID,
	})

	if statusResp["status"] != store.JobStatusCompleted {
		t.Fatalf("job did not complete: %v", statusResp)
	}

	// Parse the result JSON
	resultStr := statusResp["result"].(string)
	var result map[string]interface{}
	json.Unmarshal([]byte(resultStr), &result)

	fileCount := int(result["file_count"].(float64))
	if fileCount != 2 {
		t.Errorf("expected 2 files indexed, got %d", fileCount)
	}
}

func TestMCP_ListJobs(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()
	ts.SetupGlobals()
	// Don't start worker - jobs stay queued

	testFile := filepath.Join(ts.TempDir, "test.txt")
	os.WriteFile(testFile, []byte("test"), 0o644)

	// Create jobs via MCP
	ts.CallTool(t, "index_file", map[string]interface{}{"path": testFile})
	ts.CallTool(t, "index_file", map[string]interface{}{"path": testFile})

	// List all jobs
	resp := ts.CallTool(t, "list_jobs", map[string]interface{}{})

	count := int(resp["count"].(float64))
	if count != 2 {
		t.Errorf("expected 2 jobs, got %d", count)
	}

	// List filtered by status
	resp = ts.CallTool(t, "list_jobs", map[string]interface{}{
		"status": "queued",
	})

	count = int(resp["count"].(float64))
	if count != 2 {
		t.Errorf("expected 2 queued jobs, got %d", count)
	}
}

func TestMCP_ClearQueue(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()
	ts.SetupGlobals()

	testFile := filepath.Join(ts.TempDir, "test.txt")
	os.WriteFile(testFile, []byte("test"), 0o644)

	// Create jobs
	ts.CallTool(t, "index_file", map[string]interface{}{"path": testFile})
	ts.CallTool(t, "index_file", map[string]interface{}{"path": testFile})
	ts.CallTool(t, "index_file", map[string]interface{}{"path": testFile})

	// Clear all
	resp := ts.CallTool(t, "clear_queue", map[string]interface{}{
		"status": "all",
	})

	if resp["success"] != true {
		t.Errorf("expected success=true")
	}

	deleted := int(resp["deleted"].(float64))
	if deleted != 3 {
		t.Errorf("expected 3 deleted, got %d", deleted)
	}

	// Verify empty (plain text response for empty results has no "count" field)
	listResp := ts.CallTool(t, "list_jobs", map[string]interface{}{})
	if listResp["count"] != nil && int(listResp["count"].(float64)) != 0 {
		t.Error("expected empty job list")
	}
}

func TestMCP_Search(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()
	ts.SetupGlobals()
	ts.Queue.Start()

	// Index a document directly (sync) for immediate search
	_, err := ts.Goldie.Index("Go is a programming language", map[string]string{"type": "info"}, "doc1")
	if err != nil {
		t.Fatalf("failed to index: %v", err)
	}

	// Search via MCP
	resp := ts.CallTool(t, "search_index", map[string]interface{}{
		"query": "programming",
		"limit": float64(5),
	})

	count := int(resp["count"].(float64))
	if count == 0 {
		t.Error("expected search results")
	}

	results := resp["results"].([]interface{})
	if len(results) == 0 {
		t.Error("expected results array")
	}
}

func TestMCP_DocumentCRUD(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()
	ts.SetupGlobals()

	// Index content
	resp := ts.CallTool(t, "index_content", map[string]interface{}{
		"content":  "This is test content",
		"metadata": `{"source": "test"}`,
	})

	if resp["success"] != true {
		t.Errorf("expected success=true")
	}
	docID, ok := resp["id"].(string)
	if !ok || docID == "" {
		t.Errorf("expected auto-generated id, got %v", resp["id"])
	}

	// Count documents
	countResp := ts.CallTool(t, "count_documents", map[string]interface{}{})
	if int(countResp["count"].(float64)) != 1 {
		t.Errorf("expected count=1")
	}

	// Delete document using auto-generated ID
	delResp := ts.CallTool(t, "delete_document", map[string]interface{}{
		"id": docID,
	})
	if delResp["success"] != true {
		t.Errorf("expected delete success")
	}

	// Verify deleted
	countResp = ts.CallTool(t, "count_documents", map[string]interface{}{})
	if int(countResp["count"].(float64)) != 0 {
		t.Errorf("expected count=0 after delete")
	}
}
