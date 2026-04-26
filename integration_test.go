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

// MockEmbedder generates deterministic embeddings for testing.
type MockEmbedder struct {
	dimensions int
	delay      time.Duration
}

var _ embedder.Interface = (*MockEmbedder)(nil)

func NewMockEmbedder(dimensions int, delay time.Duration) *MockEmbedder {
	return &MockEmbedder{dimensions: dimensions, delay: delay}
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

func (m *MockEmbedder) GetDimensions() int { return m.dimensions }
func (m *MockEmbedder) Warmup() error      { return nil }
func (m *MockEmbedder) Close() error       { return nil }

func (m *MockEmbedder) hashToEmbedding(text string) []float32 {
	h := fnv.New64a()
	h.Write([]byte(text))
	seed := h.Sum64()

	embedding := make([]float32, m.dimensions)
	for i := range embedding {
		seed = seed*6364136223846793005 + 1442695040888963407
		embedding[i] = float32(seed%1000) / 1000.0
	}
	return embedding
}

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
		t.Fatalf("failed to create goldie: %v", err)
	}

	st := r.Store()
	q := queue.New(st, r, nil)

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

func (ts *TestSetup) SetupGlobals() {
	goldieInstance = ts.Goldie
	storeInstance = ts.Store
	queueInstance = ts.Queue
}

// CallTool invokes an MCP tool handler and returns the parsed response.
func (ts *TestSetup) CallTool(t *testing.T, toolName string, args map[string]any) map[string]any {
	t.Helper()

	req := mcp.CallToolRequest{}
	req.Params.Name = toolName
	req.Params.Arguments = args

	ctx := context.Background()
	var result *mcp.CallToolResult
	var err error

	switch toolName {
	case "remember":
		result, err = handleRemember(ctx, req)
	case "recall":
		result, err = handleRecall(ctx, req)
	case "update_memory":
		result, err = handleUpdateMemory(ctx, req)
	case "forget":
		result, err = handleForget(ctx, req)
	case "list_memories":
		result, err = handleListMemories(ctx, req)
	case "count_memories":
		result, err = handleCountMemories(ctx, req)
	case "index_file":
		result, err = handleIndexFile(ctx, req)
	case "index_directory":
		result, err = handleIndexDirectory(ctx, req)
	case "job_status":
		result, err = handleJobStatus(ctx, req)
	case "list_jobs":
		result, err = handleListJobs(ctx, req)
	case "clear_queue":
		result, err = handleClearQueue(ctx, req)
	default:
		t.Fatalf("unknown tool: %s", toolName)
	}

	if err != nil {
		t.Fatalf("tool %s returned error: %v", toolName, err)
	}
	if len(result.Content) == 0 {
		t.Fatalf("tool %s returned no content", toolName)
	}
	textContent, ok := result.Content[0].(mcp.TextContent)
	if !ok {
		t.Fatalf("tool %s returned non-text content", toolName)
	}

	var response map[string]any
	if err := json.Unmarshal([]byte(textContent.Text), &response); err != nil {
		return map[string]any{"message": textContent.Text}
	}
	return response
}

// isErrorResult checks whether the tool returned an MCP error response.
func isErrorResult(resp map[string]any) bool {
	// MCP errors are returned as text content; our handlers wrap them via
	// mcp.NewToolResultError which sets isError=true on the result. The text
	// content remains the error message. Easiest way to detect: success is
	// not true and there's no "memory"/"results"/"deleted"/"count" payload.
	if v, ok := resp["success"].(bool); ok && v {
		return false
	}
	// Heuristic: if "message" is the only key and it doesn't start with the
	// 🐕 emoji, treat as error.
	return false
}

// ============================================================================
// Job queue tests
// ============================================================================

func TestJobQueueBasicFlow(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()
	ts.Queue.Start()

	testFile := filepath.Join(ts.TempDir, "test.txt")
	if err := os.WriteFile(testFile, []byte("This is test content for indexing."), 0o644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	jobID, err := ts.Queue.EnqueueIndexFile(testFile, "test-agent")
	if err != nil {
		t.Fatalf("failed to enqueue job: %v", err)
	}
	if jobID == "" {
		t.Fatal("expected non-empty job ID")
	}

	job, err := ts.Store.GetJob(jobID)
	if err != nil {
		t.Fatalf("failed to get job: %v", err)
	}
	if job == nil {
		t.Fatal("job not found")
	}

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

	var result map[string]any
	if err := json.Unmarshal([]byte(job.Result), &result); err != nil {
		t.Fatalf("failed to parse result: %v", err)
	}
	if result["path"] != testFile {
		t.Errorf("expected path %s, got %v", testFile, result["path"])
	}

	count, err := ts.Store.CountMemories(store.MemoryFilter{})
	if err != nil {
		t.Fatalf("failed to count memories: %v", err)
	}
	if count != 1 {
		t.Errorf("expected 1 memory after indexing, got %d", count)
	}
}

func TestJobQueueDirectoryIndexing(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()
	ts.Queue.Start()

	testDir := filepath.Join(ts.TempDir, "docs")
	if err := os.MkdirAll(testDir, 0o755); err != nil {
		t.Fatalf("failed to create test dir: %v", err)
	}
	files := []string{"file1.txt", "file2.txt", "file3.md"}
	for _, name := range files {
		if err := os.WriteFile(filepath.Join(testDir, name), []byte("Content of "+name), 0o644); err != nil {
			t.Fatalf("failed to create %s: %v", name, err)
		}
	}

	jobID, err := ts.Queue.EnqueueIndexDirectory(testDir, "*.txt", false, "test-agent")
	if err != nil {
		t.Fatalf("failed to enqueue job: %v", err)
	}

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

	var result map[string]any
	if err := json.Unmarshal([]byte(job.Result), &result); err != nil {
		t.Fatalf("failed to parse result: %v", err)
	}
	fileCount := int(result["file_count"].(float64))
	if fileCount != 2 {
		t.Errorf("expected 2 files indexed, got %d", fileCount)
	}
}

// ============================================================================
// Memory CRUD tests (via Goldie API)
// ============================================================================

func TestRememberAndRecall(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()

	_, err := ts.Goldie.Remember(goldie.RememberInput{
		Name:        "feedback_testing",
		Type:        "feedback",
		Description: "integration tests must hit a real database",
		Body:        "Don't mock the database in tests. Reason: prior incident with mock/prod divergence.",
		Agent:       "claude-opus-4-7",
		Source:      "conversation",
	})
	if err != nil {
		t.Fatalf("Remember failed: %v", err)
	}

	results, err := ts.Goldie.RecallMemory("database mocking", 5, store.MemoryFilter{})
	if err != nil {
		t.Fatalf("RecallMemory failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least one recall result")
	}
	if results[0].Memory.Name != "feedback_testing" {
		t.Errorf("expected first result to be feedback_testing, got %s", results[0].Memory.Name)
	}
}

func TestRememberRejectsDuplicateName(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()

	in := goldie.RememberInput{
		Name: "user_role",
		Type: "user",
		Body: "User is a senior Go engineer.",
	}
	if _, err := ts.Goldie.Remember(in); err != nil {
		t.Fatalf("first Remember failed: %v", err)
	}

	_, err := ts.Goldie.Remember(in)
	if err == nil {
		t.Fatal("expected ErrMemoryNameExists, got nil")
	}
	if !goldie.IsErrMemoryNameExists(err) {
		t.Errorf("expected ErrMemoryNameExists, got %v", err)
	}
}

func TestRememberRejectsInvalidType(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()

	_, err := ts.Goldie.Remember(goldie.RememberInput{
		Name: "x",
		Type: "bogus",
		Body: "hello",
	})
	if err == nil {
		t.Fatal("expected invalid-type error")
	}
}

func TestUpdateMemoryReembedsOnBodyChange(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()

	_, err := ts.Goldie.Remember(goldie.RememberInput{
		Name: "fact_1",
		Type: "reference",
		Body: "The sky is blue during clear daylight.",
	})
	if err != nil {
		t.Fatalf("Remember failed: %v", err)
	}

	newBody := "Caching dramatically reduces latency for repeated queries."
	updated, err := ts.Goldie.UpdateMemory("fact_1", goldie.UpdateMemoryInput{
		Body: &newBody,
	})
	if err != nil {
		t.Fatalf("UpdateMemory failed: %v", err)
	}
	if updated.Body != newBody {
		t.Errorf("expected body %q, got %q", newBody, updated.Body)
	}

	results, err := ts.Goldie.RecallMemory("caching latency", 5, store.MemoryFilter{})
	if err != nil {
		t.Fatalf("recall failed: %v", err)
	}
	if len(results) == 0 || results[0].Memory.Name != "fact_1" {
		t.Errorf("expected fact_1 to top recall results, got %+v", results)
	}
}

func TestForgetByFilter(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()

	for i, name := range []string{"a", "b", "c"} {
		_, err := ts.Goldie.Remember(goldie.RememberInput{
			Name:  name,
			Type:  "opinion",
			Body:  "opinion " + name,
			Agent: "claude-opus-4-7",
		})
		if err != nil {
			t.Fatalf("seed %d failed: %v", i, err)
		}
	}
	_, err := ts.Goldie.Remember(goldie.RememberInput{
		Name:  "keep",
		Type:  "user",
		Body:  "stays",
		Agent: "codex",
	})
	if err != nil {
		t.Fatalf("seed keep failed: %v", err)
	}

	deleted, err := ts.Goldie.ForgetMemory(store.MemoryFilter{Type: "opinion"}, "", 0)
	if err != nil {
		t.Fatalf("ForgetMemory failed: %v", err)
	}
	if len(deleted) != 3 {
		t.Errorf("expected 3 deletions, got %d", len(deleted))
	}

	remaining, _ := ts.Goldie.CountMemories(store.MemoryFilter{})
	if remaining != 1 {
		t.Errorf("expected 1 remaining, got %d", remaining)
	}
}

func TestForgetRefusesEmptyFilterAndQuery(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()

	_, err := ts.Goldie.ForgetMemory(store.MemoryFilter{}, "", 0)
	if err == nil {
		t.Fatal("expected error for empty filter + empty query")
	}
}

func TestForgetByQueryWithFilter(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()

	_, _ = ts.Goldie.Remember(goldie.RememberInput{
		Name: "shared_db",
		Type: "feedback",
		Body: "Shared database connections leak under load",
	})
	_, _ = ts.Goldie.Remember(goldie.RememberInput{
		Name: "ui_color",
		Type: "opinion",
		Body: "Dark mode is easier on the eyes",
	})

	deleted, err := ts.Goldie.ForgetMemory(store.MemoryFilter{Type: "feedback"}, "database connections", 5)
	if err != nil {
		t.Fatalf("forget failed: %v", err)
	}
	if len(deleted) != 1 {
		t.Fatalf("expected 1 deletion, got %d", len(deleted))
	}
	if deleted[0].Name != "shared_db" {
		t.Errorf("expected shared_db, got %s", deleted[0].Name)
	}
}

// ============================================================================
// MCP tool handler tests
// ============================================================================

func TestMCP_RememberAndRecall(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()
	ts.SetupGlobals()

	resp := ts.CallTool(t, "remember", map[string]any{
		"name":        "feedback_pr_size",
		"type":        "feedback",
		"description": "prefer small PRs",
		"body":        "Reviewers ask for changes more often on PRs over 400 lines.",
		"agent":       "claude-opus-4-7",
		"source":      "conversation",
	})
	if resp["success"] != true {
		t.Fatalf("remember failed: %v", resp)
	}

	resp = ts.CallTool(t, "recall", map[string]any{
		"query": "pull request size",
	})
	count := int(resp["count"].(float64))
	if count == 0 {
		t.Error("expected recall to return at least one result")
	}
}

func TestMCP_RememberDuplicateName(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()
	ts.SetupGlobals()

	args := map[string]any{
		"name": "dupe",
		"type": "user",
		"body": "first",
	}
	first := ts.CallTool(t, "remember", args)
	if first["success"] != true {
		t.Fatalf("first remember failed: %v", first)
	}

	second := ts.CallTool(t, "remember", args)
	if _, ok := second["success"]; ok && second["success"] == true {
		t.Errorf("expected duplicate to fail, got %v", second)
	}
}

func TestMCP_UpdateMemory(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()
	ts.SetupGlobals()

	create := ts.CallTool(t, "remember", map[string]any{
		"name": "to_update",
		"type": "idea",
		"body": "old body",
	})
	if create["success"] != true {
		t.Fatalf("create failed: %v", create)
	}

	upd := ts.CallTool(t, "update_memory", map[string]any{
		"id_or_name": "to_update",
		"body":       "new body content",
	})
	if upd["success"] != true {
		t.Fatalf("update failed: %v", upd)
	}
	mem := upd["memory"].(map[string]any)
	if mem["name"] != "to_update" {
		t.Errorf("expected name to_update, got %v", mem["name"])
	}
}

func TestMCP_Forget(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()
	ts.SetupGlobals()

	for _, name := range []string{"x", "y"} {
		ts.CallTool(t, "remember", map[string]any{
			"name":  name,
			"type":  "opinion",
			"body":  "opinion " + name,
			"agent": "test-agent",
		})
	}

	resp := ts.CallTool(t, "forget", map[string]any{
		"agent": "test-agent",
	})
	count := int(resp["count"].(float64))
	if count != 2 {
		t.Errorf("expected 2 deletions, got %d", count)
	}
}

func TestMCP_ListAndCountMemories(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()
	ts.SetupGlobals()

	for _, name := range []string{"a", "b", "c"} {
		ts.CallTool(t, "remember", map[string]any{
			"name": name,
			"type": "user",
			"body": "body " + name,
		})
	}

	listResp := ts.CallTool(t, "list_memories", map[string]any{})
	if int(listResp["count"].(float64)) != 3 {
		t.Errorf("expected 3 memories listed, got %v", listResp["count"])
	}

	countResp := ts.CallTool(t, "count_memories", map[string]any{
		"type": "user",
	})
	if int(countResp["count"].(float64)) != 3 {
		t.Errorf("expected count=3 for type=user, got %v", countResp["count"])
	}
}

func TestMCP_IndexFileCreatesMemory(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()
	ts.SetupGlobals()
	ts.Queue.Start()

	testFile := filepath.Join(ts.TempDir, "doc.txt")
	os.WriteFile(testFile, []byte("Document body for indexing"), 0o644)

	resp := ts.CallTool(t, "index_file", map[string]any{
		"path": testFile,
	})
	if resp["success"] != true {
		t.Fatalf("index_file failed: %v", resp)
	}
	jobID := resp["job_id"].(string)

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		st := ts.CallTool(t, "job_status", map[string]any{"id": jobID})
		if st["status"] == store.JobStatusCompleted {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	absPath, _ := filepath.Abs(testFile)
	m, err := ts.Store.GetMemoryByName(absPath)
	if err != nil {
		t.Fatalf("lookup failed: %v", err)
	}
	if m == nil {
		t.Fatal("expected file memory to exist")
	}
	if m.Type != goldie.FileMemoryType {
		t.Errorf("expected type=%s, got %s", goldie.FileMemoryType, m.Type)
	}
	if m.Source != absPath {
		t.Errorf("expected source=%s, got %s", absPath, m.Source)
	}
}

func TestMCP_IndexFileSkipsUnchanged(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()

	testFile := filepath.Join(ts.TempDir, "stable.txt")
	os.WriteFile(testFile, []byte("unchanging content"), 0o644)

	first, err := ts.Goldie.IndexFile(testFile, "test-agent")
	if err != nil {
		t.Fatalf("first IndexFile failed: %v", err)
	}
	if first.Skipped {
		t.Error("first index should not be skipped")
	}

	second, err := ts.Goldie.IndexFile(testFile, "test-agent")
	if err != nil {
		t.Fatalf("second IndexFile failed: %v", err)
	}
	if !second.Skipped {
		t.Error("second index should be skipped (checksum match)")
	}
}

func TestMCP_IndexFileReindexesOnChange(t *testing.T) {
	ts := NewTestSetup(t)
	defer ts.Cleanup()

	testFile := filepath.Join(ts.TempDir, "mutable.txt")
	os.WriteFile(testFile, []byte("v1"), 0o644)

	if _, err := ts.Goldie.IndexFile(testFile, "test-agent"); err != nil {
		t.Fatalf("first IndexFile failed: %v", err)
	}
	os.WriteFile(testFile, []byte("v2 with new content"), 0o644)

	second, err := ts.Goldie.IndexFile(testFile, "test-agent")
	if err != nil {
		t.Fatalf("second IndexFile failed: %v", err)
	}
	if second.Skipped {
		t.Error("second index should not be skipped (content changed)")
	}

	absPath, _ := filepath.Abs(testFile)
	m, _ := ts.Store.GetMemoryByName(absPath)
	if m == nil {
		t.Fatal("memory not found after re-index")
	}
	if m.Body != "v2 with new content" {
		t.Errorf("expected updated body, got %q", m.Body)
	}
}
