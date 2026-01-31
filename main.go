package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"

	"github.com/srfrog/goldie-mcp/internal/embedder"
	"github.com/srfrog/goldie-mcp/internal/embedder/ollama"
	"github.com/srfrog/goldie-mcp/internal/goldie"
	"github.com/srfrog/goldie-mcp/internal/queue"
	"github.com/srfrog/goldie-mcp/internal/store"
)

const statusEmoji = "🐕"

// safeJSONMarshal marshals v to JSON, logging any errors
func safeJSONMarshal(v any) string {
	data, err := json.Marshal(v)
	if err != nil {
		errLog.Printf("JSON marshal error: %v", err)
		return `{"error":"internal error: failed to marshal response"}`
	}
	return string(data)
}

type jobResponse struct {
	ID        string    `json:"id"`
	Type      string    `json:"type"`
	Status    string    `json:"status"`
	StatusRaw string    `json:"status_raw"`
	Params    string    `json:"params"`
	Result    string    `json:"result,omitempty"`
	Error     string    `json:"error,omitempty"`
	Progress  int       `json:"progress"`
	Total     int       `json:"total"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

func formatMessage(format string, args ...any) string {
	return fmt.Sprintf(statusEmoji+" "+format, args...)
}

var (
	goldieInstance *goldie.Goldie
	storeInstance  *store.Store
	queueInstance  *queue.Queue
	errLog         *log.Logger
)

func main() {
	// Parse flags
	logFile := flag.String("l", "", "Log errors to file (default: stderr)")
	backend := flag.String("b", "minilm", "Embedding backend: minilm, ollama")
	flag.Parse()

	// Set up error logging
	var errWriter io.Writer = os.Stderr
	if *logFile != "" {
		f, err := os.OpenFile(*logFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to open log file: %v\n", err)
			os.Exit(1)
		}
		defer f.Close()
		errWriter = f
	}
	errLog = log.New(errWriter, "", log.LstdFlags)

	// Initialize RAG
	errLog.Printf("Starting RAG initialization...")
	cfg := goldie.DefaultConfig()
	cfg.Logger = errLog // Pass logger for debugging

	// Allow override via environment variable
	if dbPath := os.Getenv("GOLDIE_DB_PATH"); dbPath != "" {
		cfg.DBPath = dbPath
	}
	errLog.Printf("DB path: %s", cfg.DBPath)
	errLog.Printf("Backend: %s", *backend)

	// Create embedder based on backend
	var emb embedder.Interface
	var err error
	switch *backend {
	case "minilm":
		errLog.Printf("ONNXRUNTIME_LIB_PATH: %s", os.Getenv("ONNXRUNTIME_LIB_PATH"))
		errLog.Printf("Creating MiniLM embedder...")
		emb, err = embedder.New()
		if err != nil {
			errLog.Printf("Failed to create MiniLM embedder: %v", err)
			os.Exit(1)
		}
		cfg.Dimensions = emb.GetDimensions()
	case "ollama":
		ollamaCfg := ollama.Config{
			BaseURL:    os.Getenv("OLLAMA_HOST"),
			Model:      os.Getenv("OLLAMA_EMBED_MODEL"),
			Dimensions: 0, // Will use default based on model
		}
		if ollamaCfg.BaseURL == "" {
			ollamaCfg.BaseURL = "http://localhost:11434"
		}
		if ollamaCfg.Model == "" {
			ollamaCfg.Model = "nomic-embed-text"
		}
		// Set dimensions based on model
		switch ollamaCfg.Model {
		case "nomic-embed-text":
			ollamaCfg.Dimensions = ollama.DimensionsNomicEmbedText
		case "mxbai-embed-large":
			ollamaCfg.Dimensions = ollama.DimensionsMxbaiEmbedLarge
		case "all-minilm":
			ollamaCfg.Dimensions = ollama.DimensionsAllMiniLM
		default:
			// Check for OLLAMA_EMBED_DIMENSIONS env var for custom models
			if dimStr := os.Getenv("OLLAMA_EMBED_DIMENSIONS"); dimStr != "" {
				var dim int
				if _, err := fmt.Sscanf(dimStr, "%d", &dim); err == nil && dim > 0 {
					ollamaCfg.Dimensions = dim
				}
			}
			if ollamaCfg.Dimensions == 0 {
				ollamaCfg.Dimensions = ollama.DimensionsNomicEmbedText // fallback
			}
		}
		errLog.Printf("Creating Ollama embedder (host=%s, model=%s, dims=%d)...",
			ollamaCfg.BaseURL, ollamaCfg.Model, ollamaCfg.Dimensions)
		emb, err = ollama.New(ollamaCfg)
		if err != nil {
			errLog.Printf("Failed to create Ollama embedder: %v", err)
			os.Exit(1)
		}
		cfg.Dimensions = ollamaCfg.Dimensions
	default:
		errLog.Printf("Unknown backend: %s (supported: minilm, ollama)", *backend)
		os.Exit(1)
	}
	cfg.Embedder = emb

	errLog.Printf("Creating RAG instance...")
	goldieInstance, err = goldie.New(cfg)
	if err != nil {
		errLog.Printf("Failed to initialize RAG: %v", err)
		os.Exit(1)
	}
	defer goldieInstance.Close()
	errLog.Printf("RAG initialized successfully")

	// Warmup the embedding model
	errLog.Printf("Warming up embedding model...")
	if err := goldieInstance.Warmup(); err != nil {
		errLog.Printf("Failed to warmup embedding model: %v", err)
		os.Exit(1)
	}
	errLog.Printf("Embedding model ready")

	// Get store reference and create queue
	storeInstance = goldieInstance.Store()
	queueInstance = queue.New(storeInstance, goldieInstance, errLog)
	queueInstance.Start()
	defer queueInstance.Stop()

	// Create MCP server
	s := server.NewMCPServer(
		"goldie-mcp",
		"1.0.0",
		server.WithToolCapabilities(true),
	)

	// Register tools
	registerTools(s)

	// Set up signal handling for graceful shutdown and crash logging
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Handle crash signals to log before dying
	crashChan := make(chan os.Signal, 1)
	signal.Notify(crashChan, syscall.SIGSEGV, syscall.SIGABRT, syscall.SIGBUS)
	go func() {
		sig := <-crashChan
		errLog.Printf("CRASH: Received signal %v - likely ONNX runtime crash", sig)
		os.Exit(2)
	}()

	// Start stdio server in a goroutine
	errChan := make(chan error, 1)
	go func() {
		defer func() {
			if r := recover(); r != nil {
				errLog.Printf("Server panic: %v", r)
				errChan <- fmt.Errorf("server panic: %v", r)
			}
		}()
		// ServeStdio reads from stdin, writes to stdout
		errChan <- server.ServeStdio(s)
	}()

	// Wait for shutdown signal or server error
	select {
	case sig := <-sigChan:
		errLog.Printf("Received signal %v, shutting down", sig)
	case err := <-errChan:
		if err != nil {
			errLog.Printf("Server error: %v", err)
			os.Exit(1)
		}
	}
}

func registerTools(s *server.MCPServer) {
	// index_content tool
	s.AddTool(
		mcp.NewTool("index_content",
			mcp.WithDescription("Index text content for semantic search. Use for web pages, API responses, notes, or any text that doesn't come from a local file. For local files, use index_file instead."),
			mcp.WithString("content",
				mcp.Required(),
				mcp.Description("The text content to index"),
			),
			mcp.WithString("metadata",
				mcp.Description("Optional JSON object with metadata (e.g., {\"source\": \"https://example.com\", \"title\": \"Page Title\"})"),
			),
		),
		handleIndexContent,
	)

	// index_file tool
	s.AddTool(
		mcp.NewTool("index_file",
			mcp.WithDescription("Index a file from the filesystem for semantic search"),
			mcp.WithString("path",
				mcp.Required(),
				mcp.Description("The file path to read and index"),
			),
		),
		handleIndexFile,
	)

	// index_directory tool
	s.AddTool(
		mcp.NewTool("index_directory",
			mcp.WithDescription("Index all files matching a pattern in a directory"),
			mcp.WithString("directory",
				mcp.Required(),
				mcp.Description("The directory path to index"),
			),
			mcp.WithString("pattern",
				mcp.Description("File pattern to match (e.g., '*.md', '*.txt'). Default: '*'"),
			),
			mcp.WithBoolean("recursive",
				mcp.Description("Whether to search subdirectories recursively. Default: false"),
			),
		),
		handleIndexDirectory,
	)

	// search_index tool
	s.AddTool(
		mcp.NewTool("search_index",
			mcp.WithDescription("Search for documents using semantic similarity. Use when you need to find specific documents or your context needs more guidance. Returns document metadata and content."),
			mcp.WithString("query",
				mcp.Required(),
				mcp.Description("The search query text"),
			),
			mcp.WithNumber("limit",
				mcp.Description("Maximum number of results to return (default: 5)"),
			),
		),
		handleSearch,
	)

	// recall tool
	s.AddTool(
		mcp.NewTool("recall",
			mcp.WithDescription("Recall knowledge from indexed documents about a topic. Use the returned content to update your context directly - no need for additional lookups. Designed for natural conversation flow."),
			mcp.WithString("topic",
				mcp.Required(),
				mcp.Description("The topic to recall information about"),
			),
			mcp.WithNumber("depth",
				mcp.Description("How many sources to consult (default: 5, max: 20)"),
			),
		),
		handleRecall,
	)

	// list_files tool
	s.AddTool(
		mcp.NewTool("list_files",
			mcp.WithDescription("List unique indexed source files (not chunks)"),
		),
		handleListFiles,
	)

	// delete_document tool
	s.AddTool(
		mcp.NewTool("delete_document",
			mcp.WithDescription("Delete a document from the index"),
			mcp.WithString("id",
				mcp.Required(),
				mcp.Description("The document ID to delete"),
			),
		),
		handleDeleteDocument,
	)

	// count_documents tool
	s.AddTool(
		mcp.NewTool("count_documents",
			mcp.WithDescription("Get the total number of indexed documents"),
		),
		handleCountDocuments,
	)

	// job_status tool
	s.AddTool(
		mcp.NewTool("job_status",
			mcp.WithDescription("Get the status of an indexing job"),
			mcp.WithString("id",
				mcp.Required(),
				mcp.Description("The job ID to check"),
			),
			mcp.WithBoolean("block",
				mcp.Description("If true, wait for job to complete (default: false)"),
			),
			mcp.WithNumber("timeout",
				mcp.Description("Timeout in seconds when blocking (default: 30)"),
			),
		),
		handleJobStatus,
	)

	// list_jobs tool
	s.AddTool(
		mcp.NewTool("list_jobs",
			mcp.WithDescription("List indexing jobs"),
			mcp.WithString("status",
				mcp.Description("Filter by status: queued, processing, completed, failed (optional)"),
			),
		),
		handleListJobs,
	)

	// clear_queue tool
	s.AddTool(
		mcp.NewTool("clear_queue",
			mcp.WithDescription("Clear jobs from the queue"),
			mcp.WithString("status",
				mcp.Required(),
				mcp.Description("Status to clear: queued, completed, failed, or 'all'"),
			),
		),
		handleClearQueue,
	)
}

func handleIndexContent(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	content, ok := request.Params.Arguments["content"].(string)
	if !ok || content == "" {
		return mcp.NewToolResultError("content is required"), nil
	}

	var metadata map[string]string
	if metaStr, ok := request.Params.Arguments["metadata"].(string); ok && metaStr != "" {
		if err := json.Unmarshal([]byte(metaStr), &metadata); err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("invalid metadata JSON: %v", err)), nil
		}
	}

	// Always auto-generate ID
	result, err := goldieInstance.Index(content, metadata, "")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("indexing failed: %v", err)), nil
	}

	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"success":     true,
		"id":          result.ID,
		"chunk_count": result.ChunkCount,
		"message":     formatMessage("Indexed content with ID: %s (%d chunks)", result.ID, result.ChunkCount),
	})), nil
}

func handleIndexFile(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	path, ok := request.Params.Arguments["path"].(string)
	if !ok || path == "" {
		return mcp.NewToolResultError("path is required"), nil
	}

	jobID, err := queueInstance.EnqueueIndexFile(path)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("failed to queue job: %v", err)), nil
	}
	status := store.JobStatusQueued

	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"success":    true,
		"job_id":     jobID,
		"status":     status,
		"status_raw": status,
		"path":       path,
		"message":    formatMessage("Job queued for indexing: %s (job_id: %s)", path, jobID),
	})), nil
}

func handleIndexDirectory(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	dir, ok := request.Params.Arguments["directory"].(string)
	if !ok || dir == "" {
		return mcp.NewToolResultError("directory is required"), nil
	}

	pattern := "*"
	if p, ok := request.Params.Arguments["pattern"].(string); ok && p != "" {
		pattern = p
	}

	recursive := false
	if r, ok := request.Params.Arguments["recursive"].(bool); ok {
		recursive = r
	}

	jobID, err := queueInstance.EnqueueIndexDirectory(dir, pattern, recursive)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("failed to queue job: %v", err)), nil
	}
	status := store.JobStatusQueued

	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"success":    true,
		"job_id":     jobID,
		"status":     status,
		"status_raw": status,
		"directory":  dir,
		"pattern":    pattern,
		"recursive":  recursive,
		"message":    formatMessage("Job queued for indexing directory: %s (job_id: %s)", dir, jobID),
	})), nil
}

func handleSearch(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	query, ok := request.Params.Arguments["query"].(string)
	if !ok || query == "" {
		return mcp.NewToolResultError("query is required"), nil
	}

	limit := 5
	if limitVal, ok := request.Params.Arguments["limit"].(float64); ok {
		limit = int(limitVal)
	}

	result, err := goldieInstance.Query(query, limit)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("search failed: %v", err)), nil
	}

	// Return plain text for empty results
	if len(result.Results) == 0 {
		return mcp.NewToolResultText(formatMessage("No results found for '%s'", query)), nil
	}

	// Format results for better readability
	var formattedResults []map[string]any
	for _, r := range result.Results {
		formattedResults = append(formattedResults, map[string]any{
			"id":       r.Document.ID,
			"content":  r.Document.Content,
			"metadata": r.Document.Metadata,
			"score":    r.Score,
		})
	}

	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"query":   query,
		"count":   len(result.Results),
		"results": formattedResults,
		"message": formatMessage("Found %d result(s) for '%s'", len(result.Results), query),
	})), nil
}

func handleRecall(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	topic, ok := request.Params.Arguments["topic"].(string)
	if !ok || topic == "" {
		return mcp.NewToolResultError("topic is required"), nil
	}

	depth := 5
	if depthVal, ok := request.Params.Arguments["depth"].(float64); ok {
		depth = max(min(int(depthVal), 20), 1)
	}

	result, err := goldieInstance.Query(topic, depth)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("recall failed: %v", err)), nil
	}

	if len(result.Results) == 0 {
		return mcp.NewToolResultText(formatMessage("No knowledge found about '%s'", topic)), nil
	}

	// Group results by source and consolidate
	type sourceInfo struct {
		filename string
		excerpts []string
		score    float32
	}
	sources := make(map[string]*sourceInfo)

	for _, r := range result.Results {
		source := r.Document.ID
		filename := source
		if r.Document.Metadata != nil {
			if s, ok := r.Document.Metadata["source"]; ok && s != "" {
				source = s
			}
			if f, ok := r.Document.Metadata["filename"]; ok && f != "" {
				filename = f
			}
		}

		if _, exists := sources[source]; !exists {
			sources[source] = &sourceInfo{
				filename: filename,
				excerpts: []string{},
				score:    r.Score,
			}
		}
		// Truncate long content for summary
		content := r.Document.Content
		if len(content) > 500 {
			content = content[:500] + "..."
		}
		sources[source].excerpts = append(sources[source].excerpts, content)
	}

	// Build response - pure content, no file references
	var sb strings.Builder
	fmt.Fprintf(&sb, "%s Knowledge about '%s':\n\n", statusEmoji, topic)

	for _, info := range sources {
		for _, excerpt := range info.excerpts {
			sb.WriteString(excerpt)
			sb.WriteString("\n\n")
		}
	}

	return mcp.NewToolResultText(sb.String()), nil
}

func handleListFiles(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	docs, err := goldieInstance.ListDocuments()
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("listing documents failed: %v", err)), nil
	}

	// Return plain text for empty results
	if len(docs) == 0 {
		return mcp.NewToolResultText(formatMessage("No files indexed")), nil
	}

	// Extract unique source files
	fileSet := make(map[string]struct {
		Source     string
		Filename   string
		ChunkCount int
	})

	for _, doc := range docs {
		source := ""
		filename := ""
		if doc.Metadata != nil {
			source = doc.Metadata["source"]
			filename = doc.Metadata["filename"]
		}
		if source == "" {
			// Use ID for non-file documents
			source = doc.ID
			filename = doc.ID
		}

		if entry, exists := fileSet[source]; exists {
			entry.ChunkCount++
			fileSet[source] = entry
		} else {
			fileSet[source] = struct {
				Source     string
				Filename   string
				ChunkCount int
			}{source, filename, 1}
		}
	}

	// Convert to slice
	var files []map[string]any
	for _, entry := range fileSet {
		files = append(files, map[string]any{
			"source":      entry.Source,
			"filename":    entry.Filename,
			"chunk_count": entry.ChunkCount,
		})
	}

	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"count":   len(files),
		"files":   files,
		"message": formatMessage("Found %d file(s)", len(files)),
	})), nil
}

func handleDeleteDocument(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	id, ok := request.Params.Arguments["id"].(string)
	if !ok || id == "" {
		return mcp.NewToolResultError("id is required"), nil
	}

	// Delete document and all its chunks
	deleted := goldieInstance.DeleteDocumentAndChunks(id)

	if deleted == 0 {
		return mcp.NewToolResultError(fmt.Sprintf("document not found: %s", id)), nil
	}

	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"success":       true,
		"id":            id,
		"deleted_count": deleted,
		"message":       formatMessage("Deleted %d document(s)/chunk(s) for: %s", deleted, id),
	})), nil
}

func handleCountDocuments(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	count, err := goldieInstance.Count()
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("counting documents failed: %v", err)), nil
	}

	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"count":   count,
		"message": formatMessage("%d document(s) indexed", count),
	})), nil
}

func handleJobStatus(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	id, ok := request.Params.Arguments["id"].(string)
	if !ok || id == "" {
		return mcp.NewToolResultError("id is required"), nil
	}

	// Check if blocking is requested
	block := false
	if b, ok := request.Params.Arguments["block"].(bool); ok {
		block = b
	}

	timeout := 30 * time.Second
	if t, ok := request.Params.Arguments["timeout"].(float64); ok && t > 0 {
		timeout = time.Duration(t) * time.Second
	}

	var job *store.Job
	var err error

	if block {
		job, err = storeInstance.WaitForJob(id, timeout)
	} else {
		job, err = storeInstance.GetJob(id)
	}

	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("getting job status failed: %v", err)), nil
	}

	if job == nil {
		return mcp.NewToolResultError(fmt.Sprintf("job not found: %s", id)), nil
	}

	// For index_directory jobs, include child job statistics
	if job.Type == store.JobTypeIndexDir {
		childStats, err := storeInstance.GetChildJobStats(id)
		if err != nil {
			errLog.Printf("Failed to get child job stats: %v", err)
		}

		response := map[string]any{
			"id":         job.ID,
			"type":       job.Type,
			"status":     job.Status,
			"params":     job.Params,
			"result":     job.Result,
			"error":      job.Error,
			"progress":   job.Progress,
			"total":      job.Total,
			"created_at": job.CreatedAt,
			"updated_at": job.UpdatedAt,
		}

		if childStats != nil && childStats.Total > 0 {
			response["child_jobs"] = map[string]any{
				"total":      childStats.Total,
				"queued":     childStats.Queued,
				"processing": childStats.Processing,
				"completed":  childStats.Completed,
				"failed":     childStats.Failed,
			}
			// Update progress to reflect child jobs
			response["progress"] = childStats.Completed + childStats.Failed
			response["total"] = childStats.Total
		}

		return mcp.NewToolResultText(safeJSONMarshal(response)), nil
	}

	return mcp.NewToolResultText(safeJSONMarshal(job)), nil
}

func handleListJobs(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	status := ""
	if s, ok := request.Params.Arguments["status"].(string); ok {
		status = s
	}

	jobs, err := storeInstance.ListJobs(status)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("listing jobs failed: %v", err)), nil
	}

	// Return plain text for empty results
	if len(jobs) == 0 {
		if status != "" {
			return mcp.NewToolResultText(formatMessage("No jobs with status '%s'", status)), nil
		}
		return mcp.NewToolResultText(formatMessage("No jobs found")), nil
	}

	// Format jobs for display
	var message string
	if status != "" {
		message = formatMessage("Found %d job(s) with status '%s'", len(jobs), status)
	} else {
		message = formatMessage("Found %d job(s)", len(jobs))
	}

	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"count":   len(jobs),
		"jobs":    jobs,
		"message": message,
	})), nil
}

func handleClearQueue(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	status := ""
	if s, ok := request.Params.Arguments["status"].(string); ok {
		status = s
	}

	if status == "" {
		return mcp.NewToolResultError("status is required (queued, completed, failed, or all)"), nil
	}

	// Validate status
	validStatuses := map[string]bool{
		"queued":    true,
		"completed": true,
		"failed":    true,
		"all":       true,
	}
	if !validStatuses[status] {
		return mcp.NewToolResultError("invalid status: must be queued, completed, failed, or all"), nil
	}

	count, err := storeInstance.DeleteJobs(status)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("clearing queue failed: %v", err)), nil
	}

	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"success":    true,
		"deleted":    count,
		"status":     status,
		"status_raw": status,
		"message":    formatMessage("Cleared %d jobs with status: %s", count, status),
	})), nil
}
