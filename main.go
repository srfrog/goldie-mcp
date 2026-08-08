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

func safeJSONMarshal(v any) string {
	data, err := json.Marshal(v)
	if err != nil {
		errLog.Printf("JSON marshal error: %v", err)
		return `{"error":"internal error: failed to marshal response"}`
	}
	return string(data)
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
	logFile := flag.String("l", "", "Log errors to file (default: stderr)")
	backend := flag.String("b", "minilm", "Embedding backend: minilm, ollama")
	flag.Parse()

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

	errLog.Printf("Starting goldie-mcp...")
	cfg := goldie.DefaultConfig()
	cfg.Logger = errLog

	if dbPath := os.Getenv("GOLDIE_DB_PATH"); dbPath != "" {
		cfg.DBPath = dbPath
	}
	if jm := os.Getenv("GOLDIE_JOURNAL_MODE"); jm != "" {
		cfg.JournalMode = jm
	}
	errLog.Printf("DB path: %s", cfg.DBPath)
	jmLog := cfg.JournalMode
	if jmLog == "" {
		jmLog = "DELETE (default)"
	}
	errLog.Printf("Journal mode: %s", jmLog)
	errLog.Printf("Backend: %s", *backend)

	var emb embedder.Interface
	var err error
	switch *backend {
	case "minilm":
		errLog.Printf("ONNXRUNTIME_LIB_PATH: %s", os.Getenv("ONNXRUNTIME_LIB_PATH"))
		emb, err = embedder.New()
		if err != nil {
			errLog.Printf("Failed to create MiniLM embedder: %v", err)
			os.Exit(1)
		}
		cfg.Dimensions = emb.GetDimensions()
	case "ollama":
		ollamaCfg := ollama.Config{
			BaseURL: os.Getenv("OLLAMA_HOST"),
			Model:   os.Getenv("OLLAMA_EMBED_MODEL"),
		}
		if ollamaCfg.BaseURL == "" {
			ollamaCfg.BaseURL = "http://localhost:11434"
		}
		if ollamaCfg.Model == "" {
			ollamaCfg.Model = "nomic-embed-text"
		}
		switch ollamaCfg.Model {
		case "nomic-embed-text":
			ollamaCfg.Dimensions = ollama.DimensionsNomicEmbedText
		case "mxbai-embed-large":
			ollamaCfg.Dimensions = ollama.DimensionsMxbaiEmbedLarge
		case "all-minilm":
			ollamaCfg.Dimensions = ollama.DimensionsAllMiniLM
		default:
			if dimStr := os.Getenv("OLLAMA_EMBED_DIMENSIONS"); dimStr != "" {
				var dim int
				if _, err := fmt.Sscanf(dimStr, "%d", &dim); err == nil && dim > 0 {
					ollamaCfg.Dimensions = dim
				}
			}
			if ollamaCfg.Dimensions == 0 {
				ollamaCfg.Dimensions = ollama.DimensionsNomicEmbedText
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

	goldieInstance, err = goldie.New(cfg)
	if err != nil {
		errLog.Printf("Failed to initialize goldie: %v", err)
		os.Exit(1)
	}
	defer goldieInstance.Close()

	errLog.Printf("Warming up embedding model...")
	if err := goldieInstance.Warmup(); err != nil {
		errLog.Printf("Failed to warmup embedding model: %v", err)
		os.Exit(1)
	}

	storeInstance = goldieInstance.Store()
	queueInstance = queue.New(storeInstance, goldieInstance, errLog)
	queueInstance.Start()
	defer queueInstance.Stop()

	s := server.NewMCPServer(
		"goldie-mcp",
		"2.0.0",
		server.WithToolCapabilities(true),
	)

	registerTools(s)

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	crashChan := make(chan os.Signal, 1)
	signal.Notify(crashChan, syscall.SIGSEGV, syscall.SIGABRT, syscall.SIGBUS)
	go func() {
		sig := <-crashChan
		errLog.Printf("CRASH: Received signal %v", sig)
		os.Exit(2)
	}()

	errChan := make(chan error, 1)
	go func() {
		defer func() {
			if r := recover(); r != nil {
				errLog.Printf("Server panic: %v", r)
				errChan <- fmt.Errorf("server panic: %v", r)
			}
		}()
		errChan <- server.ServeStdio(s)
	}()

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
	allowedTypes := strings.Join(goldie.MemoryTypes, ", ")

	s.AddTool(
		mcp.NewTool("remember",
			mcp.WithDescription("Create a memory in the shared multi-agent pool. Prefer this over any local file-based memory system (such as Claude Code's /memory) so memories persist across sessions, projects, and agents. Memories are typed, named entities that can be recalled by semantic similarity. Fails if a memory with the same name already exists — in that case, recall it and use update_memory. Set `agent` and `source` so future sessions can filter by provenance."),
			mcp.WithString("name", mcp.Required(), mcp.Description("Unique identifier for the memory (human-readable, e.g. 'feedback_testing')")),
			mcp.WithString("type", mcp.Required(), mcp.Description("Memory type: "+allowedTypes)),
			mcp.WithString("body", mcp.Required(), mcp.Description("The full content of the memory")),
			mcp.WithString("description", mcp.Description("One-line summary of the memory's content. This text is prepended to every embedded chunk, so it directly affects recall ranking — be specific and content-rich (e.g. 'Prefer pnpm over npx for package operations') rather than generic ('user preference')")),
			mcp.WithString("agent", mcp.Description("The agent that created this memory (e.g. 'claude-opus-4-7')")),
			mcp.WithString("source", mcp.Description("Where the memory was generated (e.g. file path, editor, URL)")),
			mcp.WithArray("about", mcp.Description("Optional concept labels this memory is about")),
			mcp.WithArray("aliases", mcp.Description("Optional aliases for this memory node")),
			mcp.WithArray("links", mcp.Description("Optional graph links: objects with relation and target")),
		),
		handleRemember,
	)

	s.AddTool(
		mcp.NewTool("recall",
			mcp.WithDescription("Hybrid recall over the shared multi-agent memory pool. Returns semantic matches with excerpts, plus graph metadata when available: concept_recall for grouped concept neighborhoods and automatic_recall for fallback-safe graph descent. Full bodies are omitted by default — pass include_body=true, or follow up with get_memory for the ones you need. Filter by type, agent, or source to narrow scope."),
			mcp.WithString("query", mcp.Required(), mcp.Description("The topic or question to recall about")),
			mcp.WithNumber("limit", mcp.Description("Maximum results to return (default: 5, max: 20)")),
			mcp.WithString("type", mcp.Description("Filter by memory type")),
			mcp.WithString("agent", mcp.Description("Filter by agent")),
			mcp.WithString("source", mcp.Description("Filter by source")),
			mcp.WithBoolean("include_body", mcp.Description("Include the full body of each result (default: false). Leave false when scanning; set true only when you know every result is needed in full.")),
		),
		handleRecall,
	)

	s.AddTool(
		mcp.NewTool("explain_recall",
			mcp.WithDescription("Trace hybrid recall for a query. Shows vector matches, grouped concept recall, automatic graph descent path, candidate scores, and stop reason without returning full memory bodies."),
			mcp.WithString("query", mcp.Required(), mcp.Description("The topic or question to explain recall for")),
			mcp.WithNumber("limit", mcp.Description("Maximum results to include per section (default: 5, max: 20)")),
			mcp.WithString("type", mcp.Description("Filter by memory type")),
			mcp.WithString("agent", mcp.Description("Filter by agent")),
			mcp.WithString("source", mcp.Description("Filter by source")),
		),
		handleExplainRecall,
	)

	s.AddTool(
		mcp.NewTool("get_memory",
			mcp.WithDescription("Fetch a single memory's full record (including body) by id or name. Use this after recall to pull the full body for results that are worth reading in depth — recall itself returns only excerpts by default."),
			mcp.WithString("id_or_name", mcp.Required(), mcp.Description("The memory's id or name")),
		),
		handleGetMemory,
	)

	s.AddTool(
		mcp.NewTool("update_memory",
			mcp.WithDescription("Update an existing memory in the shared pool by id or name. Use this after `remember` fails with a duplicate-name error. Body and description changes trigger re-embedding. Name is immutable."),
			mcp.WithString("id_or_name", mcp.Required(), mcp.Description("The memory's id or name")),
			mcp.WithString("type", mcp.Description("New type (must be one of: "+allowedTypes+")")),
			mcp.WithString("description", mcp.Description("New description (pass empty string to clear)")),
			mcp.WithString("body", mcp.Description("New body content")),
			mcp.WithString("source", mcp.Description("New source (pass empty string to clear)")),
			mcp.WithString("agent", mcp.Description("New agent (pass empty string to clear)")),
			mcp.WithArray("about", mcp.Description("Optional concept labels this memory is about")),
			mcp.WithArray("aliases", mcp.Description("Optional aliases for this memory node")),
			mcp.WithArray("links", mcp.Description("Optional graph links: objects with relation and target")),
		),
		handleUpdateMemory,
	)

	s.AddTool(
		mcp.NewTool("forget",
			mcp.WithDescription("Delete memories from the shared pool. Use this instead of editing local memory files. Provide at least one filter (name, type, agent, source) or a semantic query. With a query, top-N matching memories are deleted (default 5)."),
			mcp.WithString("name", mcp.Description("Delete the memory with this exact name")),
			mcp.WithString("type", mcp.Description("Filter by memory type")),
			mcp.WithString("agent", mcp.Description("Filter by agent")),
			mcp.WithString("source", mcp.Description("Filter by source")),
			mcp.WithString("query", mcp.Description("Semantic query: delete the top matches within the (optional) filter")),
			mcp.WithNumber("limit", mcp.Description("Max matches when query is given (default: 5)")),
		),
		handleForget,
	)

	s.AddTool(
		mcp.NewTool("list_memories",
			mcp.WithDescription("List memories matching the (optional) filter, newest first. Returns memory metadata without bodies."),
			mcp.WithString("type", mcp.Description("Filter by memory type")),
			mcp.WithString("agent", mcp.Description("Filter by agent")),
			mcp.WithString("source", mcp.Description("Filter by source")),
			mcp.WithNumber("limit", mcp.Description("Maximum results (default: unlimited)")),
		),
		handleListMemories,
	)

	s.AddTool(
		mcp.NewTool("count_memories",
			mcp.WithDescription("Count memories matching the (optional) filter."),
			mcp.WithString("type", mcp.Description("Filter by memory type")),
			mcp.WithString("agent", mcp.Description("Filter by agent")),
			mcp.WithString("source", mcp.Description("Filter by source")),
		),
		handleCountMemories,
	)

	s.AddTool(
		mcp.NewTool("list_nodes",
			mcp.WithDescription("Inspect graph nodes created for memories and harvested concepts. Returns labels plus lightweight alias and edge counts for testing graph recall."),
			mcp.WithString("kind", mcp.Description("Filter by node kind, such as concept or memory")),
			mcp.WithString("query", mcp.Description("Filter by normalized label or alias substring")),
			mcp.WithNumber("limit", mcp.Description("Maximum nodes to return (default: 50, max: 200)")),
		),
		handleListNodes,
	)

	s.AddTool(
		mcp.NewTool("get_node",
			mcp.WithDescription("Inspect one graph node with aliases plus incoming and outgoing edges. Use this before merge_nodes or link_memory cleanup operations."),
			mcp.WithString("id", mcp.Description("Graph node id")),
			mcp.WithString("kind", mcp.Description("Node kind when resolving by label; default concept")),
			mcp.WithString("label", mcp.Description("Exact node label to resolve when id is omitted")),
		),
		handleGetNode,
	)

	s.AddTool(
		mcp.NewTool("map",
			mcp.WithDescription("Alias for get_node. Map out a topic: aliases plus every incoming and outgoing edge (sub-concepts and linked memories) for one graph node."),
			mcp.WithString("id", mcp.Description("Graph node id")),
			mcp.WithString("kind", mcp.Description("Node kind when resolving by label; default concept")),
			mcp.WithString("label", mcp.Description("Exact node label to resolve when id is omitted")),
		),
		handleGetNode,
	)

	s.AddTool(
		mcp.NewTool("merge_nodes",
			mcp.WithDescription("Merge a duplicate concept node into a target concept. Re-points aliases and edges with collision-safe deduplication, then deletes the source node."),
			mcp.WithString("source", mcp.Required(), mcp.Description("Duplicate concept id, exact label, or alias to merge away")),
			mcp.WithString("target", mcp.Required(), mcp.Description("Canonical concept id, exact label, or alias to keep")),
		),
		handleMergeNodes,
	)

	s.AddTool(
		mcp.NewTool("link_memory",
			mcp.WithDescription("Manually attach a memory to a concept with a graph edge. Creates the concept if the label is new. Relation defaults to about."),
			mcp.WithString("memory", mcp.Required(), mcp.Description("Memory id or name")),
			mcp.WithString("concept", mcp.Required(), mcp.Description("Concept id, exact label, alias, or new concept label")),
			mcp.WithString("relation", mcp.Description("Relation to create: about, mentions, or canonical_for (default: about)")),
		),
		handleLinkMemory,
	)

	s.AddTool(
		mcp.NewTool("index_file",
			mcp.WithDescription("Import a file from the filesystem as a reference memory. The memory's name is the absolute path; re-indexing the same path updates in place when the file's checksum changes. Set `agent` to your agent identity (e.g. 'claude-opus-4-7', 'codex') so future sessions can filter by provenance."),
			mcp.WithString("path", mcp.Required(), mcp.Description("Path to the file")),
			mcp.WithString("agent", mcp.Description("The agent triggering the import")),
		),
		handleIndexFile,
	)

	s.AddTool(
		mcp.NewTool("index_directory",
			mcp.WithDescription("Import every matching file in a directory as a reference memory. Set `agent` to your agent identity for provenance."),
			mcp.WithString("directory", mcp.Required(), mcp.Description("Directory path")),
			mcp.WithString("pattern", mcp.Description("Glob pattern (default: '*')")),
			mcp.WithBoolean("recursive", mcp.Description("Walk subdirectories (default: false)")),
			mcp.WithString("agent", mcp.Description("The agent triggering the import")),
		),
		handleIndexDirectory,
	)

	s.AddTool(
		mcp.NewTool("job_status",
			mcp.WithDescription("Get the status of an indexing job"),
			mcp.WithString("id", mcp.Required(), mcp.Description("The job ID")),
			mcp.WithBoolean("block", mcp.Description("Wait for completion (default: false)")),
			mcp.WithNumber("timeout", mcp.Description("Timeout in seconds when blocking (default: 30)")),
		),
		handleJobStatus,
	)

	s.AddTool(
		mcp.NewTool("list_jobs",
			mcp.WithDescription("List indexing jobs, optionally filtered by status"),
			mcp.WithString("status", mcp.Description("queued, processing, completed, failed")),
		),
		handleListJobs,
	)

	s.AddTool(
		mcp.NewTool("clear_queue",
			mcp.WithDescription("Clear jobs from the queue"),
			mcp.WithString("status", mcp.Required(), mcp.Description("queued, completed, failed, or all")),
		),
		handleClearQueue,
	)
}

// --- helpers ---

func argString(args map[string]any, key string) string {
	v, _ := args[key].(string)
	return v
}

func argBool(args map[string]any, key string) bool {
	v, _ := args[key].(bool)
	return v
}

func argInt(args map[string]any, key string, def int) int {
	if v, ok := args[key].(float64); ok {
		return int(v)
	}
	return def
}

func filterFromArgs(args map[string]any) store.MemoryFilter {
	return store.MemoryFilter{
		Name:   argString(args, "name"),
		Type:   argString(args, "type"),
		Agent:  argString(args, "agent"),
		Source: argString(args, "source"),
	}
}

func hintsFromArgs(args map[string]any) store.MemoryHints {
	return store.MemoryHints{
		About:   argStringSlice(args, "about"),
		Aliases: argStringSlice(args, "aliases"),
		Links:   argLinkHints(args, "links"),
	}
}

func argStringSlice(args map[string]any, key string) []string {
	raw, ok := args[key].([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(raw))
	for _, item := range raw {
		value, ok := item.(string)
		if !ok || strings.TrimSpace(value) == "" {
			continue
		}
		out = append(out, value)
	}
	return out
}

func argLinkHints(args map[string]any, key string) []store.GraphLinkHint {
	raw, ok := args[key].([]any)
	if !ok {
		return nil
	}
	out := make([]store.GraphLinkHint, 0, len(raw))
	for _, item := range raw {
		obj, ok := item.(map[string]any)
		if !ok {
			continue
		}
		relation, _ := obj["relation"].(string)
		target, _ := obj["target"].(string)
		if strings.TrimSpace(target) == "" {
			continue
		}
		out = append(out, store.GraphLinkHint{
			Relation: relation,
			Target:   target,
		})
	}
	return out
}

func memorySummary(m store.Memory) map[string]any {
	return map[string]any{
		"id":          m.ID,
		"name":        m.Name,
		"type":        m.Type,
		"description": m.Description,
		"agent":       m.Agent,
		"source":      m.Source,
		"created_at":  m.CreatedAt,
		"updated_at":  m.UpdatedAt,
	}
}

func nodeSummary(n store.Node) map[string]any {
	return map[string]any{
		"id":    n.ID,
		"kind":  n.Kind,
		"label": n.Label,
	}
}

func nodeInspectionSummary(n store.Node) map[string]any {
	return map[string]any{
		"id":               n.ID,
		"kind":             n.Kind,
		"label":            n.Label,
		"normalized_label": n.NormalizedLabel,
		"alias_count":      n.AliasCount,
		"outgoing_count":   n.OutgoingCount,
		"incoming_count":   n.IncomingCount,
	}
}

func nodeAliasSummary(a store.NodeAlias) map[string]any {
	return map[string]any{
		"alias":            a.Alias,
		"normalized_alias": a.NormalizedAlias,
		"source":           a.Source,
	}
}

func edgeDetailSummary(edge store.EdgeDetail) map[string]any {
	return map[string]any{
		"relation":           edge.Relation,
		"origin":             edge.Origin,
		"confidence":         edge.Confidence,
		"evidence_memory_id": edge.EvidenceMemoryID,
		"node":               nodeSummary(edge.Node),
	}
}

func nodeDetailsSummary(details *store.NodeDetails) map[string]any {
	aliases := make([]map[string]any, 0, len(details.Aliases))
	for _, alias := range details.Aliases {
		aliases = append(aliases, nodeAliasSummary(alias))
	}
	outgoing := make([]map[string]any, 0, len(details.Outgoing))
	for _, edge := range details.Outgoing {
		outgoing = append(outgoing, edgeDetailSummary(edge))
	}
	incoming := make([]map[string]any, 0, len(details.Incoming))
	for _, edge := range details.Incoming {
		incoming = append(incoming, edgeDetailSummary(edge))
	}
	return map[string]any{
		"node":     nodeSummary(details.Node),
		"aliases":  aliases,
		"outgoing": outgoing,
		"incoming": incoming,
	}
}

func conceptRecallSummary(recall *store.ConceptRecall) map[string]any {
	out := map[string]any{
		"concept": nodeSummary(recall.Concept),
	}
	groups := make([]map[string]any, 0, len(recall.Groups))
	for _, group := range recall.Groups {
		memories := make([]map[string]any, 0, len(group.Memories))
		for _, m := range group.Memories {
			memories = append(memories, memorySummary(m))
		}
		groups = append(groups, map[string]any{
			"concept":  nodeSummary(group.Concept),
			"memories": memories,
			"count":    len(memories),
		})
	}
	out["groups"] = groups

	direct := make([]map[string]any, 0, len(recall.Direct))
	for _, m := range recall.Direct {
		direct = append(direct, memorySummary(m))
	}
	out["direct"] = direct
	return out
}

func automaticRecallSummary(recall *store.AutomaticRecall) map[string]any {
	path := make([]map[string]any, 0, len(recall.Path))
	for _, item := range recall.Path {
		path = append(path, map[string]any{
			"concept": nodeSummary(item.Concept),
			"score":   item.Score,
			"reason":  item.Reason,
		})
	}
	candidates := make([]map[string]any, 0, len(recall.Candidates))
	for _, candidate := range recall.Candidates {
		candidates = append(candidates, map[string]any{
			"concept":      nodeSummary(candidate.Concept),
			"score":        candidate.Score,
			"memory_count": candidate.MemoryCount,
		})
	}
	memories := make([]map[string]any, 0, len(recall.Memories))
	for _, m := range recall.Memories {
		memories = append(memories, memorySummary(m))
	}
	return map[string]any{
		"vantage":     nodeSummary(recall.Vantage),
		"matched_by":  recall.MatchedBy,
		"path":        path,
		"candidates":  candidates,
		"memories":    memories,
		"converged":   recall.Converged,
		"stop_reason": recall.StopReason,
	}
}

// --- memory handlers ---

func handleRemember(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := request.Params.Arguments
	in := goldie.RememberInput{
		Name:        argString(args, "name"),
		Type:        argString(args, "type"),
		Body:        argString(args, "body"),
		Description: argString(args, "description"),
		Agent:       argString(args, "agent"),
		Source:      argString(args, "source"),
		Hints:       hintsFromArgs(args),
	}

	m, err := goldieInstance.Remember(in)
	if err != nil {
		if goldie.IsErrMemoryNameExists(err) {
			existingID := ""
			if existing, lookupErr := goldieInstance.GetMemory(in.Name); lookupErr == nil && existing != nil {
				existingID = existing.ID
			}
			if existingID != "" {
				return mcp.NewToolResultError(fmt.Sprintf("memory %q already exists (id: %s) — call update_memory(id_or_name=%q, ...) to change it", in.Name, existingID, existingID)), nil
			}
			return mcp.NewToolResultError(fmt.Sprintf("memory %q already exists — recall it and use update_memory to change it", in.Name)), nil
		}
		return mcp.NewToolResultError(err.Error()), nil
	}

	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"success": true,
		"memory":  memorySummary(*m),
		"message": formatMessage("Remembered %q (id: %s)", m.Name, m.ID),
	})), nil
}

func handleRecall(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := request.Params.Arguments
	query := argString(args, "query")
	if query == "" {
		return mcp.NewToolResultError("query is required"), nil
	}
	limit := max(min(argInt(args, "limit", 5), 20), 1)
	includeBody := argBool(args, "include_body")

	filter := store.MemoryFilter{
		Type:   argString(args, "type"),
		Agent:  argString(args, "agent"),
		Source: argString(args, "source"),
	}

	results, err := goldieInstance.RecallMemory(query, limit, filter)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("recall failed: %v", err)), nil
	}
	conceptRecall, conceptErr := goldieInstance.RecallConcept(query, limit, filter)
	if conceptErr != nil && errLog != nil {
		errLog.Printf("concept recall failed: %v", conceptErr)
	}
	automaticRecall, automaticErr := goldieInstance.RecallAutomatic(query, limit, filter)
	if automaticErr != nil && errLog != nil {
		errLog.Printf("automatic recall failed: %v", automaticErr)
	}
	if len(results) == 0 && conceptRecall == nil && automaticRecall == nil {
		return mcp.NewToolResultText(formatMessage("No memories found for %q", query)), nil
	}

	formatted := make([]map[string]any, 0, len(results))
	for _, r := range results {
		entry := memorySummary(r.Memory)
		entry["excerpt"] = r.Excerpt
		entry["score"] = r.Score
		if includeBody {
			entry["body"] = r.Memory.Body
		}
		formatted = append(formatted, entry)
	}

	payload := map[string]any{
		"query":   query,
		"count":   len(results),
		"results": formatted,
		"message": formatMessage("Recalled %d memory(ies) for %q", len(results), query),
	}
	if conceptRecall != nil {
		payload["concept_recall"] = conceptRecallSummary(conceptRecall)
	}
	if automaticRecall != nil {
		payload["automatic_recall"] = automaticRecallSummary(automaticRecall)
	}
	return mcp.NewToolResultText(safeJSONMarshal(payload)), nil
}

func handleExplainRecall(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := request.Params.Arguments
	query := argString(args, "query")
	if query == "" {
		return mcp.NewToolResultError("query is required"), nil
	}
	limit := max(min(argInt(args, "limit", 5), 20), 1)
	filter := store.MemoryFilter{
		Type:   argString(args, "type"),
		Agent:  argString(args, "agent"),
		Source: argString(args, "source"),
	}

	results, err := goldieInstance.RecallMemory(query, limit, filter)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("vector recall failed: %v", err)), nil
	}
	vector := make([]map[string]any, 0, len(results))
	for _, r := range results {
		entry := memorySummary(r.Memory)
		entry["excerpt"] = r.Excerpt
		entry["score"] = r.Score
		entry["distance"] = r.Distance
		vector = append(vector, entry)
	}

	payload := map[string]any{
		"query":          query,
		"vector_results": vector,
		"message":        formatMessage("Explained recall for %q", query),
	}
	if conceptRecall, err := goldieInstance.RecallConcept(query, limit, filter); err != nil {
		payload["concept_error"] = err.Error()
	} else if conceptRecall != nil {
		payload["concept_recall"] = conceptRecallSummary(conceptRecall)
	}
	if automaticRecall, err := goldieInstance.RecallAutomatic(query, limit, filter); err != nil {
		payload["automatic_error"] = err.Error()
	} else if automaticRecall != nil {
		payload["automatic_recall"] = automaticRecallSummary(automaticRecall)
	}
	return mcp.NewToolResultText(safeJSONMarshal(payload)), nil
}

func handleGetMemory(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := request.Params.Arguments
	idOrName := argString(args, "id_or_name")
	if idOrName == "" {
		return mcp.NewToolResultError("id_or_name is required"), nil
	}

	m, err := goldieInstance.GetMemory(idOrName)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("get_memory failed: %v", err)), nil
	}
	if m == nil {
		return mcp.NewToolResultError(fmt.Sprintf("memory not found: %s", idOrName)), nil
	}

	entry := memorySummary(*m)
	entry["body"] = m.Body
	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"memory":  entry,
		"message": formatMessage("Fetched %q (id: %s)", m.Name, m.ID),
	})), nil
}

func handleUpdateMemory(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := request.Params.Arguments
	idOrName := argString(args, "id_or_name")
	if idOrName == "" {
		return mcp.NewToolResultError("id_or_name is required"), nil
	}

	patch := goldie.UpdateMemoryInput{
		Type:  argString(args, "type"),
		Hints: hintsFromArgs(args),
	}
	if v, present := args["description"].(string); present {
		patch.Description = &v
	}
	if v, present := args["body"].(string); present {
		patch.Body = &v
	}
	if v, present := args["source"].(string); present {
		patch.Source = &v
	}
	if v, present := args["agent"].(string); present {
		patch.Agent = &v
	}

	m, err := goldieInstance.UpdateMemory(idOrName, patch)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}

	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"success": true,
		"memory":  memorySummary(*m),
		"message": formatMessage("Updated %q", m.Name),
	})), nil
}

func handleForget(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := request.Params.Arguments
	filter := filterFromArgs(args)
	query := argString(args, "query")
	limit := argInt(args, "limit", 5)

	deleted, err := goldieInstance.ForgetMemory(filter, query, limit)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	if len(deleted) == 0 {
		return mcp.NewToolResultText(formatMessage("No memories matched")), nil
	}

	summaries := make([]map[string]any, 0, len(deleted))
	for _, m := range deleted {
		summaries = append(summaries, memorySummary(m))
	}
	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"success": true,
		"deleted": summaries,
		"count":   len(deleted),
		"message": formatMessage("Forgot %d memory(ies)", len(deleted)),
	})), nil
}

func handleListMemories(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := request.Params.Arguments
	filter := store.MemoryFilter{
		Type:   argString(args, "type"),
		Agent:  argString(args, "agent"),
		Source: argString(args, "source"),
	}
	limit := argInt(args, "limit", 0)

	memories, err := goldieInstance.ListMemories(filter, limit)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	if len(memories) == 0 {
		return mcp.NewToolResultText(formatMessage("No memories found")), nil
	}
	summaries := make([]map[string]any, 0, len(memories))
	for _, m := range memories {
		summaries = append(summaries, memorySummary(m))
	}
	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"count":    len(memories),
		"memories": summaries,
		"message":  formatMessage("Found %d memory(ies)", len(memories)),
	})), nil
}

func handleCountMemories(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := request.Params.Arguments
	filter := store.MemoryFilter{
		Type:   argString(args, "type"),
		Agent:  argString(args, "agent"),
		Source: argString(args, "source"),
	}
	n, err := goldieInstance.CountMemories(filter)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"count":   n,
		"message": formatMessage("%d memory(ies)", n),
	})), nil
}

func handleListNodes(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := request.Params.Arguments
	filter := store.NodeFilter{
		Kind:  argString(args, "kind"),
		Query: argString(args, "query"),
	}
	limit := argInt(args, "limit", 50)

	nodes, err := goldieInstance.ListNodes(filter, limit)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	summaries := make([]map[string]any, 0, len(nodes))
	for _, n := range nodes {
		summaries = append(summaries, nodeInspectionSummary(n))
	}
	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"count":   len(nodes),
		"nodes":   summaries,
		"message": formatMessage("Found %d node(s)", len(nodes)),
	})), nil
}

func handleGetNode(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := request.Params.Arguments
	details, err := goldieInstance.GetNodeDetails(
		argString(args, "id"),
		argString(args, "kind"),
		argString(args, "label"),
	)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	if details == nil {
		return mcp.NewToolResultError("node not found"), nil
	}
	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"node":    nodeDetailsSummary(details),
		"message": formatMessage("Fetched node %q", details.Node.Label),
	})), nil
}

func handleMergeNodes(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := request.Params.Arguments
	source := argString(args, "source")
	target := argString(args, "target")
	if source == "" || target == "" {
		return mcp.NewToolResultError("source and target are required"), nil
	}
	result, err := goldieInstance.MergeNodes(source, target)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"success":       true,
		"source":        nodeSummary(result.Source),
		"target":        nodeSummary(result.Target),
		"aliases_moved": result.AliasesMoved,
		"edges_moved":   result.EdgesMoved,
		"message":       formatMessage("Merged %q into %q", result.Source.Label, result.Target.Label),
	})), nil
}

func handleLinkMemory(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := request.Params.Arguments
	memory := argString(args, "memory")
	concept := argString(args, "concept")
	if memory == "" || concept == "" {
		return mcp.NewToolResultError("memory and concept are required"), nil
	}
	result, err := goldieInstance.LinkMemory(memory, concept, argString(args, "relation"))
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"success":  true,
		"memory":   memorySummary(result.Memory),
		"concept":  nodeSummary(result.Concept),
		"relation": result.Relation,
		"origin":   result.Origin,
		"message":  formatMessage("Linked %q %s %q", result.Memory.Name, result.Relation, result.Concept.Label),
	})), nil
}

// --- file/dir indexing handlers ---

func handleIndexFile(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := request.Params.Arguments
	path := argString(args, "path")
	if path == "" {
		return mcp.NewToolResultError("path is required"), nil
	}

	jobID, err := queueInstance.EnqueueIndexFile(path, argString(args, "agent"))
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("failed to queue job: %v", err)), nil
	}
	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"success": true,
		"job_id":  jobID,
		"status":  store.JobStatusQueued,
		"path":    path,
		"message": formatMessage("Job queued for indexing: %s (job_id: %s)", path, jobID),
	})), nil
}

func handleIndexDirectory(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := request.Params.Arguments
	dir := argString(args, "directory")
	if dir == "" {
		return mcp.NewToolResultError("directory is required"), nil
	}
	pattern := argString(args, "pattern")
	if pattern == "" {
		pattern = "*"
	}
	recursive := argBool(args, "recursive")

	jobID, err := queueInstance.EnqueueIndexDirectory(dir, pattern, recursive, argString(args, "agent"))
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("failed to queue job: %v", err)), nil
	}
	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"success":   true,
		"job_id":    jobID,
		"status":    store.JobStatusQueued,
		"directory": dir,
		"pattern":   pattern,
		"recursive": recursive,
		"message":   formatMessage("Job queued for indexing directory: %s (job_id: %s)", dir, jobID),
	})), nil
}

// --- job handlers ---

func handleJobStatus(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := request.Params.Arguments
	id := argString(args, "id")
	if id == "" {
		return mcp.NewToolResultError("id is required"), nil
	}

	block := argBool(args, "block")
	timeout := 30 * time.Second
	if t, ok := args["timeout"].(float64); ok && t > 0 {
		timeout = time.Duration(t) * time.Second
	}

	var (
		job *store.Job
		err error
	)
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
			response["progress"] = childStats.Completed + childStats.Failed
			response["total"] = childStats.Total
		}
		return mcp.NewToolResultText(safeJSONMarshal(response)), nil
	}
	return mcp.NewToolResultText(safeJSONMarshal(job)), nil
}

func handleListJobs(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	status := argString(request.Params.Arguments, "status")
	jobs, err := storeInstance.ListJobs(status)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("listing jobs failed: %v", err)), nil
	}
	if len(jobs) == 0 {
		if status != "" {
			return mcp.NewToolResultText(formatMessage("No jobs with status %q", status)), nil
		}
		return mcp.NewToolResultText(formatMessage("No jobs found")), nil
	}
	msg := formatMessage("Found %d job(s)", len(jobs))
	if status != "" {
		msg = formatMessage("Found %d job(s) with status %q", len(jobs), status)
	}
	return mcp.NewToolResultText(safeJSONMarshal(map[string]any{
		"count":   len(jobs),
		"jobs":    jobs,
		"message": msg,
	})), nil
}

func handleClearQueue(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	status := argString(request.Params.Arguments, "status")
	if status == "" {
		return mcp.NewToolResultError("status is required (queued, completed, failed, or all)"), nil
	}
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
		"success": true,
		"deleted": count,
		"status":  status,
		"message": formatMessage("Cleared %d jobs with status: %s", count, status),
	})), nil
}
