package store

import (
	"path/filepath"
	"strings"
	"testing"
)

func newTestStore(t *testing.T) *Store {
	t.Helper()

	st, err := New(filepath.Join(t.TempDir(), "test.db"), 3, "")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	t.Cleanup(func() {
		if err := st.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	})
	return st
}

func addTestMemory(t *testing.T, st *Store, name string) *Memory {
	t.Helper()

	m := &Memory{
		Name: name,
		Type: "reference",
		Body: "body for " + name,
	}
	if err := st.AddMemory(m, []string{m.Body}, [][]float32{{0.1, 0.2, 0.3}}); err != nil {
		t.Fatalf("AddMemory failed: %v", err)
	}
	return m
}

func TestGraphSchemaInitIdempotent(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")

	st, err := New(dbPath, 3, "")
	if err != nil {
		t.Fatalf("first New failed: %v", err)
	}
	if err := st.Close(); err != nil {
		t.Fatalf("first Close failed: %v", err)
	}

	st, err = New(dbPath, 3, "")
	if err != nil {
		t.Fatalf("second New failed: %v", err)
	}
	if err := st.Close(); err != nil {
		t.Fatalf("second Close failed: %v", err)
	}
}

func TestAddMemoryCreatesMemoryNode(t *testing.T) {
	st := newTestStore(t)

	m := addTestMemory(t, st, "alpha")
	if m.NodeID == "" {
		t.Fatal("expected AddMemory to populate NodeID")
	}

	fetched, err := st.GetMemoryByName("alpha")
	if err != nil {
		t.Fatalf("GetMemoryByName failed: %v", err)
	}
	if fetched.NodeID == "" {
		t.Fatal("expected fetched memory to have NodeID")
	}
	if fetched.NodeID != m.NodeID {
		t.Fatalf("expected node %q, got %q", m.NodeID, fetched.NodeID)
	}

	nodes, err := st.CountNodes(NodeKindMemory)
	if err != nil {
		t.Fatalf("CountNodes failed: %v", err)
	}
	if nodes != 1 {
		t.Fatalf("expected 1 memory node, got %d", nodes)
	}
}

func TestAddMemoryEnqueuesGraphHarvestWithoutRunningIt(t *testing.T) {
	st := newTestStore(t)

	addTestMemory(t, st, "tuplia_cloud_auth")

	jobs, err := st.ListJobs(JobStatusQueued)
	if err != nil {
		t.Fatalf("ListJobs failed: %v", err)
	}
	foundHarvest := false
	for _, job := range jobs {
		if job.Type == JobTypeGraphHarvest {
			foundHarvest = true
			break
		}
	}
	if !foundHarvest {
		t.Fatal("expected AddMemory to enqueue graph_harvest")
	}

	concepts, err := st.CountNodes(NodeKindConcept)
	if err != nil {
		t.Fatalf("CountNodes failed: %v", err)
	}
	if concepts != 0 {
		t.Fatalf("expected no synchronous concept harvest, got %d concept nodes", concepts)
	}
}

func TestDeleteMemoryRemovesMemoryNode(t *testing.T) {
	st := newTestStore(t)

	m := addTestMemory(t, st, "delete_me")
	ok, err := st.DeleteMemoryByID(m.ID)
	if err != nil {
		t.Fatalf("DeleteMemoryByID failed: %v", err)
	}
	if !ok {
		t.Fatal("expected memory to be deleted")
	}

	nodes, err := st.CountNodes(NodeKindMemory)
	if err != nil {
		t.Fatalf("CountNodes failed: %v", err)
	}
	if nodes != 0 {
		t.Fatalf("expected memory node to be deleted, got %d nodes", nodes)
	}
}

func TestBackfillMemoryNodesBatch(t *testing.T) {
	st := newTestStore(t)

	m := addTestMemory(t, st, "needs_backfill")
	if _, err := st.db.Exec("UPDATE memories SET node_id = NULL WHERE id = ?", m.ID); err != nil {
		t.Fatalf("clearing node_id failed: %v", err)
	}

	missing, err := st.CountMemoriesMissingNodes()
	if err != nil {
		t.Fatalf("CountMemoriesMissingNodes failed: %v", err)
	}
	if missing != 1 {
		t.Fatalf("expected 1 missing node, got %d", missing)
	}

	n, err := st.BackfillMemoryNodesBatch(100)
	if err != nil {
		t.Fatalf("BackfillMemoryNodesBatch failed: %v", err)
	}
	if n != 1 {
		t.Fatalf("expected 1 backfilled memory, got %d", n)
	}

	fetched, err := st.GetMemory(m.ID)
	if err != nil {
		t.Fatalf("GetMemory failed: %v", err)
	}
	if fetched.NodeID == "" {
		t.Fatal("expected backfill to restore NodeID")
	}
}

func TestEnqueueGraphBackfillIfNeeded(t *testing.T) {
	st := newTestStore(t)

	m := addTestMemory(t, st, "queued_backfill")
	if _, err := st.db.Exec("UPDATE memories SET node_id = NULL WHERE id = ?", m.ID); err != nil {
		t.Fatalf("clearing node_id failed: %v", err)
	}

	if err := st.EnqueueGraphBackfillIfNeeded(); err != nil {
		t.Fatalf("EnqueueGraphBackfillIfNeeded failed: %v", err)
	}

	jobs, err := st.ListJobs(JobStatusQueued)
	if err != nil {
		t.Fatalf("ListJobs failed: %v", err)
	}
	found := false
	for _, job := range jobs {
		if job.Type == JobTypeGraphBackfill {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected queued graph backfill job")
	}
}

func TestHarvestedConceptRecallGroupsFacets(t *testing.T) {
	st := newTestStore(t)

	addTestMemory(t, st, "tuplia_cloud_passwordless_auth")
	addTestMemory(t, st, "tuplia_cloud_pricing")
	addTestMemory(t, st, "tuplia_deployment_modes")
	if err := st.RefreshHarvestedConcepts(); err != nil {
		t.Fatalf("RefreshHarvestedConcepts failed: %v", err)
	}

	recall, err := st.RecallConcept("Tuplia", 10, MemoryFilter{})
	if err != nil {
		t.Fatalf("RecallConcept failed: %v", err)
	}
	if recall == nil {
		t.Fatal("expected Tuplia concept recall")
	}
	if recall.Concept.Label != "Tuplia" {
		t.Fatalf("expected Tuplia concept, got %q", recall.Concept.Label)
	}
	if len(recall.Groups) != 1 {
		t.Fatalf("expected 1 facet group, got %d", len(recall.Groups))
	}
	if recall.Groups[0].Concept.Label != "Tuplia Cloud" {
		t.Fatalf("expected Tuplia Cloud facet, got %q", recall.Groups[0].Concept.Label)
	}
	if len(recall.Groups[0].Memories) != 2 {
		t.Fatalf("expected 2 Tuplia Cloud memories, got %d", len(recall.Groups[0].Memories))
	}
	if len(recall.Direct) != 1 {
		t.Fatalf("expected 1 direct Tuplia memory, got %d", len(recall.Direct))
	}
	if recall.Direct[0].Name != "tuplia_deployment_modes" {
		t.Fatalf("expected direct memory tuplia_deployment_modes, got %q", recall.Direct[0].Name)
	}
}

func TestHarvestStripsMemoryTypePrefix(t *testing.T) {
	st := newTestStore(t)

	m1 := &Memory{Name: "feedback_tuplia_auth_no_passwords", Type: "feedback", Body: "body"}
	if err := st.AddMemory(m1, []string{m1.Body}, [][]float32{{0.1, 0.2, 0.3}}); err != nil {
		t.Fatalf("AddMemory m1 failed: %v", err)
	}
	m2 := &Memory{Name: "project_tuplia_auth_policy", Type: "project", Body: "body"}
	if err := st.AddMemory(m2, []string{m2.Body}, [][]float32{{0.1, 0.2, 0.3}}); err != nil {
		t.Fatalf("AddMemory m2 failed: %v", err)
	}
	if err := st.RefreshHarvestedConcepts(); err != nil {
		t.Fatalf("RefreshHarvestedConcepts failed: %v", err)
	}

	recall, err := st.RecallConcept("Tuplia", 10, MemoryFilter{})
	if err != nil {
		t.Fatalf("RecallConcept failed: %v", err)
	}
	if recall == nil {
		t.Fatal("expected Tuplia concept recall")
	}
	if len(recall.Groups) != 1 {
		t.Fatalf("expected 1 facet group, got %d", len(recall.Groups))
	}
	if recall.Groups[0].Concept.Label != "Tuplia Auth" {
		t.Fatalf("expected Tuplia Auth facet, got %q", recall.Groups[0].Concept.Label)
	}
}

func TestHarvestFindsRootInsideNameWhenSeededByPath(t *testing.T) {
	st := newTestStore(t)

	for _, name := range []string{"note_tuplia_cloud_auth", "decision_tuplia_cloud_pricing"} {
		m := &Memory{
			Name:   name,
			Type:   "project",
			Body:   "body",
			Source: "/Users/srfrog/Documents/Business/Tuplia/context.md",
		}
		if err := st.AddMemory(m, []string{m.Body}, [][]float32{{0.1, 0.2, 0.3}}); err != nil {
			t.Fatalf("AddMemory %s failed: %v", name, err)
		}
	}
	if err := st.RefreshHarvestedConcepts(); err != nil {
		t.Fatalf("RefreshHarvestedConcepts failed: %v", err)
	}

	recall, err := st.RecallConcept("Tuplia", 10, MemoryFilter{})
	if err != nil {
		t.Fatalf("RecallConcept failed: %v", err)
	}
	if recall == nil {
		t.Fatal("expected Tuplia concept recall")
	}
	if len(recall.Groups) != 1 {
		t.Fatalf("expected 1 facet group, got %d", len(recall.Groups))
	}
	if recall.Groups[0].Concept.Label != "Tuplia Cloud" {
		t.Fatalf("expected Tuplia Cloud facet, got %q", recall.Groups[0].Concept.Label)
	}
}

func TestHarvestReattachesFromRootToPromotedFacet(t *testing.T) {
	st := newTestStore(t)

	m1 := &Memory{
		Name:   "note_tuplia_cloud_auth",
		Type:   "project",
		Body:   "body",
		Source: "/Users/srfrog/Documents/Business/Tuplia/context.md",
	}
	if err := st.AddMemory(m1, []string{m1.Body}, [][]float32{{0.1, 0.2, 0.3}}); err != nil {
		t.Fatalf("AddMemory m1 failed: %v", err)
	}
	if err := st.RefreshHarvestedConcepts(); err != nil {
		t.Fatalf("first RefreshHarvestedConcepts failed: %v", err)
	}

	recall, err := st.RecallConcept("Tuplia", 10, MemoryFilter{})
	if err != nil {
		t.Fatalf("first RecallConcept failed: %v", err)
	}
	if len(recall.Direct) != 1 {
		t.Fatalf("expected first memory attached to root, got %d direct", len(recall.Direct))
	}

	m2 := &Memory{
		Name:   "decision_tuplia_cloud_pricing",
		Type:   "project",
		Body:   "body",
		Source: "/Users/srfrog/Documents/Business/Tuplia/context.md",
	}
	if err := st.AddMemory(m2, []string{m2.Body}, [][]float32{{0.1, 0.2, 0.3}}); err != nil {
		t.Fatalf("AddMemory m2 failed: %v", err)
	}
	if err := st.RefreshHarvestedConcepts(); err != nil {
		t.Fatalf("second RefreshHarvestedConcepts failed: %v", err)
	}

	recall, err = st.RecallConcept("Tuplia", 10, MemoryFilter{})
	if err != nil {
		t.Fatalf("second RecallConcept failed: %v", err)
	}
	if len(recall.Direct) != 0 {
		t.Fatalf("expected promoted memories removed from direct root recall, got %d", len(recall.Direct))
	}
	if len(recall.Groups) != 1 {
		t.Fatalf("expected 1 facet group, got %d", len(recall.Groups))
	}
	if len(recall.Groups[0].Memories) != 2 {
		t.Fatalf("expected 2 memories in promoted facet, got %d", len(recall.Groups[0].Memories))
	}
}

func TestHarvestReattachKeepsHintOriginAboutEdges(t *testing.T) {
	st := newTestStore(t)

	m := &Memory{
		Name:   "note_tuplia_cloud_auth",
		Type:   "project",
		Body:   "body",
		Source: "/Users/srfrog/Documents/Business/Tuplia/context.md",
	}
	if err := st.AddMemory(m, []string{m.Body}, [][]float32{{0.1, 0.2, 0.3}}); err != nil {
		t.Fatalf("AddMemory failed: %v", err)
	}
	if err := st.RefreshHarvestedConcepts(); err != nil {
		t.Fatalf("RefreshHarvestedConcepts failed: %v", err)
	}

	tx, err := st.db.Begin()
	if err != nil {
		t.Fatalf("begin failed: %v", err)
	}
	authNodeID, err := ensureConceptNodeTx(tx, "Auth")
	if err != nil {
		tx.Rollback()
		t.Fatalf("ensureConceptNodeTx failed: %v", err)
	}
	if err := ensureEdgeWithOriginTx(tx, m.NodeID, RelationAbout, authNodeID, m.ID, "hint"); err != nil {
		tx.Rollback()
		t.Fatalf("ensureEdgeWithOriginTx failed: %v", err)
	}
	if err := tx.Commit(); err != nil {
		t.Fatalf("commit failed: %v", err)
	}

	if err := st.RefreshHarvestedConcepts(); err != nil {
		t.Fatalf("second RefreshHarvestedConcepts failed: %v", err)
	}

	var hintEdges int
	if err := st.db.QueryRow(`
		SELECT COUNT(*)
		FROM edges
		WHERE src_node_id = ? AND dst_node_id = ? AND relation = ? AND origin = ?
	`, m.NodeID, authNodeID, RelationAbout, "hint").Scan(&hintEdges); err != nil {
		t.Fatalf("count hint edges failed: %v", err)
	}
	if hintEdges != 1 {
		t.Fatalf("expected hint edge to survive harvest, got %d", hintEdges)
	}
}

func TestAutomaticRecallConvergesOnContainedConcept(t *testing.T) {
	st := newTestStore(t)

	addTestMemory(t, st, "feedback_tuplia_auth_no_passwords")
	addTestMemory(t, st, "project_tuplia_auth_policy")
	addTestMemory(t, st, "tuplia_cloud_passwordless")
	addTestMemory(t, st, "tuplia_cloud_pricing")
	if err := st.RefreshHarvestedConcepts(); err != nil {
		t.Fatalf("RefreshHarvestedConcepts failed: %v", err)
	}

	recall, err := st.RecallAutomatic("how does Tuplia handle auth", 5, MemoryFilter{})
	if err != nil {
		t.Fatalf("RecallAutomatic failed: %v", err)
	}
	if recall == nil {
		t.Fatal("expected automatic recall")
	}
	if !recall.Converged {
		t.Fatalf("expected converged recall, stop_reason=%s", recall.StopReason)
	}
	if recall.Vantage.Label != "Tuplia Auth" {
		t.Fatalf("expected Tuplia Auth vantage, got %q", recall.Vantage.Label)
	}
	if len(recall.Memories) == 0 {
		t.Fatal("expected automatic recall memories")
	}
	if !strings.Contains(recall.Memories[0].Name, "auth") {
		t.Fatalf("expected auth memory first, got %q", recall.Memories[0].Name)
	}
}

func TestAutomaticRecallFallsBackOnFlatGradient(t *testing.T) {
	st := newTestStore(t)

	addTestMemory(t, st, "feedback_tuplia_auth_no_passwords")
	addTestMemory(t, st, "project_tuplia_auth_policy")
	addTestMemory(t, st, "tuplia_cloud_passwordless")
	addTestMemory(t, st, "tuplia_cloud_pricing")
	if err := st.RefreshHarvestedConcepts(); err != nil {
		t.Fatalf("RefreshHarvestedConcepts failed: %v", err)
	}

	recall, err := st.RecallAutomatic("Tuplia", 5, MemoryFilter{})
	if err != nil {
		t.Fatalf("RecallAutomatic failed: %v", err)
	}
	if recall == nil {
		t.Fatal("expected automatic recall")
	}
	if recall.Converged {
		t.Fatalf("expected fallback, got converged path %v", recall.Path)
	}
	if recall.StopReason != "flat_gradient" {
		t.Fatalf("expected flat_gradient, got %q", recall.StopReason)
	}
	if len(recall.Candidates) < 2 {
		t.Fatalf("expected competing branch candidates, got %d", len(recall.Candidates))
	}
}
