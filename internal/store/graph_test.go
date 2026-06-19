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

func TestMemoryNodeInspectionFindsMemoryName(t *testing.T) {
	st := newTestStore(t)

	m := addTestMemory(t, st, "goldie_v4_live_smoke_20260619_a7f9c2")

	nodes, err := st.ListNodes(NodeFilter{Kind: NodeKindMemory, Query: "goldie_v4_live_smoke"}, 5)
	if err != nil {
		t.Fatalf("ListNodes failed: %v", err)
	}
	if len(nodes) != 1 {
		t.Fatalf("expected 1 memory node, got %d", len(nodes))
	}
	if nodes[0].ID != m.NodeID {
		t.Fatalf("expected memory node %q, got %q", m.NodeID, nodes[0].ID)
	}

	details, err := st.GetNodeDetails("", NodeKindMemory, m.Name)
	if err != nil {
		t.Fatalf("GetNodeDetails failed: %v", err)
	}
	if details == nil {
		t.Fatal("expected memory node details")
	}
	if details.Node.ID != m.NodeID {
		t.Fatalf("expected memory node %q, got %q", m.NodeID, details.Node.ID)
	}
}

func TestAddMemoryEnqueuesGraphHarvestWithoutRunningIt(t *testing.T) {
	st := newTestStore(t)

	addTestMemory(t, st, "acme_cloud_auth")

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

	addTestMemory(t, st, "acme_cloud_passwordless_auth")
	addTestMemory(t, st, "acme_cloud_pricing")
	addTestMemory(t, st, "acme_deployment_modes")
	if err := st.RefreshHarvestedConcepts(); err != nil {
		t.Fatalf("RefreshHarvestedConcepts failed: %v", err)
	}

	recall, err := st.RecallConcept("Acme", 10, MemoryFilter{})
	if err != nil {
		t.Fatalf("RecallConcept failed: %v", err)
	}
	if recall == nil {
		t.Fatal("expected Acme concept recall")
	}
	if recall.Concept.Label != "Acme" {
		t.Fatalf("expected Acme concept, got %q", recall.Concept.Label)
	}
	if len(recall.Groups) != 1 {
		t.Fatalf("expected 1 facet group, got %d", len(recall.Groups))
	}
	if recall.Groups[0].Concept.Label != "Acme Cloud" {
		t.Fatalf("expected Acme Cloud facet, got %q", recall.Groups[0].Concept.Label)
	}
	if len(recall.Groups[0].Memories) != 2 {
		t.Fatalf("expected 2 Acme Cloud memories, got %d", len(recall.Groups[0].Memories))
	}
	if len(recall.Direct) != 1 {
		t.Fatalf("expected 1 direct Acme memory, got %d", len(recall.Direct))
	}
	if recall.Direct[0].Name != "acme_deployment_modes" {
		t.Fatalf("expected direct memory acme_deployment_modes, got %q", recall.Direct[0].Name)
	}
}

func TestHarvestStripsMemoryTypePrefix(t *testing.T) {
	st := newTestStore(t)

	m1 := &Memory{Name: "feedback_acme_auth_no_passwords", Type: "feedback", Body: "body"}
	if err := st.AddMemory(m1, []string{m1.Body}, [][]float32{{0.1, 0.2, 0.3}}); err != nil {
		t.Fatalf("AddMemory m1 failed: %v", err)
	}
	m2 := &Memory{Name: "project_acme_auth_policy", Type: "project", Body: "body"}
	if err := st.AddMemory(m2, []string{m2.Body}, [][]float32{{0.1, 0.2, 0.3}}); err != nil {
		t.Fatalf("AddMemory m2 failed: %v", err)
	}
	if err := st.RefreshHarvestedConcepts(); err != nil {
		t.Fatalf("RefreshHarvestedConcepts failed: %v", err)
	}

	recall, err := st.RecallConcept("Acme", 10, MemoryFilter{})
	if err != nil {
		t.Fatalf("RecallConcept failed: %v", err)
	}
	if recall == nil {
		t.Fatal("expected Acme concept recall")
	}
	if len(recall.Groups) != 1 {
		t.Fatalf("expected 1 facet group, got %d", len(recall.Groups))
	}
	if recall.Groups[0].Concept.Label != "Acme Auth" {
		t.Fatalf("expected Acme Auth facet, got %q", recall.Groups[0].Concept.Label)
	}
}

func TestHarvestFindsRootInsideNameWhenSeededByPath(t *testing.T) {
	st := newTestStore(t)

	for _, name := range []string{"note_acme_cloud_auth", "decision_acme_cloud_pricing"} {
		m := &Memory{
			Name:   name,
			Type:   "project",
			Body:   "body",
			Source: "/Users/srfrog/Documents/Business/Acme/context.md",
		}
		if err := st.AddMemory(m, []string{m.Body}, [][]float32{{0.1, 0.2, 0.3}}); err != nil {
			t.Fatalf("AddMemory %s failed: %v", name, err)
		}
	}
	if err := st.RefreshHarvestedConcepts(); err != nil {
		t.Fatalf("RefreshHarvestedConcepts failed: %v", err)
	}

	recall, err := st.RecallConcept("Acme", 10, MemoryFilter{})
	if err != nil {
		t.Fatalf("RecallConcept failed: %v", err)
	}
	if recall == nil {
		t.Fatal("expected Acme concept recall")
	}
	if len(recall.Groups) != 1 {
		t.Fatalf("expected 1 facet group, got %d", len(recall.Groups))
	}
	if recall.Groups[0].Concept.Label != "Acme Cloud" {
		t.Fatalf("expected Acme Cloud facet, got %q", recall.Groups[0].Concept.Label)
	}
}

func TestHarvestReattachesFromRootToPromotedFacet(t *testing.T) {
	st := newTestStore(t)

	m1 := &Memory{
		Name:   "note_acme_cloud_auth",
		Type:   "project",
		Body:   "body",
		Source: "/Users/srfrog/Documents/Business/Acme/context.md",
	}
	if err := st.AddMemory(m1, []string{m1.Body}, [][]float32{{0.1, 0.2, 0.3}}); err != nil {
		t.Fatalf("AddMemory m1 failed: %v", err)
	}
	if err := st.RefreshHarvestedConcepts(); err != nil {
		t.Fatalf("first RefreshHarvestedConcepts failed: %v", err)
	}

	recall, err := st.RecallConcept("Acme", 10, MemoryFilter{})
	if err != nil {
		t.Fatalf("first RecallConcept failed: %v", err)
	}
	if len(recall.Direct) != 1 {
		t.Fatalf("expected first memory attached to root, got %d direct", len(recall.Direct))
	}

	m2 := &Memory{
		Name:   "decision_acme_cloud_pricing",
		Type:   "project",
		Body:   "body",
		Source: "/Users/srfrog/Documents/Business/Acme/context.md",
	}
	if err := st.AddMemory(m2, []string{m2.Body}, [][]float32{{0.1, 0.2, 0.3}}); err != nil {
		t.Fatalf("AddMemory m2 failed: %v", err)
	}
	if err := st.RefreshHarvestedConcepts(); err != nil {
		t.Fatalf("second RefreshHarvestedConcepts failed: %v", err)
	}

	recall, err = st.RecallConcept("Acme", 10, MemoryFilter{})
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
		Name:   "note_acme_cloud_auth",
		Type:   "project",
		Body:   "body",
		Source: "/Users/srfrog/Documents/Business/Acme/context.md",
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

	addTestMemory(t, st, "feedback_acme_auth_no_passwords")
	addTestMemory(t, st, "project_acme_auth_policy")
	addTestMemory(t, st, "acme_cloud_passwordless")
	addTestMemory(t, st, "acme_cloud_pricing")
	if err := st.RefreshHarvestedConcepts(); err != nil {
		t.Fatalf("RefreshHarvestedConcepts failed: %v", err)
	}

	recall, err := st.RecallAutomatic("how does Acme handle auth", nil, 5, MemoryFilter{})
	if err != nil {
		t.Fatalf("RecallAutomatic failed: %v", err)
	}
	if recall == nil {
		t.Fatal("expected automatic recall")
	}
	if !recall.Converged {
		t.Fatalf("expected converged recall, stop_reason=%s", recall.StopReason)
	}
	if recall.Vantage.Label != "Acme Auth" {
		t.Fatalf("expected Acme Auth vantage, got %q", recall.Vantage.Label)
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

	addTestMemory(t, st, "feedback_acme_auth_no_passwords")
	addTestMemory(t, st, "project_acme_auth_policy")
	addTestMemory(t, st, "acme_cloud_passwordless")
	addTestMemory(t, st, "acme_cloud_pricing")
	if err := st.RefreshHarvestedConcepts(); err != nil {
		t.Fatalf("RefreshHarvestedConcepts failed: %v", err)
	}

	recall, err := st.RecallAutomatic("Acme", nil, 5, MemoryFilter{})
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

func TestAutomaticRecallUsesFuzzyConceptEmbedding(t *testing.T) {
	st := newTestStore(t)

	addTestMemory(t, st, "feedback_acme_auth_no_passwords")
	addTestMemory(t, st, "project_acme_auth_policy")
	addTestMemory(t, st, "acme_cloud_passwordless")
	addTestMemory(t, st, "acme_cloud_pricing")
	if err := st.RefreshHarvestedConcepts(); err != nil {
		t.Fatalf("RefreshHarvestedConcepts failed: %v", err)
	}

	concepts, err := st.ListNodes(NodeFilter{Kind: NodeKindConcept, Query: "Acme"}, 20)
	if err != nil {
		t.Fatalf("ListNodes failed: %v", err)
	}
	conceptIDs := map[string]string{}
	for _, concept := range concepts {
		conceptIDs[concept.Label] = concept.ID
	}
	if conceptIDs["Acme Auth"] == "" || conceptIDs["Acme Cloud"] == "" {
		t.Fatalf("expected auth and cloud concepts, got %v", conceptIDs)
	}

	err = st.ReplaceNodeEmbeddings([]NodeEmbedding{
		{ID: conceptIDs["Acme Auth"], Embedding: []float32{0.1, 0.2, 0.3}},
		{ID: conceptIDs["Acme Cloud"], Embedding: []float32{0.9, 0.8, 0.7}},
	})
	if err != nil {
		t.Fatalf("ReplaceNodeEmbeddings failed: %v", err)
	}

	recall, err := st.RecallAutomatic("authentication", []float32{0.1, 0.2, 0.3}, 5, MemoryFilter{})
	if err != nil {
		t.Fatalf("RecallAutomatic failed: %v", err)
	}
	if recall == nil {
		t.Fatal("expected fuzzy automatic recall")
	}
	if recall.MatchedBy != "fuzzy_vector" {
		t.Fatalf("expected fuzzy_vector match, got %q", recall.MatchedBy)
	}
	if recall.Vantage.Label != "Acme Auth" {
		t.Fatalf("expected Acme Auth vantage, got %q", recall.Vantage.Label)
	}
	if !recall.Converged {
		t.Fatalf("expected fuzzy recall to converge, stop_reason=%s", recall.StopReason)
	}
}

func TestMergeConceptNodesDeduplicatesEdgeCollisions(t *testing.T) {
	st := newTestStore(t)

	m := addTestMemory(t, st, "acme_cloud_platform_auth")
	tx, err := st.db.Begin()
	if err != nil {
		t.Fatalf("begin failed: %v", err)
	}
	targetID, err := ensureConceptNodeTx(tx, "Acme Cloud")
	if err != nil {
		tx.Rollback()
		t.Fatalf("ensure target failed: %v", err)
	}
	sourceID, err := ensureConceptNodeTx(tx, "Acme Cloud Platform")
	if err != nil {
		tx.Rollback()
		t.Fatalf("ensure source failed: %v", err)
	}
	if err := ensureAliasTx(tx, sourceID, "acme-cloud-platform", "test"); err != nil {
		tx.Rollback()
		t.Fatalf("ensure alias failed: %v", err)
	}
	if err := ensureEdgeWithOriginTx(tx, m.NodeID, RelationAbout, targetID, m.ID, "manual"); err != nil {
		tx.Rollback()
		t.Fatalf("ensure target edge failed: %v", err)
	}
	if err := ensureEdgeWithOriginTx(tx, m.NodeID, RelationAbout, sourceID, m.ID, "manual"); err != nil {
		tx.Rollback()
		t.Fatalf("ensure source edge failed: %v", err)
	}
	if err := tx.Commit(); err != nil {
		t.Fatalf("commit failed: %v", err)
	}

	result, err := st.MergeConceptNodes("Acme Cloud Platform", "Acme Cloud")
	if err != nil {
		t.Fatalf("MergeConceptNodes failed: %v", err)
	}
	if result.Target.Label != "Acme Cloud" {
		t.Fatalf("expected target Acme Cloud, got %q", result.Target.Label)
	}
	if source, err := st.GetNodeDetails(sourceID, "", ""); err != nil {
		t.Fatalf("GetNodeDetails source failed: %v", err)
	} else if source != nil {
		t.Fatalf("expected source node removed, got %v", source)
	}

	var edgeCount int
	if err := st.db.QueryRow(`
		SELECT COUNT(*)
		FROM edges
		WHERE src_node_id = ? AND dst_node_id = ? AND relation = ?
	`, m.NodeID, targetID, RelationAbout).Scan(&edgeCount); err != nil {
		t.Fatalf("count edges failed: %v", err)
	}
	if edgeCount != 1 {
		t.Fatalf("expected one deduped target edge, got %d", edgeCount)
	}

	details, err := st.GetNodeDetails("", NodeKindConcept, "Acme Cloud")
	if err != nil {
		t.Fatalf("GetNodeDetails failed: %v", err)
	}
	foundAlias := false
	for _, alias := range details.Aliases {
		if alias.NormalizedAlias == "acme cloud platform" {
			foundAlias = true
			break
		}
	}
	if !foundAlias {
		t.Fatalf("expected merged source label alias, got %v", details.Aliases)
	}
}

func TestLinkMemoryToConceptCreatesManualAboutEdge(t *testing.T) {
	st := newTestStore(t)

	m := addTestMemory(t, st, "manual_auth_note")
	result, err := st.LinkMemoryToConcept(m.ID, "Auth", RelationAbout, "manual")
	if err != nil {
		t.Fatalf("LinkMemoryToConcept failed: %v", err)
	}
	if result.Concept.Label != "Auth" {
		t.Fatalf("expected Auth concept, got %q", result.Concept.Label)
	}

	recall, err := st.RecallConcept("Auth", 5, MemoryFilter{})
	if err != nil {
		t.Fatalf("RecallConcept failed: %v", err)
	}
	if recall == nil || len(recall.Direct) != 1 {
		t.Fatalf("expected direct Auth recall, got %v", recall)
	}
	if recall.Direct[0].Name != "manual_auth_note" {
		t.Fatalf("expected manual_auth_note, got %q", recall.Direct[0].Name)
	}
}

func TestApplyMemoryHintsSurviveHarvest(t *testing.T) {
	st := newTestStore(t)

	m := addTestMemory(t, st, "acme_cloud_passwordless_auth")
	_, err := st.ApplyMemoryHints(m.ID, MemoryHints{
		About:   []string{"Auth"},
		Aliases: []string{"passwordless auth note"},
		Links: []GraphLinkHint{{
			Relation: RelationMentions,
			Target:   "Passwordless Auth",
		}},
	})
	if err != nil {
		t.Fatalf("ApplyMemoryHints failed: %v", err)
	}
	if err := st.RefreshHarvestedConcepts(); err != nil {
		t.Fatalf("RefreshHarvestedConcepts failed: %v", err)
	}

	authRecall, err := st.RecallConcept("Auth", 5, MemoryFilter{})
	if err != nil {
		t.Fatalf("RecallConcept Auth failed: %v", err)
	}
	if authRecall == nil || len(authRecall.Direct) != 1 {
		t.Fatalf("expected hint Auth recall, got %v", authRecall)
	}

	details, err := st.GetNodeDetails(m.NodeID, "", "")
	if err != nil {
		t.Fatalf("GetNodeDetails failed: %v", err)
	}
	foundAlias := false
	foundMention := false
	for _, alias := range details.Aliases {
		if alias.NormalizedAlias == "passwordless auth note" && alias.Source == "hint" {
			foundAlias = true
		}
	}
	for _, edge := range details.Outgoing {
		if edge.Relation == RelationMentions && edge.Origin == "hint" && edge.Node.Label == "Passwordless Auth" {
			foundMention = true
		}
	}
	if !foundAlias {
		t.Fatalf("expected hint alias, got %v", details.Aliases)
	}
	if !foundMention {
		t.Fatalf("expected hint mentions edge, got %v", details.Outgoing)
	}
}
