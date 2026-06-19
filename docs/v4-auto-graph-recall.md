# Goldie v4: Graph Recall

## Problem

Goldie v3 stores memories well, but agents still have to make good retrieval choices.
In production, agents often create long names, overloaded descriptions, and duplicate
topic-specific memories. Later recall works only when semantic search happens to find
the right chunk or when the agent guesses the same wording.

The server should absorb more of that work. Agents should be able to store a simple
memory, and Goldie should attach it to stable concepts that can be recalled by aliases,
related terms, and graph proximity.

Grouped graph recall is the dependable v4 core: `recall "Tuplia"` should return a
clean concept neighborhood with facets and high-signal memories. Automatic recall is
the additive frontier on top. If the gradient is clear, Goldie can descend from a
broad or paraphrased query to a specific memory; if the gradient is ambiguous, v4
still ships useful grouped recall rather than pretending precision.

## Goal

Every memory stored in Goldie v4 gets a graph node. Goldie also maintains concept
nodes such as products, projects, repositories, people, topics, files, and decisions.
Memories link to those concepts through typed edges.

The user-facing behavior should stay simple:

- `remember` still accepts plain memory text.
- `recall "Tuplia"` resolves `Tuplia` as a concept, not just a vector query.
- `recall "Tuplia"` returns the concept's facets (`Tuplia Cloud`, `Tuplia
  Architecture`, `Tuplia Pricing`, ...) grouped, not just a flat list of memories.
- Memory names become less important as retrieval handles, but the descriptive
  names agents invent are harvested into the concept hierarchy instead of being
  lost between sessions.
- A query like `how does Tuplia handle auth` can auto-resolve to the Tuplia vantage,
  descend through the Tuplia Cloud/auth branch, and return the strongest specific
  memory plus the path it took when the gradient is clear.
- Ambiguous queries fall back to flat recall and say they did not converge.
- Duplicate or near-duplicate memories can be detected through shared concepts.
- Agents can optionally provide hints, but they should not have to manually maintain
  the graph.

## Non-Goals

- Do not build a general-purpose knowledge graph engine.
- Do not require agents to supply triples or ontology terms.
- Do not replace vector search.
- Do not make `name` the primary retrieval mechanism.
- Do not make `name` optional in v4. It remains `UNIQUE NOT NULL` and
  stays the handle for `update_memory`, `forget`, and the no-upsert dedup contract
  until graph recall is proven.
- Do not expose graph complexity in the basic MCP tool workflow.
- Do not make recall a black box. Automatic recall must be inspectable.
- Do not put a model call on the hot path. Automatic descent is deterministic graph
  traversal over existing vector scores and edge weights. Optional model re-rank can
  sit beside it later, never under it.
- Do not auto-descend through a flat gradient. A confident wrong point is worse than
  an honest neighborhood.

## Release Boundary

The v4.0 launch boundary is grouped graph recall: memory nodes, harvested concepts
and facets, graph-backed `recall`, async harvest/backfill, minimal `list_nodes`
inspection, and additive `concept_recall` output.

Automatic recall, fuzzy concept resolution, cleanup tools, and explicit agent hints
are v4.x follow-ups. They improve the experience without making v4.0 feel unfinished
if they need more tuning.

## Data Model

Add graph nodes and edges beside the existing memory tables.

```sql
CREATE TABLE nodes (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  label TEXT NOT NULL,
  normalized_label TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(kind, normalized_label)
);

CREATE TABLE node_aliases (
  id TEXT PRIMARY KEY,
  node_id TEXT NOT NULL REFERENCES nodes(id),
  alias TEXT NOT NULL,
  normalized_alias TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(node_id, normalized_alias)
);

CREATE INDEX node_aliases_normalized_alias_idx
ON node_aliases(normalized_alias);

CREATE TABLE edges (
  id TEXT PRIMARY KEY,
  src_node_id TEXT NOT NULL REFERENCES nodes(id),
  relation TEXT NOT NULL,
  dst_node_id TEXT NOT NULL REFERENCES nodes(id),
  confidence REAL NOT NULL DEFAULT 1.0,
  evidence_memory_id TEXT REFERENCES memories(id),
  origin TEXT NOT NULL DEFAULT 'harvest',
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(src_node_id, relation, dst_node_id)
);

CREATE INDEX edges_src_relation_idx
ON edges(src_node_id, relation);

CREATE INDEX edges_dst_relation_idx
ON edges(dst_node_id, relation);
```

Add a graph node pointer to memories:

```sql
ALTER TABLE memories ADD COLUMN node_id TEXT REFERENCES nodes(id);
```

`nodes(kind, normalized_label)` must be unique. Concept resolution is a concurrent
write path: multiple agents can discover `concept:Tuplia` at the same time. The write
path should use `INSERT ... ON CONFLICT(kind, normalized_label) DO NOTHING` or an
equivalent atomic upsert pattern, then select the node by `(kind, normalized_label)`.
Do not rely on check-then-insert logic without the unique constraint.

For `memory` nodes, use the memory ID as the normalized identity rather than the
normalized memory name. Memory names are unique as written, but two valid names can
normalize to the same value, such as `Foo` and `foo`. Memory nodes are internal
identity nodes; display can still come from the `memories.name` row.

## Schema Changes

Goldie does not currently have a migration framework. Existing schema setup is mostly
`CREATE TABLE IF NOT EXISTS`, and SQLite does not support `ALTER TABLE ... ADD COLUMN
IF NOT EXISTS`.

v4.0 needs an idempotent schema-change helper before adding `memories.node_id`.
The smallest version is:

1. Query `PRAGMA table_info(memories)`.
2. If `node_id` is absent, run `ALTER TABLE memories ADD COLUMN node_id TEXT
   REFERENCES nodes(id)`.
3. If `node_id` is already present, skip the alter.

A broader migration system using `PRAGMA user_version` can come later, but v4 should
not add a startup path that succeeds only once.

## Normalization

Define one `NormalizeLabel` function and use it everywhere labels, aliases, and
queries are resolved. Do not duplicate normalization logic between write and read
paths.

Initial v4 normalization:

1. Trim leading and trailing space.
2. Convert to lower case.
3. Collapse runs of whitespace to a single ASCII space.
4. Leave punctuation, path separators, hyphens, underscores, file extensions, and
   non-ASCII characters intact.

Aliases, not normalization, should handle meaningful variants such as `tuplia cloud`,
`tuplia_cloud`, and `tuplia-cloud`. Keeping normalization conservative avoids turning
paths, repo names, and product names into ambiguous strings.

## Deletion and Foreign Keys

Do not depend on SQLite cascades for graph cleanup. Goldie already uses explicit
cleanup for memory chunks and `memories_vec` because `memories_vec` is a `vec0`
virtual table and cannot participate in foreign-key cascades.

The current SQLite connection enables `_foreign_keys=on`, so references can still be
useful as integrity checks. Cleanup should nevertheless follow the existing store
pattern:

1. Load the memory row and its `node_id`.
2. Delete graph edges where the memory node is the source or destination.
3. Delete graph edges whose `evidence_memory_id` points at the memory.
4. Delete aliases for the memory node.
5. Delete vector rows and memory chunks with the existing chunk cleanup path.
6. Delete the memory row, which clears the `memories.node_id` reference.
7. Delete the now-unreferenced memory node.

Evidence edges should be deleted when their evidence memory is forgotten. Do not keep
them as orphaned assertions by setting `evidence_memory_id` to `NULL`. A later version
can support downgrading confidence when multiple evidence memories support the same
edge, but v4 should start with the simpler policy: one evidence memory deleted means
its single-evidence edge is deleted.

Concept nodes should not be automatically deleted when a memory is deleted. They may
still be connected to other memories, aliases, or future memories. Orphan concept
cleanup should be a separate maintenance operation, and it must delete referencing
edges and aliases before deleting the concept node.

## Node Kinds

Start with a small, open set:

- `memory`
- `concept`
- `project`
- `product`
- `repo`
- `file`
- `person`
- `topic`
- `decision`

The exact kind is useful for ranking and display, but recall should work even when a
concept is only classified as `concept`.

## Edge Relations

Start with a small relation vocabulary:

- `about`
- `mentions`
- `part_of`
- `related_to`
- `supersedes`
- `supports`
- `constrains`
- `from_source`
- `canonical_for`

Relation names should be open but normalized and validated against the known set
above, with a fallback such as `related_to`. This matches the current posture of
Goldie's memory type whitelist without forcing every future relation into the schema.
Canonical context should be represented with a strong edge such as `canonical_for`,
not with a new memory type. Keeping canonicality in the graph avoids touching the
memory type whitelist, filters, and tool documentation.

Examples:

```text
memory:feedback_tuplia_auth_no_passwords --about--> concept:Tuplia
memory:feedback_tuplia_auth_no_passwords --about--> topic:Auth
memory:feedback_tuplia_auth_no_passwords --constrains--> topic:Passwordless Auth
product:Tuplia Cloud --part_of--> product:Tuplia
memory:/Users/.../tuplia-context.md --from_source--> file:/Users/.../tuplia-context.md
file:/Users/.../tuplia-context.md --about--> product:Tuplia
```

## Write Path

When `remember` stores a memory:

1. Create the existing `memories` row.
2. Create a `memory` node for that row.
3. Link `memories.node_id` to the memory node.
4. Enqueue a debounced graph harvest job.
5. Commit without running full concept extraction under the write lock.

The graph harvest job does the concept work asynchronously:

1. Extract likely concepts from memory names and sources.
2. Resolve extracted concepts against existing node labels and aliases.
3. Create missing concept nodes only for high-confidence concepts.
4. Re-attach harvested `about` edges and `part_of` facet edges.

Keep extraction simple. Prefer deterministic parsing before adding model calls:

- Existing node labels and aliases.
- Repo and path positions, such as `tuplia-infra` or `/Users/.../Business/Tuplia`.
- Existing `[[wiki_style_links]]`, once explicit-hint support lands.
- Explicit MCP fields such as `about`, if added later.

Do not create nodes from raw title-case terms in v4.0. Sentence-initial words,
capitalized common nouns, and phrases like `Auth` or `Passwordless Auth` will pollute
the graph quickly if they mint concepts automatically. Title-case extraction may
produce candidate terms for inspection, but it should not create graph nodes unless
the term also matches an existing node or alias, appears in a path/repo position, or
comes from an explicit link or hint.

## Concept Harvesting from Names

The concept hierarchy comes from the descriptive names and sources agents already
produce. A name like `tuplia_cloud_passwordless_auth` or a path like
`/Users/srfrog/Documents/Business/Tuplia/...` encodes a concept hierarchy that the
agent will not remember next session. Goldie recovers it deterministically, with no
model call. This is the primary source of facets; agent hints (v4.3) are optional
reinforcement, not the mechanism.

Compound model: a facet is a child concept whose label extends its parent.

```text
concept:Tuplia Cloud --part_of--> concept:Tuplia
concept:Tuplia Architecture --part_of--> concept:Tuplia
```

Decompose a name or source into tokens by splitting on `_`, `-`, `/`, whitespace, and
case boundaries, then apply three rules:

1. Parent gate. Only mint a compound concept when an existing concept is a proper
   prefix of the token sequence. `Tuplia Cloud` is minted only because `Tuplia`
   already exists. Bare terms such as `Cloud` are never minted on their own. This is
   the same anti-pollution rule as the title-case ban, applied to harvested names.
2. Frequency gate. Only promote a token prefix to a concept when it is shared by at
   least N memories (start with N=2). `Tuplia Cloud` shared by several names is a real
   facet. `Tuplia Cloud Passwordless Auth` seen once is not a facet; that memory
   attaches to its nearest existing ancestor (`Tuplia Cloud`). The frequency gate is
   the deterministic stand-in for clustering: a facet is a shared name prefix, not a
   one-off tail.
3. Longest-prefix attachment. When several existing concepts are prefixes, attach
   `part_of` to the longest. `Tuplia Cloud Pricing` attaches to `Tuplia Cloud`, which
   is itself `part_of Tuplia`. The hierarchy assembles itself and recall walks
   `part_of` downward from the matched node.

Root seeding. The parent gate needs roots to exist first, but roots are bare terms the
title-case rule forbids. Seed roots two deterministic ways:

- Path and repo positions, as in the existing extraction rules
  (`/Users/.../Business/Tuplia/...` seeds `Tuplia`).
- The frequency gate applied to the leading token: a leading token shared by at least
  N memories is promoted to a root concept even though it is bare. Frequency, not
  capitalization, justifies the root.

Harvesting runs after writes through the async queue and again during the migration
backfill over existing names and sources, so v3 databases gain the same hierarchy
retroactively.

Full hierarchy refresh is queued, not run synchronously inside every `remember`
transaction. The write path only creates the memory node and enqueues a debounced
graph harvest job. This avoids O(N) harvest work and long write locks on every memory
write. The queue job recomputes the deterministic hierarchy in batches of work.

When a one-off tail later clears the frequency gate, harvest re-attaches existing
memories by deleting their previous harvested `about` edges (`origin = 'harvest'` and
`evidence_memory_id = memory.id`) before inserting the single best current harvested
`about` edge along the `part_of` hierarchy. That prevents the same memory from
appearing under both a root concept and a newly promoted facet.

This does not mean a memory can only have one `about` edge globally. Cross-axis
attachments are valid: a memory may be about `Tuplia Cloud` and also about `Auth`, or
future user hints may attach multiple explicit `about` concepts. Those non-harvested
edges must use a different `origin`, such as `hint`, so the harvest re-attachment pass
does not clobber them.

## MCP API Changes

Keep current tools compatible.

Optional additions to `remember` and `update_memory`:

```json
{
  "about": ["Tuplia", "Auth"],
  "aliases": ["Tuplia Cloud"],
  "links": [
    {
      "relation": "constrains",
      "target": "Passwordless Auth"
    }
  ]
}
```

These are hints, not required fields. If omitted, Goldie derives graph links itself.

Graph inspection tools:

- `list_nodes` - inspect concept and memory nodes, including alias and edge counts
  for testing grouped recall.
- `merge_nodes` - merge duplicate concepts and move aliases/edges.
- `link_memory` - manually attach a memory to a concept.
- `explain_recall` - show vector matches, alias matches, and graph expansion.

`list_nodes` is the minimal v4 test surface. `explain_recall` is required for
automatic recall inspectability. The cleanup tools can wait until graph-backed recall
is working and concept cleanup becomes necessary.

## Recall Path

`recall` should become hybrid retrieval:

1. Normalize the query.
2. Resolve exact node label and alias matches.
3. Run existing vector KNN over memory chunks.
4. Expand graph neighbors from matched concept nodes, including `part_of` children
   (the concept's facets) and the memories attached to each.
5. Merge vector matches and graph matches.
6. Rank by:
   - vector score
   - exact alias or label match
   - graph distance
   - edge relation strength
   - memory type priority
   - recency
7. Shape the result by what matched:
   - Plain queries return the existing flat memory result shape, unchanged.
   - A query that resolves to a concept returns a grouped result: the concept and its
     canonical context, then one group per facet (`Tuplia Cloud`, `Tuplia
     Architecture`, ...) with that facet's top memories, then root-level memories
     attached directly to the concept. Vector matches fill in related results.
   The grouped shape is additive; clients that ignore the grouping still get memories.

Known limitation: graph resolution in v4 starts with exact normalized label and alias
matches. `Tuplia` resolves well if it is a known label or alias. `the Tuplia auth
thing`, misspellings, and vague paraphrases may fall back entirely to vector search
unless an exact alias exists for the phrase. v4.1 adds embeddings for concept
labels, aliases, and descriptions, using a `nodes_vec` table following the existing
`memories_vec` pattern, to support fuzzy concept resolution.

Example metadata:

```json
{
  "matched_node": "Tuplia",
  "matched_by": ["alias", "vector", "graph"],
  "graph_distance": 1
}
```

## Automatic Recall

Automatic recall is the access pattern layered over graph recall. The caller should
not need to know the exact concept, the facet, or the target memory. The caller gives
Goldie a task-shaped query; Goldie resolves a vantage, descends the graph, and returns
a specific recall when the gradient is clear.

Recall is locally cylindrical:

- `axis` - the vantage concept.
- `radius` - relevance, computed from graph distance plus vector similarity.
- `angle` - facets around the vantage, such as `Tuplia Cloud` or `Tuplia Pricing`.
- `gradient` - weighted, asymmetric edges and type/recency boosts that make traversal
  converge instead of returning a flat neighborhood.

Automatic recall runs in three steps:

1. Resolve the vantage.
   - Exact label and alias match first.
   - Fuzzy match over concept labels/descriptions later through `nodes_vec`, following
     the existing `memories_vec` pattern.
   - Return the vantage plus resolution confidence.
2. Descend the gradient.
   - Score reachable concepts and memories with the same ranking formula.
   - Use a small beam rather than pure greedy descent, so one local maximum does not
     dominate too early.
   - Stop when descending further dilutes the score or the depth cap is reached.
   - Return the memory or tight set of memories plus the path and per-step scores.
3. Detect ambiguity.
   - If no vantage resolves, fall back to flat vector recall.
   - If the top candidates are too close, mark the result unconverged and return the
     grouped neighborhood instead of inventing precision.

The first implementation can return automatic recall as additive metadata beside the
existing flat `results` array. Clients that ignore it still work. Clients that use it
get the specific recall, path, convergence flag, and fallback reason.

## Inspectability

Automatic recall must show its work. Once traversal is automatic, a bad edge weight or
bad vantage can otherwise become a silent wrong turn.

Every automatic recall should return a compact trace by default:

- the resolved vantage and resolution confidence
- the path of concepts and memories considered
- the final memory or tight memory set
- the score at each step
- the stop reason: `converged`, `flat_gradient`, `unresolved_vantage`, `depth_cap`, or
  `beam_exhausted`

`explain_recall` should become the full trace tool. It should show candidate scores at
each step, edges followed, relation weights, vector contributions, and why the descent
stopped. A wrong recall must be diagnosable to a bad vantage or a specific bad edge
weight, not to "the graph."

## Ranking Rules

Keep ranking legible and tunable:

- Exact alias or label matches should strongly boost results.
- Memories directly linked with `about` should rank above `mentions`.
- Canonical reference memories should rank high for broad concept queries.
- `feedback` and `project` memories should rank high when they contain constraints or
  decisions.
- Duplicate memories should not both dominate the top results.

Start v4.0 with a concrete weighted sum, then tune from observed results:

```text
score =
  0.45 * vector_similarity +
  0.25 * concept_match +
  0.15 * graph_proximity +
  0.07 * relation_strength +
  0.05 * type_priority +
  0.03 * recency
```

All signals should be normalized to `0..1`.

- `vector_similarity`: existing vector result normalized so higher is better; missing
  vector match is `0`.
- `concept_match`: `1` when the query exactly matches a label or alias for a concept
  directly linked to the candidate memory; otherwise `0`.
- `graph_proximity`: `1` for a direct concept-to-memory edge, `0.5` for two hops,
  `0.25` for three hops, `0` beyond that.
- `relation_strength`: `canonical_for=1`, `about=0.95`,
  `constrains/supports/supersedes=0.9`, `from_source=0.8`, `mentions=0.5`,
  `related_to=0.4`.
- `type_priority`: start with `feedback=0.9`, `project=0.85`, `reference=0.75`,
  `todo=0.7`, `reminder=0.65`, `idea=0.55`, `opinion=0.5`, `user=0.5`.
- `recency`: simple decay such as `1` for 30 days, `0.5` for 180 days, `0.2` older.

## Migration

For existing v3 databases:

1. Create graph tables.
2. Create one `memory` node for each existing memory.
3. Backfill `memories.node_id`.
4. Extract concepts from existing names and sources.
5. Create aliases from obvious variants:
   - original label
   - snake_case and title-case variants
   - only variants whose normalized form differs from the node label's normalized form
6. Create file and repo nodes from paths and repository-looking sources.
7. Link memories to concepts.

Do not add file paths or basenames as aliases of product/project concepts. Paths are
`file` or `repo` nodes linked by edges. Aliases should be name variants of the same
concept, not neighboring resources.

This migration can be best-effort. It should not block startup if concept extraction
fails for one memory.

Backfill should run through the existing async job queue instead of blocking schema
initialization. Startup should create the graph tables and guarded `node_id` column,
then enqueue or resume a graph backfill job that processes existing memories in
batches. `recall` must tolerate partially backfilled databases by using vector search
for memories without graph nodes.

## Tuplia Example

Today, `recall "Tuplia"` returns useful memories because the word appears in paths,
names, descriptions, and body chunks. v4 makes the hierarchy explicit by harvesting it
from those same names. `Tuplia` is seeded as a root from the path and its frequent
leading token. `Tuplia Cloud` clears the frequency gate (shared by two names) and
becomes a facet; `tuplia_deployment_modes` is a one-off tail and attaches to the root.

```text
concept:Tuplia                                      (root: path + frequent leading token)
concept:Tuplia Cloud --part_of--> concept:Tuplia    (facet: prefix shared by 2+ names)

file:/Users/srfrog/Documents/Business/Tuplia/tuplia-context.md --about--> concept:Tuplia
memory:/Users/.../tuplia-context.md --canonical_for--> concept:Tuplia
memory:/Users/.../tuplia-context.md --from_source--> file:/Users/.../tuplia-context.md
memory:tuplia_sources_product_spec --about--> concept:Tuplia
memory:tuplia_deployment_modes --about--> concept:Tuplia
memory:tuplia_cloud_passwordless_auth --about--> concept:Tuplia Cloud
memory:feedback_tuplia_auth_no_passwords --about--> concept:Tuplia Cloud
```

Then `recall "Tuplia"` resolves the root concept, pulls its canonical context, groups
the directly attached memories under their facets (`Tuplia Cloud`) and the root, and
uses vector recall to fill in related results.

Automatic recall goes one step further:

```text
query: how does Tuplia handle auth

resolve vantage:
  concept:Tuplia

descend:
  concept:Tuplia
    -> concept:Tuplia Cloud
      -> memory:feedback_tuplia_auth_no_passwords

return:
  memory: feedback_tuplia_auth_no_passwords
  path: Tuplia -> Tuplia Cloud -> feedback_tuplia_auth_no_passwords
  stop_reason: converged
```

If the auth branch is not clearly stronger than other Tuplia branches, automatic
recall should return the grouped Tuplia neighborhood and mark the descent
`flat_gradient` instead of pretending it found the one right memory.

## Delivery Versions

### v4.0: Grouped Graph Recall

This is the launchable v4 boundary.

- Add `nodes`, `node_aliases`, and `edges`.
- Add `memories.node_id`.
- Create a memory node on every `remember` and `index_file`.
- Add explicit graph cleanup to the memory delete path.
- Backfill existing memories through the async job queue.
- Extract obvious concepts on write.
- Harvest compound facets from names and sources using the parent gate, the frequency
  gate (N=2), and longest-prefix attachment.
- Seed root concepts from path/repo positions and frequent leading tokens.
- Create aliases for exact labels and normalized variants.
- Create harvested aliases for underscore and hyphen variants.
- Link memories to concepts with `about` and `mentions`, and facets to parents with
  `part_of`.
- Defer `[[wiki_style_links]]` parsing until explicit-hint support.
- Do not mint concept nodes from raw title-case extraction.
- Resolve query labels and aliases.
- Expand direct memory links and `part_of` facets from matched concepts.
- Merge graph results with vector results.
- Use the initial weighted ranking formula.
- Return the grouped facet shape for concept queries; keep the flat shape for plain
  queries.
- Add recall metadata for debugging.
- Ship minimal `list_nodes` inspection for testing harvested concepts and edge counts.

### v4.1: Automatic Recall

- Resolve a vantage from exact labels, aliases, and contained concept labels in the
  query.
- Implement deterministic descent over weighted graph candidates.
- Return a compact path, convergence flag, branch scores, and stop reason as
  additive `automatic_recall` metadata.
- Fall back to grouped/flat recall when the gradient is ambiguous.
- Add `nodes_vec` for fuzzy concept resolution.
- Promote `explain_recall` to a first-class trace tool.
- Auto-recall quality is bounded by concept hygiene until v4.2 cleanup tools land.
  Exact duplicates are prevented by `UNIQUE(kind, normalized_label)`, but
  near-duplicates like `Tuplia Cloud` and `Tuplia Cloud Platform` can still flatten or
  split the gradient. `flat_gradient` fallback is the protection in this phase.

### v4.2: Cleanup Tools

- Expand concept inspection beyond the minimal `list_nodes` test surface.
- Add manual merge/link tools.
- Add dedupe assistance for near-equivalent memories.
- Make `merge_nodes` collision-safe: re-point aliases and edges with
  `INSERT OR IGNORE` or equivalent deduplication so `UNIQUE(src_node_id, relation,
  dst_node_id)` collisions do not abort the merge.

### v4.3: Optional Agent Hints

- Add optional `about`, `aliases`, and `links` parameters.
- Keep plain `remember` as the default path.
- Document how agents should use hints without requiring them.

## Open Questions

Resolved for v4: `recall` returns facet-grouped concept summaries for concept queries
(see Recall Path). Automatic concept creation is bounded by the parent gate plus the
frequency gate (see Concept Harvesting from Names). The compound model
(`Tuplia Architecture --part_of--> Tuplia`) is chosen over faceted intersection; the
known trade is no cross-cutting facet (`Tuplia Architecture` and `Goldie Architecture`
share no `Architecture` node).

- What frequency threshold N best separates real facets from one-off name tails?
  Start with N=2 and tune against a real database.
- Should harvested facet labels preserve the original casing and spacing reconstructed
  from the source name, or render from the normalized form?
- What margin counts as a flat gradient? Too tight and v4 hallucinates precision; too
  loose and it falls back constantly.
- What beam width keeps descent cheap without making it brittle?
- Should the full path be returned always, or only through `explain_recall`, given
  token cost on the hot path?
- Does vantage resolution ever return multiple axes for a query spanning two concepts,
  and if so, should Goldie descend both and merge or refuse as ambiguous?

## Success Criteria

### v4.0

- `recall "Tuplia"` reliably returns canonical Tuplia context and high-signal Tuplia
  memories even when names/descriptions vary.
- `recall "Tuplia"` surfaces the concept's facets (`Tuplia Cloud`, `Tuplia
  Architecture`, ...) harvested from existing memory names, grouped under the concept.
- `list_nodes kind=concept query=Tuplia` shows the harvested root and facets with
  alias and edge counts.

### v4.x

- `recall "how does Tuplia handle auth"` can return a specific high-confidence memory
  without the caller naming intermediate concepts or the target memory.
- Every automatic recall is inspectable: the vantage, path, scores, and stop reason
  are visible.
- When the gradient is ambiguous, recall falls back and says so rather than returning a
  confident wrong point.
- A wrong recall is diagnosable to a bad vantage or specific edge weight, not to a
  black box.
- Agents can store short, plain memories without inventing complex names.
- Duplicate concepts can be inspected and merged.
- Existing v3 databases migrate without losing memories.
- Vector search still works as before, but recall improves when a stable concept is
  known.
