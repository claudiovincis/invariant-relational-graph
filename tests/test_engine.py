"""
Comprehensive test suite for SubtractionEngine.
Run:   python test_engine.py
Each test prints PASS/FAIL. Exit code 0 = all passed.
"""
import sys, os, tempfile, json

# Make import work from both repo root and tests/ folder
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_HERE) == "tests":
    # Running from tests/ folder: engine is in ../src/
    sys.path.insert(0, os.path.join(_HERE, "..", "src"))
else:
    # Running from workspace root
    sys.path.insert(0, os.path.join(_HERE, "IRG", "repo", "src"))

from subtraction_engine import SubtractionEngine

FAILS = 0
PASSES = 0

def check(cond, msg):
    global PASSES, FAILS
    if cond:
        PASSES += 1
        print(f"  PASS  {msg}")
    else:
        FAILS += 1
        print(f"  FAIL  {msg}  <--")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════
section("1. Basic operations: add, compact, ask")

g = SubtractionEngine()
g.add("cat", "is", "animal")
g.add("cat", "is", "mammal")
g.add("cat", "eats", "fish")
g.compact()

check(g.n_nodes == 4, f"n_nodes == 4 (got {g.n_nodes})")
check(g.n_edges == 3, f"n_edges == 3 (got {g.n_edges})")

r = g.ask("what is cat?")
check("cat" in r["concepts"], "ask() finds 'cat'")
defs = [d["concept"] for d in r["concepts"]["cat"]["top_definitions"]]
check(len(defs) == 3, f"3 definitions for cat (got {len(defs)}: {defs})")


# ═══════════════════════════════════════════════════════════
section("2. Deduplication: same edge added twice")

g2 = SubtractionEngine()
g2.add("a", "rel", "b")
g2.add("a", "rel", "b")
g2.add("a", "rel", "b")
g2.compact()
check(g2.n_edges == 1, f"dedup: n_edges == 1 (got {g2.n_edges})")
check(g2.n_nodes == 2, f"dedup: n_nodes == 2 (got {g2.n_nodes})")


# ═══════════════════════════════════════════════════════════
section("3. is_concept: two-senses rule")

g3 = SubtractionEngine()
g3.add("cat", "is", "animal", sense="text")
g3.add("mammal", "is", "animal", sense="text")
g3.add("is", "is", "animal", sense="text")
# Second sense: category file
g3.add("feline", "category", "cat", sense="categories")
g3.add("mammal", "category", "cat", sense="categories")
g3.compact()

cat_id = g3.id_by_label.get(("en", "cat"))
mammal_id = g3.id_by_label.get(("en", "mammal"))
is_id = g3.id_by_label.get(("en", "is"))
animal_id = g3.id_by_label.get(("en", "animal"))

check(cat_id is not None, "cat has ID")
check(g3.is_concept(cat_id) == True, "cat IS concept (2 senses)")
check(g3.is_concept(mammal_id) == True, "mammal IS concept (2 senses)")
check(g3.is_concept(is_id) == False, "'is' is NOT concept (1 sense)")
check(g3.is_concept(animal_id) == False, "animal: only 1 sense")


# ═══════════════════════════════════════════════════════════
section("4. merge_concept: cross-lingual equivalence")

g4 = SubtractionEngine()
g4.merge_concept({"it": "gatto", "en": "cat", "fr": "chat"})
g4.add("gatto", "e", "animale", language="it")
g4.add("cat", "is", "animal", language="en")
g4.add("chat", "est", "animal", language="fr")
g4.compact()

cat_en = g4.id_by_label.get(("en", "cat"))
cat_it = g4.id_by_label.get(("it", "gatto"))
cat_fr = g4.id_by_label.get(("fr", "chat"))
check(cat_en == cat_it == cat_fr, f"same node: en={cat_en} it={cat_it} fr={cat_fr}")

r_en = g4.ask("what is cat?", language="en")
r_it = g4.ask("cos'e un gatto?", language="it")
defs_en = [d["concept"] for d in r_en["concepts"]["cat"]["top_definitions"]]
defs_it = [d["concept"] for d in r_it["concepts"]["gatto"]["top_definitions"]]
# Note: 'animal' FR is a different node from 'animal' EN (not merged).
# The 2 defs are: 'animale' (IT edge) + 'animal' (EN edge). FR's 'animal' is separate.
check(len(defs_en) == 2, f"EN query: 2 defs (got {len(defs_en)}: {defs_en})")
check(len(defs_it) == 2, f"IT query: 2 defs (got {len(defs_it)}: {defs_it})")
check(set(defs_en) == set(defs_it), "EN and IT queries return same definitions")


# ═══════════════════════════════════════════════════════════
section("5. merge_concept: merge onto existing node")

g5 = SubtractionEngine()
g5.add("gatto", "e", "animale", language="it")  # creates node for gatto(IT)
g5.merge_concept({"it": "gatto", "en": "cat"})   # should reuse same node
g5.add("cat", "is", "animal", language="en")
g5.compact()

cat_it = g5.id_by_label.get(("it", "gatto"))
cat_en = g5.id_by_label.get(("en", "cat"))
check(cat_it == cat_en, f"merge onto existing: it={cat_it} en={cat_en}")


# ═══════════════════════════════════════════════════════════
section("6. context_intersection: shared concepts")

g6 = SubtractionEngine()
g6.add("cat", "is", "mammal")
g6.add("cat", "is", "animal")
g6.add("cat", "has", "tail")
g6.add("dog", "is", "mammal")
g6.add("dog", "is", "animal")
g6.add("dog", "has", "tail")
g6.add("whale", "is", "mammal")
g6.add("whale", "lives in", "ocean")

# Second sense to make concepts recognizable
for word in ["mammal", "animal", "tail", "ocean"]:
    g6.add(word, "is_entry", word, sense="dictionary")

g6.compact()

# cat & dog share mammal, animal, tail
ci = g6.context_intersection("cat", "dog")
shared = [s[0] for s in ci["shared"]]
check("mammal" in shared, f"cat-dog share 'mammal': {shared}")
check("animal" in shared, f"cat-dog share 'animal': {shared}")
check("tail" in shared, f"cat-dog share 'tail': {shared}")
check(ci["empty"] == False, "cat-dog: not empty")


# ═══════════════════════════════════════════════════════════
section("7. context_intersection: multi-hop bridge (no direct overlap)")

g7 = SubtractionEngine()
g7.add("cat", "is", "mammal")
g7.add("mammal", "is", "vertebrate")
g7.add("whale", "is", "vertebrate")
for w in ["mammal", "vertebrate"]:
    g7.add(w, "is_entry", w, sense="dictionary")
g7.compact()

ci = g7.context_intersection("cat", "whale")
# cat → mammal → vertebrate. whale → vertebrate.
# cat's direct neighbors: {mammal}. whale's: {vertebrate}. No direct overlap.
# But there's a multi-hop bridge: cat → mammal → vertebrate ← whale
check(len(ci["shared"]) == 0, f"cat-whale no direct: shared={ci['shared']}")
bridge = ci.get("multi_hop_bridge")
check(bridge is not None, "cat-whale: multi-hop bridge found")
if bridge:
    check("vertebrate" in bridge["path"], f"bridge includes vertebrate: {bridge['path']}")
check(ci["empty"] == False, "cat-whale: not empty (bridge)")

# Now test with no direct overlap: cat → mammal, whale → mammal → ?? 
# Actually let's use a chain: cat → mammal → animal, whale → animal only
# cat neighbors: {mammal}
# whale neighbors: {animal}
# No direct overlap → multi-hop should kick in
g7b = SubtractionEngine()
g7b.add("cat", "is", "mammal")
g7b.add("mammal", "is", "animal")
g7b.add("whale", "is", "animal")
for w in ["mammal", "animal"]:
    g7b.add(w, "is_entry", w, sense="dictionary")
g7b.compact()

ci2 = g7b.context_intersection("cat", "whale")
# cat neighbors: {mammal}. whale neighbors: {animal}.
# mammal ≠ animal → no direct intersection
# But there's a path: cat → mammal → animal ← whale
# Actually mammal → animal, and whale → animal. So cat's neighbors include mammal,
# but not animal. whale's neighbors include animal. No shared IDs.
# Path: cat → mammal → animal ← whale (2 hops from cat to whale via animal)
# Actually: find_path should find cat → mammal → animal ← whale
check(len(ci2["shared"]) == 0, f"cat-whale no direct overlap: shared={ci2['shared']}")
bridge = ci2.get("multi_hop_bridge")
check(bridge is not None, "cat-whale: multi-hop bridge found")
if bridge:
    check(len(bridge["path"]) >= 3, f"bridge path length >= 3: {bridge['path']}")


# ═══════════════════════════════════════════════════════════
section("8. Ask with context filter")

g8 = SubtractionEngine()
g8.add("einstein", "was born in", "ulm")
g8.add("einstein", "developed", "relativity")
g8.add("einstein", "won", "nobel prize")
g8.add("curie", "was born in", "warsaw")
g8.add("curie", "won", "nobel prize")
for w in ["ulm", "relativity", "nobel prize", "warsaw"]:
    g8.add(w, "is_entry", w, sense="dictionary")
g8.compact()

r = g8.ask("einstein physics")
check("einstein" in r["concepts"], "ask with context: finds einstein")
defs = [d["concept"] for d in r["concepts"]["einstein"]["top_definitions"]]
check("relativity" in defs, f"context 'physics' keeps relativity: {defs}")


# ═══════════════════════════════════════════════════════════
section("9. Save/Load roundtrip")

g9 = SubtractionEngine()
g9.merge_concept({"it": "gatto", "en": "cat"})
g9.add("cat", "is", "animal", language="en")
g9.add("gatto", "e", "animale", language="it")
g9.add("cat", "eats", "fish", language="en")
g9.compact()

with tempfile.NamedTemporaryFile(suffix=".irgn", delete=False) as tmp:
    tmp_path = tmp.name
g9.save(tmp_path)

g9_loaded = SubtractionEngine.load(tmp_path)
check(g9_loaded.n_nodes == g9.n_nodes, f"save/load: nodes {g9.n_nodes} == {g9_loaded.n_nodes}")
check(g9_loaded.n_edges == g9.n_edges, f"save/load: edges {g9.n_edges} == {g9_loaded.n_edges}")

r = g9_loaded.ask("what is cat?", language="en")
check("cat" in r["concepts"], "loaded: ask finds cat")
defs = [d["concept"] for d in r["concepts"]["cat"]["top_definitions"]]
check("animal" in defs, f"loaded: data intact: {defs}")

os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════
section("10. Edge cases: empty, unknown, self-loop")

# Empty graph
g_empty = SubtractionEngine()
g_empty.compact()
check(g_empty.n_nodes == 0, f"empty: n_nodes == 0 (got {g_empty.n_nodes})")
check(g_empty.n_edges == 0, f"empty: n_edges == 0 (got {g_empty.n_edges})")
r = g_empty.ask("what is x?")
check(r["concepts"] == {}, f"empty ask: no concepts (got {r['concepts']})")

# Unknown node
g_unk = SubtractionEngine()
g_unk.add("a", "rel", "b")
g_unk.compact()
r = g_unk.ask("what is z?")
check(r["concepts"] == {}, f"unknown query: no concepts (got {r['concepts']})")

ci = g_unk.context_intersection("a", "z")
check(ci["empty"] == True, "intersection with unknown: empty=True")

# Self-loop
g_slf = SubtractionEngine()
g_slf.add("cat", "is", "cat")
g_slf.compact()
check(g_slf.n_edges == 1, "self-loop: accepted")


# ═══════════════════════════════════════════════════════════
section("11. _get_or_create idempotency")

g_id = SubtractionEngine()
id1 = g_id._get_or_create("cat", "en")
id2 = g_id._get_or_create("cat", "en")
check(id1 == id2, f"get_or_create idempotent: {id1} == {id2}")


# ═══════════════════════════════════════════════════════════
section("12. add_label")

g_lbl = SubtractionEngine()
nid = g_lbl._get_or_create("cat", "en")
g_lbl.add_label(nid, "it", "gatto")
g_lbl.add_label(nid, "fr", "chat")
check(g_lbl.id_by_label.get(("it", "gatto")) == nid, "add_label: IT lookup")
check(g_lbl.id_by_label.get(("fr", "chat")) == nid, "add_label: FR lookup")
check(g_lbl.labels[nid] == {"en": "cat", "it": "gatto", "fr": "chat"}, f"add_label: all labels: {dict(g_lbl.labels[nid])}")


# ═══════════════════════════════════════════════════════════
section("13. n_concepts property")

g_nc = SubtractionEngine()
g_nc.add("cat", "is", "animal", sense="text")
g_nc.add("dog", "is", "animal", sense="text")
g_nc.add("feline", "category", "cat", sense="categories")
g_nc.add("canine", "category", "dog", sense="categories")
g_nc.compact()
check(g_nc.n_concepts == 2, f"n_concepts: cat+dog = 2 (got {g_nc.n_concepts})")


# ═══════════════════════════════════════════════════════════
section("14. Stress: 1000 edges, no crash, compact idempotent")

g_stress = SubtractionEngine()
for i in range(1000):
    g_stress.add(f"node_{i}", "links_to", f"node_{(i+1) % 100}")
g_stress.compact()
g_stress.compact()  # double compact should be no-op
check(g_stress.n_nodes > 0, f"stress: n_nodes > 0 ({g_stress.n_nodes})")
check(g_stress.n_edges == 1000, f"stress: n_edges == 1000 ({g_stress.n_edges})")


# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  RESULTS: {PASSES} passed, {FAILS} failed")
print(f"{'='*60}")

if FAILS > 0:
    print(f"\n*** {FAILS} TEST(S) FAILED — DO NOT PUBLISH ***")
    sys.exit(1)
else:
    print(f"\nAll {PASSES} tests passed. Safe to publish.")
    sys.exit(0)
