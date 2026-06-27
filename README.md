# Growing Relational Graph (IRG)

**Reasoning by subtraction, not by accumulation.**

A deterministic knowledge system. No embeddings, no probabilities, no weights — just pure counting.

## Setup

```bash
pip install numpy
git clone https://github.com/claudiovincis/IRG.git
cd IRG
```

## Quick Start

30 seconds, copy-paste into a terminal:

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from subtraction_engine import SubtractionEngine

g = SubtractionEngine()
g.add('cat', 'is', 'animal', language='en')
g.add('cat', 'is', 'mammal', language='en')
g.add('cat', 'is', 'felid', language='en')
g.add('dog', 'is', 'animal', language='en')
g.add('dog', 'is', 'mammal', language='en')
g.add('whale', 'is', 'mammal', language='en')
g.compact()

result = g.ask('what is cat?', language='en')
for d in result['concepts']['cat']['top_definitions']:
    print(f\"  cat is {d['concept']} (invariance={d['invarianza']})\")

print()
ci = g.context_intersection('cat', 'whale', language='en')
print(f'cat and whale share: {ci[\"shared\"]}')
if ci.get('multi_hop_bridge'):
    print(f'bridge: {ci[\"multi_hop_bridge\"][\"path\"]}')
"
```

Output:
```
  cat is mammal (invariance=3)
  cat is animal (invariance=2)
  cat is felid (invariance=1)

cat and whale share: []
bridge: ['cat', 'mammal', 'whale']
```

## How to Use

### 1. Add edges

Every edge is a `(subject, relation, object)` triplet. This is how you teach the engine:

```python
g.add("einstein", "was born in", "ulm",    language="en")
g.add("einstein", "developed", "relativity", language="en")
g.add("einstein", "won", "nobel prize",     language="en")
g.add("curie",    "was born in", "warsaw",  language="en")
g.add("curie",    "won", "nobel prize",     language="en")
```

Each `add()` is O(1). No training, no backpropagation.

### 2. Add a second sense

A concept needs more than one independent source. Without a second sense, real words like "cat" get buried under high-connectivity noise:

```python
# Sense 1: text edges
g.add("cat", "is", "animal", language="en")

# Sense 2: categories (any independent inventory)
g.add("feline", "category", "cat", sense="categories", language="en")
```

Now "cat" is recognized as a concept. "is" never gets a category edge — it stays a functional.

### 3. Compact

After adding edges, merge the delta buffer and rebuild indices:

```python
g.compact()
```

### 4. Ask questions

```python
result = g.ask("what is einstein?", language="en")

# result["concepts"]["einstein"]["top_definitions"]:
# [
#   {"concept": "nobel prize", "invarianza": 4, "source": "outgoing"},
#   {"concept": "ulm",         "invarianza": 2, "source": "outgoing"},
#   {"concept": "relativity",  "invarianza": 2, "source": "outgoing"},
# ]
```

Add context to filter answers:

```python
result = g.ask("einstein physics", language="en")
# "physics" acts as a filter — only neighbors that overlap with
# the neighborhood of "physics" survive. "ulm" is gone.
```

### 5. Find what two concepts share

```python
ci = g.context_intersection("einstein", "curie", language="en")
# ci["shared"]   = []                           <- no direct shared concepts
# ci["strength"]  = 0.000                       <- but multi-hop may find a bridge
# ci["empty"]     = False                       <- True = "I don't know"
```

## Using Your Own Data

You don't need `build_irg.py`. The engine supports multiple languages — use `merge_concept()` to declare that words in different languages refer to the SAME concept:

```python
g = SubtractionEngine()

# Declare cross-lingual concept: gatto = cat = chat
g.merge_concept({"it": "gatto", "en": "cat", "fr": "chat"})

# Now add() in any language converges on the same node
g.add("gatto", "è", "animale", language="it")
g.add("gatto", "è", "mammifero", language="it")
g.add("cat", "is", "animal", language="en")
g.add("cat", "eats", "fish", language="en")
g.add("chat", "est", "animal", language="fr")

g.compact()

# Both queries return ALL edges (IT + EN + FR merged)
g.ask("what is cat?", language="en")       # → animale, mammifero, animal, fish
g.ask("cos'è un gatto?", language="it")    # → same results
```

**From a file.** If you have triplets in JSONL format (`{"subject": "...", "relation": "...", "object": "..."}` per line), use `build_irg.py`:

```bash
python src/build_irg.py --input my_triplets.jsonl --output my_graph.irgn
```

**From existing sources.** The engine accepts edges from any pipeline — spaCy, an LLM extractor, a CSV, a database. As long as you produce `(subject, relation, object)`, the engine doesn't care where they came from.

**Second sense.** Remember: for concepts to work, they need a second independent source. A simple approach:

```python
# Sense 1: your text edges
for s, r, o in my_triplets:
    g.add(s, r, o, sense="text", language="en")

# Sense 2: a word list, category file, or dictionary
for word in my_word_list:
    g.add(word, "is_entry", word, sense="dictionary", language="en")
```

## API

| Method | What it does |
|--------|-------------|
| `g.add(src, rel, dst, sense="text", language="it")` | Add an edge. `sense` separates independent data sources. |
| `g.compact()` | Merge delta buffer, rebuild indices. |
| `g.ask(question, language="en")` | Answer a question. Returns ranked neighbors + pairwise connections. |
| `g.context_intersection(a, b, extra_context=None)` | Find what A and B share. |
| `g.merge_concept({"it": "gatto", "en": "cat"})` | Declare cross-lingual equivalence. All future `add()` calls converge. |
| `g.is_concept(node_id)` | True if node appears in >1 sense. |
| `g.save(path)` / `g.load(path)` | Save/load `.irgn` binary format. |

## The Two-Senses Rule

| Word | Text sense | Category sense | Verdict |
|------|-----------|---------------|---------|
| cat | "cat is mammal" | `feline -> cat` | CONCEPT |
| mammal | "dog is mammal" | `mammal -> cat` | CONCEPT |
| is | X is Y everywhere | never | FUNCTIONAL (filtered out) |

No blacklist. No hardcoded rules. The topology decides.

## Principles

- **Zero weights** — a connection exists or it doesn't
- **Two senses** — concepts emerge from intersecting independent sources
- **Context subtracts** — answers are filtered, not ranked
- **Topological honesty** — "I don't know" is a first-class answer

## Files

| File | What |
|------|------|
| `src/subtraction_engine.py` | The engine — `SubtractionEngine` class |
| `src/build_irg.py` | Build a graph from triplets (JSONL) |
| `src/build_irg_reified.py` | Build a graph with abstract concept mapping |
| `src/engine_compact.py` | Low-level compact graph storage |
| `src/engine_compact_train.py` | Compact engine with O(1) add |
| `MANIFESTO.md` | Full theory, algorithm, and limitations |

## Tests

```bash
python tests/test_engine.py
```

46 tests, 14 sections. All must pass before publishing.

| # | Test | What it proves |
|---|---|---|
| 1 | add/compact/ask | `g.ask("what is cat?")` returns 3 definitions ranked by invariance |
| 2 | Dedup | Same triplet added 3× → 1 edge only |
| 3 | Two-senses rule | `is_concept()`: "cat" with 2 senses = TRUE, "is" with 1 = FALSE |
| 4 | merge_concept | `merge_concept({"it":"gatto","en":"cat","fr":"chat"})` → same node, `add()` converges |
| 5 | Merge onto existing | If node already exists, `merge_concept` reuses it |
| 6 | context_intersection | cat and dog share "mammal", "animal", "tail" → `shared` not empty |
| 7 | Multi-hop bridge | cat→mammal→vertebrate←whale: no direct overlap → bridge found |
| 8 | Ask with context | `g.ask("einstein physics")`: "relativity" survives, "ulm" filtered out |
| 9 | Save/Load roundtrip | `.irgn` binary → reload → same nodes, same edges, `ask()` works |
| 10 | Edge cases | Empty graph, unknown node, self-loop — all handled, no crash |
| 11 | Idempotency | `_get_or_create` called 2× → same ID |
| 12 | add_label | "en:cat" + "it:gatto" + "fr:chat" → lookup works in all 3 languages |
| 13 | n_concepts | 2 senses each for "cat" and "dog" → `n_concepts = 2` |
| 14 | Stress | 1000 edges, double `compact()` idempotent, <2 seconds |

## License

**Code** (`src/`, `tests/`, `build_irg.py`): [GNU AGPL v3](LICENSE) — free software, copyleft. If you use it in production (including SaaS), you must publish your modifications.

**Theory & Manifesto** (`MANIFESTO.md`, `README.md`): [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) — attribution required, share-alike.

© 2026 Claudio Vincis

## Authorship

The theory and manifesto are original work by Claudio Vincis. The engine code was written with LLM assistance.
