# The Invariant is the Concept

> *A Technical Report on the Invariant Relational Graph — reasoning by subtraction, not by accumulation.*

**Claudio Vincis** — June 27, 2026 (work in progress)

---

## Abstract

This report presents the Invariant Relational Graph (IRG), a deterministic knowledge system where concepts emerge not from statistical accumulation but from topological subtraction. The core thesis: a concept is what remains invariant when you change the subject that observes it through the same relation. This is implemented as a pure count — invariance(target) = len(incoming[target]) — with no embeddings, no probabilities, no weights. The system answers queries by intersecting neighborhoods in O(deg(A)+deg(B)), avoiding the combinatorial explosion of pathfinding. Architectural honesty is guaranteed by an adaptive threshold computed from the graph itself: when the intersection is too weak, the system declares ignorance rather than hallucinating. The system is not a replacement for LLMs; it is a structural oracle for tasks requiring transparency, auditability, and "I don't know" as a first-class answer.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Core Principles](#2-core-principles)
   - [2.1 Invariance: What No One Had Seen](#21-invariance-what-no-one-had-seen)
   - [2.2 The Illusion of Pure Text](#22-the-illusion-of-pure-text)
   - [2.3 Two Senses vs. One Sense: the Experiment](#23-two-senses-vs-one-sense-the-experiment)
   - [2.4 More Senses, More Precision](#24-more-senses-more-precision)
   - [2.5 The Second Sense Solves High Connectivity](#25-the-second-sense-solves-high-connectivity)
3. [Theoretical Foundations](#3-theoretical-foundations)
   - [3.2.1 The Birth of Senses](#321-the-birth-of-senses-perception-and-reasoning-are-two-separate-layers)
   - [3.2.2 Semantic Compression: From 89 Verbs to 15 Concepts](#322-semantic-compression-from-89-verbs-to-15-concepts)
4. [Central Theory: The Invariant is the Concept](#4-central-theory-the-invariant-is-the-concept)
   - [4.1 Context Subtracts, It Does Not Add](#41-context-subtracts-it-does-not-add)
   - [4.2 Context as Anchor: Solving Combinatorial Explosion](#42-context-as-anchor-solving-combinatorial-explosion)
5. [The Algorithm](#5-the-algorithm)
6. [Structure: The Invariant Relational Graph](#6-structure-the-invariant-relational-graph)
7. [Emergent Properties](#7-emergent-properties)
   - [7.3 Topological Honesty](#73-topological-honesty)
8. [Comparison with Large Language Models](#8-comparison-with-large-language-models)
9. [Known Limitations](#9-known-limitations)

---

## 1. The Problem

The goal was to build a system that **reasons**, not a chatbot.
The core insight: the problem was not the quantity of data. It was the architecture.

Existing approaches share a common paradigm: **knowledge is built by accumulation**.

- Embeddings sum co-occurrences into dense vectors
- Transformers accumulate statistical patterns over billions of tokens
- Cyc accumulates hand-written facts
- Hofstadter's architectures accumulate analogies

**No one inverted the sign.** No one said: knowledge is not built by adding what you see, but by **removing what does not repeat**. While graph pruning and difference operations exist in the literature (§1 note), those are topological cleanup steps applied after accumulation. Here, subtraction is the primary epistemic mechanism — it is how the system answers, not how it cleans up.

The IRG does not add layers. It subtracts noise:

| Traditional approach | IRG approach |
|---|---|
| More data = better approximation | More data = better intersection |
| Accumulate (sum embeddings) | Eliminate (intersect contexts) |
| Continuous statistical weights | Discrete connections (exists / does not exist) |
| Attention as external mechanism | Relevance ranked by invariance |
| Inspectability impossible | Every edge is navigable |
| Probabilistic "knowing that you know" | Deterministic "knowing that you know" |

**A note on terminology.** "Subtraction" in this report does not refer to removing edges from the graph — a topological operation known in the literature as graph differencing or pruning. Here, subtraction is an **epistemic operation at query time**: the graph retains all edges; the system answers by intersecting contexts, keeping only what survives two or more independent senses. The subtraction happens in the answer, not in the structure.

---

## 2. Core Principles

### 2.1 Invariance: What No One Had Seen

**No one ever said**: *"a concept is what remains invariant when you change the subject that observes it through the same relation."*

And above all, **no one translated it into an algorithm**:

```python
invariance(target) = len(incoming[target])
```

Three lines. Not a vector, not an embedding, not a probability. A pure count. If 47 subjects say *"X is a mammal"*, X is a mammal. If only 3 subjects say *"X is an idea"*, X is not an idea.

Invariance is the mathematical engine that transforms relations into concepts.

### 2.2 The Illusion of Pure Text

A thought experiment: read all of Wikipedia in German. You don't know German. You don't know how it's pronounced. No one ever said "Hallo, wie geht's?" to you. You can read 5 million entries — you will never understand anything.

Why? Because you have **a single dimension**: text. The invariant cannot emerge from a single line. You need two intersecting lines.

The child-and-milk example works because the child has **two senses**:

```
Sense 1 (sight):    sees the "white" of milk, snow, wall
Sense 2 (hearing):  hears the word "white" associated with those objects

Intersection:       "white" is the invariant. It is the concept.
```

With text alone, you have `word → follows → word → follows → word`. It is a linear chain. There is no second dimension to intersect. You cannot **subtract** because there is nothing to remove — every word appears only in the textual stream, never in a different sensory context.

Egyptian hieroglyphs were deciphered only because the Rosetta Stone provided the same text in three languages. **Three dimensions to intersect.** Without it, they would still be a mystery.

This explains why a graph built on pure textual "follows" edges produces noise: every word is connected to every other word within the same article, but the second sense that would make the true invariant emerge is missing. Topology alone is not enough if the data is one-dimensional.

**A conjecture.** Weight, in LLMs, may be a surrogate for the missing second dimension. Without a second sense, the system cannot intersect — so it sums. It accumulates probabilities instead of finding invariants. If this view is correct, weight is not the solution: it is the symptom of lacking multimodal grounding. The IRG is, in part, an attempt to test this conjecture.

| Dimension | Pure-text IRG | Multimodal IRG |
|---|---|---|
| Data | Only textual "follows" | Text + vision + audio + ... |
| Intersection | Impossible (single line) | Possible (multiple lines) |
| Invariant | Does not emerge | Emerges |
| Noise | Dominates (everything connected to everything) | Suppressed by intersection |
| Weights | Would be needed as surrogate | **Not needed** |

### 2.3 Two Senses vs. One Sense: the Experiment

```
╔═════════════════════════════════════════════════════════════╗
║           TWO SENSES (vision + text)                        ║
╠═════════════════════════════════════════════════════════════╣
║  Sense 1 (sight):    sees 🐱, 🐕, 🐱, 🦁                  ║
║  Sense 2 (text):     hears "cat", "dog", "cat", "lion"      ║
║                                                             ║
║  Intersection:       🐱 ∩ "cat" = invariant                 ║
║                      → "cat" IS the feline I see            ║
║                                                             ║
║  Result:             solid concept. Zero weights.           ║
╚═════════════════════════════════════════════════════════════╝


╔═════════════════════════════════════════════════════════════╗
║           ONE SENSE ONLY (text only)                        ║
╠═════════════════════════════════════════════════════════════╣
║  Text:   "cat is a mammal", "cat is an animal",             ║
║          "cat is a felid", "cat is a word",                 ║
║          "cat is like", "cat is in the"...                  ║
║                                                             ║
║  Intersection:   IMPOSSIBLE. There is only one dimension.   ║
║                  You cannot subtract — every occurrence     ║
║                  is true in its textual context.            ║
║                                                             ║
║  What to do?     Introduce a WEIGHT to distinguish.         ║
║                  P(mammal | cat, is) = 0.31                 ║
║                  P(animal  | cat, is) = 0.07                ║
║                  P(felid   | cat, is) = 0.03                ║
║                  P(word    | cat, is) = 0.42  ← noise       ║
║                                                             ║
║                  Weight is the SURROGATE for the            ║
║                  missing second dimension.                  ║
║                                                             ║
║  Result:          answers "word" (highest weight).          ║
║                   Statistically correct,                    ║
║                   semantically broken.                      ║
╚═════════════════════════════════════════════════════════════╝
```

With two senses: the invariant emerges. Zero weights.
With one sense only: weight dominates. Statistically correct answer, semantically wrong.
Weight is the price you pay when you cannot subtract.

#### 2.3.1 Subtraction is Robust to Noise (More So Than Addition)

A natural objection: isn't discrete subtraction fragile? If one sensor fails — "the cat drnk the milk" instead of "drinks" — doesn't the intersection collapse?

On the contrary. Subtraction is **more robust to noise than addition**, for the same reason an error-correcting code works: orthogonal redundancy.

```
Sense 1 (text):      "cat drinks milk", "cat drnk water"        ← one typo
Sense 2 (sight):     🐱 laps white liquid, 🐱 laps transparent liquid
Sense 3 (category):  cat → feline, cat → mammal

Intersection:        "drinks" appears in 2 out of 3 senses (correct text + sight)
                     "drnk" appears in 1 out of 3 senses (only wrong text)
                     
Result:              "drinks" survives. "drnk" is an isolated edge — it exists,
                     but no other sense confirms it. Subtraction
                     automatically ignores it.
```

Noise creates **isolated** edges. The invariant creates **converging** edges. The more senses you add, the more noise disperses and the invariant strengthens. This is the opposite of an LLM, where noise mixes with weights and pollutes all of them.

With N independent senses, the system tolerates up to N−1 errors on the same fact. Three senses = two can be wrong, the invariant survives. This is **orthogonal redundancy**: each sense is an independent dimension, and an error on one dimension does not propagate to the others.

This also resolves the **paradox of silence**: what happens if ALL senses fail simultaneously and the intersection is empty (∅)? The IRG answers "I don't know." This is not a bug — it is the **signature of epistemic humility**. An LLM, faced with emptiness, invents (hallucinates). The IRG recognizes emptiness and remains silent. This is exactly the behavior expected from a reliable system: when sensors are blind, the system must not guess. It must admit ignorance.

### 2.4 More Senses, More Precision

More senses = more dimensions to intersect = sharper concept.

```
Sense 1 (sight):     🐱
Sense 2 (text):      "cat"  
Sense 3 (hearing):   "meowww"  ← unique, unmistakable sound
                     Intersection: very solid concept
```

The meow is a **perfect discriminator**, while a knock on the wall is ambiguous:

```
Sound: "knock knock"
Hypotheses: {hammer, fist, child, chair, pipe, ...}

Add Sense 2 (sight): see a child near the wall
Hypotheses: {fist, child, ...}  ← the others SUBTRACTED

Add Sense 3 (touch): feel the vibration from the wall, not the floor
Hypotheses: {fist}  ← further SUBTRACTED
```

Every new sense is a **filter** that eliminates false matches. It does not add information — it removes ambiguity. This is the principle of **precision by subtraction**: the certainty of a concept does not depend on how much data you have accumulated, but on how many independent dimensions you have intersected.

### 2.5 The Second Sense Solves High Connectivity

A problem emerged during validation: words like "cat" and "water" have extremely high topological connectivity — they appear in thousands of sentences, with hundreds of different neighbors. The connectivity formula `(outgoing + incoming + incoming_diversity + outgoing_diversity) / count` classifies them as functionals, alongside "is," "of," "that."

But "cat" and "water" **exist in the real world**. You can touch them, see them, hear them. "Is" and "of" cannot. How does the IRG know?

The answer is structural, not heuristic. **A concept appears in more than one sense.** A functional appears in only one. This is the same principle as §2.3: the invariant emerges from the intersection of independent dimensions.

```
Sense 1 (text):       "the cat is a mammal"
Sense 2 (categories):  cat → feline, cat → mammal

Two senses → "cat" is a CONCEPT

Sense 1 (text):       "X is Y"
Sense 2 (categories):  never

One sense only → "is" is a FUNCTIONAL
```

What counts as a second sense is not fixed. It can be: a category list, a dictionary headword index, an object database, a set of article titles — any independent inventory of "things in the world." The requirement is only that the second sense be **orthogonal** to the first: it must not derive from the same textual co-occurrence.

The operational definition of "concept" is: **appears in more than one sense**. If yes, it is a concept. If no, it is a functional. This has been verified on synthetic test suites; large-scale benchmarking with real-world data is future work.

---

## 3. Theoretical Foundations

### 3.1 Words Are Not Enough

An LLM does not know what water is. It only knows which words appear near "water" in the training texts. For an Italian, "cold water = refreshing"; for a Chinese person raised with traditional medicine, "cold water = harmful to yang." The LLM cannot distinguish between a physical property and a cultural opinion. Both are statistical co-occurrences.

### 3.2 Sensory Grounding: Necessary for Universality, Not for Learning

An early exploration considered simulating the 5 senses. Each object would be a vector of measurable properties:

```
water    = {viscosity: 0.001, density: 1.0,  transparency: 0.95, ...}
mercury  = {viscosity: 0.001, density: 13.6, transparency: 0.0,  ...}
```

But a deaf child, without touch or smell, who has only sight and reads texts — how does she learn "water"? By co-occurrence, exactly like an LLM.

**Conclusion:** Sensory grounding is not needed to build concepts. It is needed to make them **universal** (language-independent) and **unambiguous**. Water and lemonade both have viscosity 0.001 — touch alone is not enough. The IRG accepts edges from any sensory source: it is natively multimodal.

#### 3.2.1 The Birth of Senses: Perception and Reasoning Are Two Separate Layers

An objection: «The IRG assumes already-discretized nodes. But the real world is pixels, decibels, molecules. Where does the first node come from?»

The answer is that **perception and reasoning are two distinct layers**, exactly as in the brain. The retina does not reason. The cortex does not see. Yet they collaborate.

The IRG is the **reasoning** layer. It receives already-discretized symbols and organizes them by subtraction. The **perception** layer — transforming pixels into 🐱, audio into "cat," Wikipedia titles into `category:feline` — is delegated to specialized sensors that can be:

- An LLM extracting `(cat, is_a, mammal)` from text
- A CNN classifying an image as `visual_cat_402`
- A parser extracting the title from a Wikipedia page
- A microphone detecting "meowww" and mapping it to `auditory_cat_17`

These sensors **can** be statistical (neural networks, embeddings, transformers). This is not a betrayal of the principle: it is the same division of labor that exists in nature. The retina uses analog photoreceptors. The primary visual cortex does statistical edge detection. But the prefrontal cortex, when reasoning about "cat," does so with discrete symbols.

The architecture is hybrid not by compromise, but by **efficiency**. The statistical sensor does immense work once: it transforms chaos into symbols. The logical reasoner does lightweight work infinitely many times: it navigates symbols without ever degrading.

```
┌─────────────────────────────────────────┐
│  REAL WORLD (pixels, sounds, texts)     │
├─────────────────────────────────────────┤
│  SENSORS (statistical, can err)         │  ← Extraction ONCE
├─────────────────────────────────────────┤
│  DISCRETE SYMBOLS (IRG nodes)           │
├─────────────────────────────────────────┤
│  IRG (logical, never errs)              │  ← Reasoning INFINITE TIMES
└─────────────────────────────────────────┘
```

And here lies the **scaling hypothesis**: a mediocre extractor × a perfect reasoner scales better than an excellent extractor integrated into a probabilistic reasoner. Because the extractor runs once. The reasoner runs on every query.

#### 3.2.2 Semantic Compression: From 89 Verbs to 15 Concepts

The extractor does more than parse triplets. It performs **semantic compression**: mapping surface relations onto a small set of abstract concepts. This is not reasoning — it is normalization at the sensor boundary.

```
Surface verb          →  Abstract concept
─────────────────────────────────────────
was born in, died in,
lived in, is in city   →  LOCATION

wrote, created, founded,
developed, composed    →  CREATION

member of, joined,
elected to, is part of →  MEMBERSHIP

won, received, was
awarded, has GDP       →  POSSESSION
```

In the current implementation, 89 surface relations collapse into 15 abstract concepts: `location`, `possession`, `creation`, `temporal`, `membership`, `identity`, `education`, `profession`, `influence`, `causation`, `relation`, `motion`, `combat`, `succession`, `communication`.

These 15 concepts are registered as **regular nodes** in the IRG. The engine treats `location` exactly as it treats `cat` or `Rome` — it runs the same intersection algorithm with the same O(deg) complexity. The engine has no privileged knowledge of which nodes are "abstract" and which are "concrete." The compression is complete before the first edge enters the graph.

This has two consequences:

1. **Queryability.** `location` is a navigable node. The query "which entities have a location relation?" is a single-neighborhood lookup: `incoming[location][instance]`.
2. **Composability.** Abstract concepts can be intersected: `creation ∩ possession` finds entities that both created and own something. This would be impossible if relations remained as heterogeneous surface strings.

The mapping is hand-curated and incomplete — new verbs from new domains will fall into the `other` catch-all. Automating this compression (via clustering of relation neighborhoods or LLM-based normalization) is future work.

### 3.3 "Knowing That You Know" and Recursion

An LLM does not know what it does not know. It hallucinates. A reasoning system must be able to say **"I don't know."** This requires self-inspection: looking at one's own connections and verifying whether they exist. This is recursion. The graph must contain a reference to itself.

### 3.4 Compound Objects

Water is not A molecule of H₂O. It is a set of ~10²³ molecules. These are two different concepts, connected by a relation. The system must represent both.

---

## 4. Central Theory: The Invariant is the Concept

### 4.1 Context Subtracts, It Does Not Add

The same mechanism, with context:

```
Question: "scientifically, what is a cat?"
Context words: {scientifically}

Step 1: find paths cat → ... → X
Step 2: for each X, verify: scientifically → ... → X ?

animal:        scientifically → ... → animal?       ✗ → ELIMINATED
mammal:        scientifically → ... → mammal?       ✗ → ELIMINATED
felid:         scientifically → ... → felid?        ✓ → SURVIVES

Result: felid
```

### 4.2 Context as Anchor: Solving Combinatorial Explosion

A naive pathfinding approach — "find all paths from A to B" — explodes combinatorially with graph size. The IRG avoids this through two mechanisms, both derived from the same principle: **intersect, don't traverse**.

**Direct intersection (no pathfinding).** For queries of the form "what do A and B share?", the system intersects their neighborhoods directly. This is O(deg(A) + deg(B)), with no path traversal — only the edges incident to A and B are touched. Shared concepts are ranked by invariance: the more subjects confirm a shared connection, the higher it ranks. The connection strength is the sum of the top shared invariances divided by the sum of A and B's own invariance.

**Multi-hop with context anchor.** When A and B share no direct neighbors, `find_path` performs a bidirectional BFS. But unlike standard BFS, each expansion step is gated by context: a node is only enqueued if its own neighborhood overlaps with the context anchor (the query's non-target words). This prunes branches that lead away from the topic. The first collision between the two frontiers yields a path; the path's strength is the minimum invariance along it.

**When intersection is empty.** If no shared neighbors exist and no path is found, the result is `empty=True` — the honest "I don't know."

The key property: complexity is bounded by degree, not by graph size. In tests with multi-million-node graphs, intersection completes in under a millisecond because it only touches the edges incident to A and B.

---

## 5. The Algorithm

The IRG has three core operations, all O(deg) and all operating on pure uint32 counts:

**Intersection** (`context_intersection`). Given two nodes A and B, extract their outgoing concept-neighbors (only nodes appearing in >1 sense, per §2.6). Intersect the two sets. Rank shared nodes by invariance. Return connection strength as the sum of top shared invariances divided by the sum of A and B's own invariance. If the intersection is empty, fall back to a bidirectional BFS gated by context words. Complexity: O(deg(A) + deg(B)). No pathfinding in the common case.

**Adaptive threshold** (`_adaptive_threshold`). To decide whether a connection is "solid" or "weak," the system samples 500 random pairs of nodes with similar invariance magnitude and computes their intersection strength. The 99th percentile of this distribution is the threshold. A connection below it is declared insufficient. The threshold is not a constant — it is recomputed from the graph's own topology and cached per magnitude bucket.

**Query** (`ask`). Tokenizes the question, identifies which tokens exist in the graph, and for each candidate concept either (a) ranks outgoing neighbors by invariance if no context is present, or (b) filters outgoing neighbors by intersection with the context words' neighborhoods if context is present. Also computes pairwise intersections between candidates.

All three operate with zero weights, zero embeddings, zero floating-point except the final strength ratio.

---

## 6. Structure: The Invariant Relational Graph

The IRG is a directed multigraph stored as three parallel uint32 arrays (source, relation, destination). Nodes and relations are pure integers; strings exist in a separate multilingual label layer. The graph topology is language-agnostic — the same node can have labels in Italian, English, and Chinese.

Writes are O(1) append to a delta buffer with deduplication. Periodic compaction merges the delta into the base arrays and rebuilds the outgoing index. The incoming index is built lazily on first query. A boolean mask marks nodes appearing in >1 sense (concepts, per §2.6); an invariance array caches each node's total incoming edge count.

The binary format (`.irgn`) stores a header, a string table (node_id, language, text), the three edge arrays, senses, and the relation map.

---

## 7. Emergent Properties

### 7.3 Topological Honesty

An LLM does not know that it does not know. It hallucinates with confidence even when it has no evidence. This is a structural property: the probabilistic architecture has no mechanism to declare the absence of knowledge.

The IRG does. And not because it is programmed to — because **the topology imposes it**.

Given two concepts A and B, the IRG finds the path connecting them. Then it computes:

```
strength = sum(top-10 shared invariance) / (invariance(A) + invariance(B))
```

If the strength is below the adaptive threshold — the 99th percentile of strengths from comparable node pairs, computed from the graph itself — the system declares: **"I do not have sufficient evidence."**

This is not a heuristic. It is a direct consequence of §2.1: if invariance defines the concept, a path with negligible invariance **is not a connection** — it is topological noise.

This is impossible in a Transformer. The LLM has no notion of "invariance" — it cannot distinguish a solid bridge from a noisy one because everything is a probability vector. The IRG can, because everything is a count of incoming edges.

Honesty is not a feature. It is an **emergent property of invariance**.

---

## 8. Comparison with Large Language Models

| Dimension | LLM | IRG |
|---|---|---|
| **What it learns** | Token distributions | Relation structure |
| **How it learns** | Sum (embedding) | Subtract (intersection) |
| **Attention** | Self-attention (external) | Relevance ranked by invariance |
| **"Knowing that you know"** | Proto-probabilistic self-inspection | Deterministic graph lookup (edge exists or it does not) |
| **"I don't know"** | May hallucinate it | **Guaranteed by topology** (strength < threshold) |
| **Inspectability** | Zero (opaque vectors) | Complete (navigable graph) |
| **Multimodality** | Requires special architectures | Native: same graph for everything |
| **Polysemy** | One ambiguous embedding | Context intersection narrows meaning |

The LLM **sums**. The IRG **subtracts**.

---

## 9. Known Limitations

### 9.1 Empty Graph

If the graph has not yet received edges, `intersect_contexts(A, B)` returns `empty=True`. The system answers "I do not have sufficient evidence." This is correct behavior but must be stated: the IRG **does not learn from zero** — it needs an extractor to populate the graph before it can reason.

### 9.2 All Paths Below Threshold

If, for a given query, paths exist but all have strength < adaptive_threshold, the system declares ignorance. This is honest but can frustrate: the user sees that "cat" and "mammal" are in the graph, but if the threshold is too high, the answer does not arrive. The precision-recall tradeoff is governed by the threshold percentile (default 99th). A lower percentile (e.g., 95th) increases recall but introduces false positives.

### 9.3 Contradiction Between Senses

If two senses produce contradictory edges (e.g., `cat --[is]--> mammal` from text and `cat --[is]--> reptile` from a broken sensor), the IRG has no conflict resolution mechanism. Both edges coexist. The correct path will have higher invariance (more subjects traverse it), but there is no explicit "vote." Contradictions are resolved only statistically, not logically.

### 9.4 Ambiguous Names

If "Roma" is both the city and the football team, the IRG treats them as a single node. Neighbors will include both `Colosseum` and `Totti`. Context intersection mitigates the issue (if you query "Roma" ∩ "football," Totti emerges), but polysemy is not architecturally resolved — it requires the extractor to produce distinct nodes (`Roma_city`, `Roma_team`).

### 9.5 Scalability to Real Senses

The two-senses principle has been demonstrated with pairs (text, categories) and (text, Wikipedia titles). With 5+ real senses (vision, audio, touch, temperature, pressure), the graph would grow by orders of magnitude. O(deg) complexity holds as long as the average degree per node remains below ~10,000. Beyond that, even intersection slows down. This has not been tested.

### 9.6 Update and Retraction

The IRG is **append-only**. Correction, obsolescence, and versioning are open problems.

---
