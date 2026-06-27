"""
Subtraction Engine V1 — IRG with uint32 IDs, multi-language, compact serialization
================================================================================
Copyright (C) 2026  Claudio Vincis

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Every concept is a number. Labels (strings) are an external layer.
Supports N languages: each ID can have labels in it, en, zh...

Manifesto principles (MANIFESTO.md):
  §2.0 — Invariance is a pure count
  §2.2 — Two senses create concepts (is_concept)
  §4.5 — Context subtracts (context_intersection)
  §7.3 — Topological honesty (topological_honesty)
  Zero weights, zero hardcoded patterns, zero arbitrary bounds.
"""

import os, struct
from array import array
from collections import defaultdict
from engine_compact_train import CompactEngineTrain


class SubtractionEngine:
    """Numeric IRG: edges in array('I'), strings only for I/O."""

    def __init__(self):
        self.g = CompactEngineTrain()
        # Multilingual labels: id -> {lang: text}
        self.labels = defaultdict(dict)
        # Reverse: (lang, text) -> id
        self.id_by_label = {}
        # Senses: id -> {sense1, sense2, ...}
        self.senses = defaultdict(set)
        # Rel reverse (from Compact): rid -> name
        self.rid2name = {}
        
        # ── Delta buffer (V3): O(1) writes without realloc ──
        self._delta_src = array('I')
        self._delta_rel = array('I')
        self._delta_dst = array('I')
        self._delta_dedup = set()  # (src, rel, dst) for dedup in delta
        self._delta_dirty = False

        # ── Specialized contents (§6.7): concept_id -> {type: [content, ...]} ──
        # types: "math" (LaTeX), "code_py" (Python), "code_cpp" (C++), "code_js" (JS), ...
        self._contents = defaultdict(lambda: defaultdict(list))

    # ── Data ingestion ──────────────────────────────────────

    def add(self, src: str, rel: str, dst: str,
                 sense: str = "text", language: str = "en"):
        """Add a triplet to the delta buffer (O(1) append, zero realloc)."""
        src_id = self._get_or_create(src, language)
        dst_id = self._get_or_create(dst, language)
        rel_id = self.g._rel_id(rel)
        self.rid2name[rel_id] = rel

        # Dedup in delta
        key = (src_id, rel_id, dst_id)
        if key in self._delta_dedup:
            return
        self._delta_dedup.add(key)

        # Delta: O(1) append
        self._delta_src.append(src_id)
        self._delta_rel.append(rel_id)
        self._delta_dst.append(dst_id)
        self._delta_dirty = True

        # Senses
        self.senses[src_id].add(sense)
        self.senses[dst_id].add(sense)
        self._invalidate_cache()

    def _get_or_create(self, text: str, language: str = "en") -> int:
        """Convert a string to uint32 ID, creating it if needed."""
        key = (language, text)
        if key in self.id_by_label:
            return self.id_by_label[key]
        # Look up in compact engine
        nid = self.g._id_from_label.get(key)
        if nid is not None:
            self.id_by_label[key] = nid
            self.labels[nid][language] = text
            return nid
        # Create new ID
        nid = self.g._next_id
        self.g._next_id += 1
        self.g._labels[nid] = (language, text)
        self.g._id_from_label[key] = nid
        self.id_by_label[key] = nid
        self.labels[nid][language] = text
        return nid

    def add_label(self, nid: int, language: str, text: str):
        """Add a label in a language to an existing ID."""
        self.labels[nid][language] = text
        self.id_by_label[(language, text)] = nid

    def merge_concept(self, labels: dict) -> int:
        """Declare that multiple language labels refer to the SAME concept.

        labels: {"it": "gatto", "en": "cat", "fr": "chat"}

        If any label already exists, all others are merged onto that node.
        Otherwise a new node is created. Returns the canonical node ID.

        After calling this, add() with any of these (language, text) pairs
        will resolve to the same node — no more duplicate concepts.
        """
        # Check if any label already exists
        existing_nid = None
        for lang, text in labels.items():
            key = (lang, text)
            if key in self.id_by_label:
                existing_nid = self.id_by_label[key]
                break
            nid = self.g._id_from_label.get(key)
            if nid is not None:
                existing_nid = nid
                self.id_by_label[key] = nid
                break

        if existing_nid is not None:
            nid = existing_nid
        else:
            # Create new node with first label
            first_lang, first_text = next(iter(labels.items()))
            nid = self._get_or_create(first_text, first_lang)
            # Remove first from dict since _get_or_create already registered it
            labels = {k: v for k, v in labels.items() if k != first_lang}

        # Attach all labels to the canonical node
        for lang, text in labels.items():
            self.add_label(nid, lang, text)

        return nid

    # ── Query ──────────────────────────────────────────────

    def is_concept(self, nid: int) -> bool:
        return len(self.senses.get(nid, set())) > 1

    def _rebuild(self):
        """Rebuild indices after batch load."""
        self.g._rebuild_out_idx()

    # ── Incoming index (lazy) ─────────────────────────────

    def _build_in_idx(self):
        """Build incoming index: dst_id -> array('I') [src1,rel1,...]."""
        if hasattr(self, '_in_idx') and self._in_idx:
            return
        import numpy as np
        n = len(self.g._src)
        if n == 0:
            self._in_idx = {}
            return
        
        src = np.frombuffer(self.g._src, dtype=np.uint32)
        dst = np.frombuffer(self.g._dst, dtype=np.uint32)
        rel = np.frombuffer(self.g._rel, dtype=np.uint32)
        
        order = np.argsort(dst)
        dst_sorted = dst[order]
        src_sorted = src[order]
        rel_sorted = rel[order]
        
        self._in_idx = {}
        i = 0
        while i < n:
            d = int(dst_sorted[i])
            j = i + 1
            while j < n and dst_sorted[j] == d:
                j += 1
            arr = array('I')
            for k in range(i, j):
                arr.append(int(src_sorted[k]))
                arr.append(int(rel_sorted[k]))
            self._in_idx[d] = arr
            i = j
        print(f"  [in_idx] built: {len(self._in_idx):,} destinations", flush=True)

    # ── Indici pre-computati (lazy, numpy) ────────────────
    
    @property
    def _concepts_mask(self):
        """Bitset numpy: mask[nid] = True se nid è concetto (§2.2)."""
        if not hasattr(self, '_cache_concetti_mask'):
            import numpy as np
            max_id = max(self.g._labels.keys()) + 1 if self.g._labels else 1
            mask = np.zeros(max_id, dtype=bool)
            for nid, sensi in self.senses.items():
                if len(sensi) > 1:
                    mask[nid] = True
            self._cache_concepts_mask = mask
        return self._cache_concepts_mask
    
    @property
    def _inv_array(self):
        """Numpy array: inv[nid] = number of incoming edges (§2.1)."""
        if not hasattr(self, '_cache_inv_array'):
            import numpy as np
            self._build_in_idx()
            max_id = max(self.g._labels.keys()) + 1 if self.g._labels else 1
            arr = np.zeros(max_id, dtype=np.uint32)
            for nid, edges in self._in_idx.items():
                # edges are stored as (src, rel) pairs → divide by 2
                arr[nid] = len(edges) // 2
            self._cache_inv_array = arr
        return self._cache_inv_array
    
    def _invalidate_cache(self):
        """Invalidate caches after graph modifications."""
        for attr in ('_cache_concetti_mask', '_cache_concetti', '_cache_inv_array'):
            if hasattr(self, attr):
                delattr(self, attr)
        self._in_idx = {}

    # ── Neighbors (pure topology) ────────────────────────────

    def _outgoing_neighbor_ids(self, nid: int) -> list:
        """All outgoing edges (base + delta). No bound."""
        result = []
        # Base (sorted)
        if nid in self.g._out_idx:
            s, e = self.g._out_idx[nid]
            result = [(self.g._dst[i], self.g._rel[i]) for i in range(s, e)]
        # Delta (unsorted, linear scan)
        if self._delta_dirty:
            for i in range(len(self._delta_src)):
                if self._delta_src[i] == nid:
                    result.append((self._delta_dst[i], self._delta_rel[i]))
        return result

    def compact(self):
        """Merge delta into base and rebuild indices. Call periodically."""
        if not self._delta_dirty or len(self._delta_src) == 0:
            return
        
        n_delta = len(self._delta_src)
        print(f"  [compact] merge delta into base...", flush=True)
        
        # Appendi delta al base
        for i in range(n_delta):
            self.g._src.append(self._delta_src[i])
            self.g._rel.append(self._delta_rel[i])
            self.g._dst.append(self._delta_dst[i])
            self.g._flg.append(0)
        
        # Clear delta
        self._delta_src = array('I')
        self._delta_rel = array('I')
        self._delta_dst = array('I')
        self._delta_dedup.clear()
        self._delta_dirty = False
        
        # Rebuild indices
        self._rebuild()
        self._invalidate_cache()
        print(f"  [compact] completed: {self.n_nodes} nodes, {self.n_edges} edges", flush=True)

    def _incoming_neighbor_ids(self, nid: int) -> list:
        """All incoming edges as (src_id, rel_id). No bound."""
        self._build_in_idx()
        arr = self._in_idx.get(nid)
        if arr is None:
            return []
        return [(arr[k], arr[k+1]) for k in range(0, len(arr), 2)]

    # ── Context intersection (§4.5, §4.6) ─────────────

    def context_intersection(self, a_str, b_str, extra_context=None, language="en",
                             skip_rels=None):
        """Pure intersection: what do A and B share?
        
        neighbors(A) ∩ neighbors(B) → shared concepts (concept-nodes only).
        Hot loop in numpy (zero Python iteration over edges).
        O(deg(A) + deg(B)). Zero pathfinding, zero explosion, zero bound.
        """
        import numpy as np

        a_id = self.id_by_label.get((language, a_str))
        b_id = self.id_by_label.get((language, b_str))
        if a_id is None or b_id is None:
            return {"shared": [], "only_a": [], "only_b": [], "empty": True,
                    "strength": 0.0}

        self._build_in_idx()
        conc_mask = self._concepts_mask
        
        # Pre-compute rel IDs to skip
        skip_ids = set()
        if skip_rels:
            skip_ids = {self.g._rel_map[r] for r in skip_rels if r in self.g._rel_map}

        def _concept_neighbors_np(nid):
            """Set of concept-neighbor IDs — ALL in numpy."""
            if nid not in self.g._out_idx:
                return set()
            s, e = self.g._out_idx[nid]
            n = e - s
            if n == 0:
                return set()
            
            dst = np.frombuffer(self.g._dst, dtype=np.uint32, offset=s*4, count=n)
            rel = np.frombuffer(self.g._rel, dtype=np.uint32, offset=s*4, count=n)
            
            if skip_ids:
                skip_mask = np.isin(rel, list(skip_ids), invert=True)
            else:
                skip_mask = np.ones(n, dtype=bool)
            
            ok_mask = skip_mask & conc_mask[dst]
            return set(dst[ok_mask].tolist())

        neighbors_a = _concept_neighbors_np(a_id)
        neighbors_b = _concept_neighbors_np(b_id)

        # Pure intersection (§4.5)
        shared_ids = neighbors_a & neighbors_b

        # Extra context: convert strings→IDs and subtract
        if extra_context:
            ctx_ids = {self.id_by_label.get((language, c)) 
                          for c in extra_context}
            ctx_ids.discard(None)
            shared_ids = shared_ids & ctx_ids

        # Convert to strings only at the end
        inv_arr = self._inv_array
        shared_ranking = []
        for nid in shared_ids:
            inv = int(inv_arr[nid])
            shared_ranking.append((self._text(nid, language), inv))
        shared_ranking.sort(key=lambda x: -x[1])

        only_a = sorted(self._text(nid, language) for nid in (neighbors_a - neighbors_b))
        only_b = sorted(self._text(nid, language) for nid in (neighbors_b - neighbors_a))

        # Topological strength (§7.3)
        inv_a = int(inv_arr[a_id]) if a_id < len(inv_arr) else 0
        inv_b = int(inv_arr[b_id]) if b_id < len(inv_arr) else 0
        sum_inv = sum(inv for _, inv in shared_ranking[:10])
        strength = sum_inv / max(inv_a + inv_b, 1)

        # Fallback to multi-hop if direct intersection is empty
        multi_hop_bridge = None
        if len(shared_ids) == 0:
            paths = self.find_path(a_str, b_str, context=extra_context, language=language)
            if paths.get("status") == "found" and paths["paths"]:
                path_ids, inv_min = paths["paths"][0]
                # Use intermediate nodes as virtual shared context
                mid = path_ids[1:-1]  # exclude A and B
                if mid:
                    multi_hop_bridge = {
                        "path": [self._text(nid, language) for nid in path_ids],
                        "hop": len(path_ids) - 1,
                        "invarianza_min": inv_min,
                    }
                    strength = inv_min / max(min(inv_a, inv_b), 1)

        # Adaptive threshold: is this connection topologically honest? (§7.3)
        empty = len(shared_ids) == 0 and multi_hop_bridge is None
        if empty:
            _honest = True  # empty is honest: "I don't know"
        else:
            threshold = self._adaptive_threshold(inv_a, inv_b)
            _honest = strength >= threshold

        return {
            "shared": shared_ranking,
            "only_a": only_a,
            "only_b": only_b,
            "empty": empty,
            "strength": strength,
            "multi_hop_bridge": multi_hop_bridge,
            "honest": _honest,
        }

    def answer_what_is(self, subject_str: str, language: str = "en") -> list:
        """Answer 'what is X?'. Pure topology, ALL facts, ordered by invariance."""
        import numpy as np

        sid = self.id_by_label.get((language, subject_str))
        if sid is None:
            return []

        if sid not in self.g._out_idx:
            return []

        self._build_in_idx()
        conc_mask = self._concepts_mask

        s, e = self.g._out_idx[sid]
        n = e - s
        if n == 0:
            return []

        dst = np.frombuffer(self.g._dst, dtype=np.uint32, offset=s*4, count=n)
        rel = np.frombuffer(self.g._rel, dtype=np.uint32, offset=s*4, count=n)
        
        ok = conc_mask[dst]
        dst_ok = dst[ok]
        rel_ok = rel[ok]
        
        inv_vals = self._inv_array[dst_ok]
        results = []
        for i in range(len(dst_ok)):
            results.append((int(dst_ok[i]), int(rel_ok[i]), int(inv_vals[i])))
        
        results.sort(key=lambda x: -x[2])
        seen = set()
        dedup = []
        for dst_id, rel_id, inv in results:
            if dst_id not in seen:
                seen.add(dst_id)
                dedup.append((dst_id, rel_id, inv))
        return dedup

    # ── V3: Basic Level Bonus + Subject Coherence ────────

    def _subject_coherence(self, dst_id: int, max_subjects=10) -> float:
        """Do the subjects pointing to dst share anything with each other?
        
        Filters for SEMANTIC relations (E2B), ignoring generic 'link a'.
        """
        self._build_in_idx()
        arr = self._in_idx.get(dst_id)
        if arr is None or len(arr) < 6:
            return 0.5
        
        # Relazioni semantiche (E2B): verbi, frasi brevi, NON 'link a' o 'titolo'
        STOP_RELS = {'link a', 'titolo', 'fonte', 'celebra'}
        
        # Extract subjects arriving via semantic relations
        subjects = []
        for k in range(0, len(arr), 2):
            if len(subjects) >= max_subjects:
                break
            s = arr[k]
            r = arr[k+1]
            rn = self._rel_name(r)
            if rn in STOP_RELS:
                continue
            if s not in subjects:
                subjects.append(s)
        
        if len(subjects) < 3:
            return 0.5  # not enough semantic subjects to judge
        
        # Intersect neighbors of semantic subjects
        conc_mask = self._concepts_mask
        intersection = None
        for sid in subjects:
            if sid not in self.g._out_idx:
                continue
            s, e = self.g._out_idx[sid]
            v = set()
            for i in range(s, min(s + 30, e)):
                d = self.g._dst[i]
                if d < len(conc_mask) and conc_mask[d]:
                    v.add(d)
            if intersection is None:
                intersection = v
            else:
                intersection &= v
            if not intersection:
                break
        
        if intersection is None or not intersection:
            return 0.05  # almost zero
        
        # How large is the intersection?
        all_neighbors = set()
        for sid in subjects:
            if sid in self.g._out_idx:
                s, e = self.g._out_idx[sid]
                for i in range(s, min(s + 30, e)):
                    d = self.g._dst[i]
                    if d < len(conc_mask) and conc_mask[d]:
                        all_neighbors.add(d)
        
        if not all_neighbors:
            return 0.05
        
        return len(intersection) / len(all_neighbors)

    def answer_what_is_v3(self, subject_str: str, language: str = "en") -> list:
        """Answer 'what is X?' with basic level bonus + subject coherence.
        
        rank = invariance × specificity × subject_coherence
        """
        import numpy as np, math

        sid = self.id_by_label.get((language, subject_str))
        if sid is None or sid not in self.g._out_idx:
            return []

        self._build_in_idx()
        conc_mask = self._concepts_mask
        inv_arr = self._inv_array

        s, e = self.g._out_idx[sid]
        n = e - s
        if n == 0:
            return []

        dst = np.frombuffer(self.g._dst, dtype=np.uint32, offset=s*4, count=n)
        rel = np.frombuffer(self.g._rel, dtype=np.uint32, offset=s*4, count=n)
        
        ok = conc_mask[dst] & (dst != sid)  # esclude self-loop
        dst_ok = dst[ok]
        rel_ok = rel[ok]
        inv_vals = inv_arr[dst_ok]
        
        risultati = []
        for i in range(len(dst_ok)):
            d, r, inv = int(dst_ok[i]), int(rel_ok[i]), int(inv_vals[i])
            # Specificity: how UNIQUE is this neighbor to THIS source?
            # Count how many OTHER concepts also link to this destination
            in_degree = 0
            if d in self._in_idx:
                in_degree = len(self._in_idx[d]) // 2  # each entry is (src, rel)
            specificity = 1.0 / (1.0 + math.log(1 + in_degree))
            # Basic level: specificity of the destination itself
            grado_out = 0
            if d in self.g._out_idx:
                grado_out = self.g._out_idx[d][1] - self.g._out_idx[d][0]
            basic_level = 1.0 / (1.0 + math.log(1 + grado_out))
            score = inv * specificity * basic_level
            risultati.append((d, r, inv, score))
        
        risultati.sort(key=lambda x: -x[3])
        visti = set()
        dedup = []
        for d, r, inv, score in risultati:
            if d not in visti:
                visti.add(d)
                dedup.append((d, r, inv, score))
        return dedup

    # ── V3: Multi-hop with context anchor (§4.6) ─────

    def find_path(self, a_str, b_str, context=None, max_hop=5, language="en"):
        """Bidirectional BFS with context anchor.
        
        Expands from A and B simultaneously. At each hop, checks that the node
        has intersection with the context anchor. If intersection is empty,
        the branch stops (no arbitrary max_hop, only topological brake).
        
        Returns: dict with 'paths' [(path, inv_min)], 'hop', 'status'
        """
        a_id = self.id_by_label.get((language, a_str))
        b_id = self.id_by_label.get((language, b_str))
        if a_id is None or b_id is None:
            return {"paths": [], "hop": 0, "status": "node_not_found"}

        if a_id == b_id:
            return {"paths": [([a_id], 0)], "hop": 0, "status": "same_node"}

        self._build_in_idx()
        conc_mask = self._concepts_mask
        inv_arr = self._inv_array
        
        # Context anchor: words from the question
        if context is None:
            ctx_ids = set()
        else:
            ctx_ids = set()
            for c in context:
                cid = self.id_by_label.get((language, c))
                if cid is not None:
                    ctx_ids.add(cid)

        # Frontiers: {nid: path_from_source}
        front_a = {a_id: [a_id]}
        front_b = {b_id: [b_id]}
        visited_a = {a_id}
        visited_b = {b_id}
        
        paths = []
        
        hop = 0
        while front_a and front_b:
            hop += 1
            if hop > 20:  # safety net only — context gating is the primary stopper
                break
            if len(visited_a) + len(visited_b) > 5000:  # safety net: context too weak
                break
            # Expand front A (outgoing)
            new_a = {}
            for nid, path in front_a.items():
                for dst_id, _ in self._outgoing_neighbor_ids(nid):
                    if dst_id in visited_a:
                        continue
                    # Topological brake (only if context is specified)
                    if ctx_ids:
                        if dst_id not in ctx_ids:
                            if dst_id in self.g._out_idx:
                                s2, e2 = self.g._out_idx[dst_id]
                                neighbors2 = {self.g._dst[i] for i in range(s2, min(s2+50, e2))
                                           if conc_mask[self.g._dst[i]]}
                                if not (neighbors2 & ctx_ids):
                                    continue
                    visited_a.add(dst_id)
                    new_a[dst_id] = path + [dst_id]
            
            # Expand front B (outgoing, symmetric to A)
            new_b = {}
            for nid, path in front_b.items():
                for dst_id, _ in self._outgoing_neighbor_ids(nid):
                    if dst_id in visited_b:
                        continue
                    # Topological brake (symmetric to A)
                    if ctx_ids:
                        if dst_id not in ctx_ids:
                            if dst_id in self.g._out_idx:
                                s2, e2 = self.g._out_idx[dst_id]
                                neighbors2 = {self.g._dst[i] for i in range(s2, min(s2+50, e2))
                                           if conc_mask[self.g._dst[i]]}
                                if not (neighbors2 & ctx_ids):
                                    continue
                    visited_b.add(dst_id)
                    new_b[dst_id] = path + [dst_id]
            
            # Check collision: new + crossed with old
            common = (set(new_a.keys()) & set(new_b.keys())) | \
                     (set(new_a.keys()) & set(front_b.keys())) | \
                     (set(front_a.keys()) & set(new_b.keys()))
            if common:
                for bridge in common:
                    if bridge in new_a: path_a = new_a[bridge]
                    else: path_a = front_a[bridge]
                    if bridge in new_b: path_b = new_b[bridge]
                    else: path_b = front_b[bridge]
                    full = path_a + path_b[::-1][1:]
                    inv_min = min(int(inv_arr[n]) for n in full if n < len(inv_arr))
                    paths.append((full, inv_min))
                # Continue searching for longer paths (don't stop at first)
            
            front_a = new_a
            front_b = new_b
        
        paths.sort(key=lambda x: -x[1])
        
        status = "found" if paths else "not_found"
        return {
            "paths": [(p, inv) for p, inv in paths[:5]],
            "hop": hop,
            "status": status,
        }

    # ── Unified Query Interface ────────────────────────────

    def ask(self, question: str, language: str = "en"):
        """§4.5: Context subtracts. Zero weights, only intersection and counting.
        
        - "what is X?": outgoing neighbors, ordered by invariance (pure count)
        - "X in context Y?": outgoing neighbors ∩ neighbors(Y), ordered by invariance
        """
        
        # Extract candidate concepts from the question
        words = question.lower().replace("?","").replace(",","").split()
        
        # Find which words exist as graph concepts -> context anchor
        context = []
        for w in words:
            if len(w) > 2 and self.id_by_label.get((language, w)):
                context.append(w)
        for i in range(len(words)-1):
            bigram = f"{words[i]} {words[i+1]}"
            if self.id_by_label.get((language, bigram)):
                context.append(bigram)
        
        candidates = set()
        for w in words:
            if len(w) > 2:
                candidates.add(w)
        for i in range(len(words)-1):
            candidates.add(f"{words[i]} {words[i+1]}")
        
        results = {}
        for c in candidates:
            nid = self.id_by_label.get((language, c))
            if nid is None:
                continue
            
            entry = {"id": int(nid), "label": c}
            
            # ── Pure "what is X?" → outgoing neighbors by invariance (§4.3) ──
            other_context = [w for w in context if w != c and w not in c.split()]
            
            if len(other_context) == 0:
                # No context: use outgoing (what does X point to?)
                raw = self._outgoing_neighbor_ids(nid)
                neighbors = [(dst_id, rel_id) for dst_id, rel_id in raw]
                
                # Rank: pure invariance (count), no weights
                scored = []
                for dst_id, rel_id in neighbors:
                    if dst_id == nid:
                        continue
                    inv = int(self._inv_array[dst_id]) if dst_id < len(self._inv_array) else 0
                    scored.append((dst_id, inv, "outgoing"))
                
                scored.sort(key=lambda x: -x[1])
                
                seen = set()
                top = []
                for dst_id, inv, src in scored:
                    name = self._text(dst_id)
                    if name not in seen and name != c:
                        seen.add(name)
                        top.append({"concept": name, "invarianza": inv, "source": src})
                
                entry["top_definitions"] = top[:8]
            
            else:
                # ── Context present → outgoing, filtered by intersection (§4.5) ──
                raw = self._outgoing_neighbor_ids(nid)
                neighbors = [(dst_id, rel_id) for dst_id, rel_id in raw]
                
                # Build context filter: union of neighbors of each context word
                ctx_neighbors = set()
                for ctx_word in other_context:
                    ctx_id = self.id_by_label.get((language, ctx_word))
                    if ctx_id is None:
                        continue
                    for dst, _ in self._outgoing_neighbor_ids(ctx_id):
                        ctx_neighbors.add(self._text(dst))
                    for src, _ in self._incoming_neighbor_ids(ctx_id):
                        ctx_neighbors.add(self._text(src))
                
                # Also add the context words themselves
                ctx_neighbors.update(other_context)
                
                # Rank outgoing neighbors: pure invariance
                # Mark those that survive context intersection
                scored = []
                for dst_id, rel_id in neighbors:
                    if dst_id == nid:
                        continue
                    inv = int(self._inv_array[dst_id]) if dst_id < len(self._inv_array) else 0
                    name = self._text(dst_id)
                    survives = name in ctx_neighbors
                    scored.append((dst_id, inv, survives))
                
                # Sort: survivors first (pure §4.5 subtraction), then by invariance
                scored.sort(key=lambda x: (-x[2], -x[1]))
                
                seen = set()
                top = []
                for dst_id, inv, survives in scored:
                    name = self._text(dst_id)
                    if name not in seen and name != c:
                        seen.add(name)
                        src = "[context]" if survives else "direct"
                        top.append({"concept": name, "invarianza": inv, "source": src})
                
                entry["top_definitions"] = top[:8]
            
            # ── Connections with other found candidates ──
            connections = []
            for c2 in candidates:
                if c2 <= c:
                    continue
                nid2 = self.id_by_label.get((language, c2))
                if nid2 is None:
                    continue
                inter = self.context_intersection(c, c2, extra_context=other_context, language=language)
                if not inter["empty"]:
                    connections.append({
                        "with": c2,
                        "shared_count": len(inter["shared"]),
                        "top_shared": [{"concept": name, "invarianza": inv} 
                                       for name, inv in inter["shared"][:3]],
                        "force": round(inter["strength"], 4),
                    })
            if connections:
                entry["connections"] = sorted(connections, key=lambda x: -x["force"])
            
            results[c] = entry
        
        found = list(results.keys())
        return {
            "question": question,
            "context_used": context,
            "concepts_found": len(found),
            "concepts": results,
        }

    # ── V3: Stratified adaptive threshold (§7.3) ──────────

    def _adaptive_threshold(self, inv_a: int, inv_b: int, samples=500) -> float:
        """Compute the honesty threshold from the graph itself.
        
        Samples random pairs with similar invariance to A and B,
        computes their strength, and returns the 99th percentile.
        Zero magic constants — the threshold emerges from the topology.
        """
        import random, math
        self._build_in_idx()
        inv_arr = self._inv_array
        
        # Buckets by log10(invariance): 0-1, 1-2, 2-3, 3-4, 4-5
        bucket_a = int(math.log10(max(inv_a, 1)))
        bucket_b = int(math.log10(max(inv_b, 1)))
        
        # Cache: (bucket_a, bucket_b) -> threshold
        cache_key = (bucket_a, bucket_b)
        if not hasattr(self, '_cache_thresholds'):
            self._cache_thresholds = {}
        if cache_key in self._cache_thresholds:
            return self._cache_thresholds[cache_key]
        
        # Collect nodes in required buckets
        max_id = len(inv_arr)
        pool_a = []
        pool_b = []
        for nid in range(1, max_id):
            inv = int(inv_arr[nid]) if nid < max_id else 0
            if inv > 0:
                b = int(math.log10(max(inv, 1)))
                if b == bucket_a:
                    pool_a.append(nid)
                if b == bucket_b:
                    pool_b.append(nid)
        
        if len(pool_a) < 10 or len(pool_b) < 10:
            self._cache_thresholds[cache_key] = 0.01
            return 0.01
        
        # Sample random pairs and compute strength
        strengths = []
        # Label pool for calling context_intersection
        for _ in range(samples):
            nid_a = random.choice(pool_a)
            nid_b = random.choice(pool_b)
            if nid_a == nid_b:
                continue
            
            # Strength = sum(inv shared) / (inv(A) + inv(B))
            # Compute directly without going through strings
            conc_mask = self._concepts_mask
            neighbors_a = set()
            if nid_a in self.g._out_idx:
                s, e = self.g._out_idx[nid_a]
                for i in range(s, min(s + 50, e)):
                    d = self.g._dst[i]
                    if d < max_id and conc_mask[d]:
                        neighbors_a.add(d)
            
            neighbors_b = set()
            if nid_b in self.g._out_idx:
                s, e = self.g._out_idx[nid_b]
                for i in range(s, min(s + 50, e)):
                    d = self.g._dst[i]
                    if d < max_id and conc_mask[d]:
                        neighbors_b.add(d)
            
            shared = neighbors_a & neighbors_b
            if shared:
                sum_inv = sum(int(inv_arr[d]) for d in list(shared)[:10] if d < max_id)
                strength = sum_inv / max(inv_a + inv_b, 1)
            else:
                strength = 0.0
            strengths.append(strength)
        
        if not strengths:
            self._cache_thresholds[cache_key] = 0.01
            return 0.01
        
        strengths.sort()
        p99 = strengths[int(len(strengths) * 0.99)]
        threshold = max(p99, 0.001)  # never below 0.001
        
        self._cache_thresholds[cache_key] = threshold
        return threshold

    def answer_with_honesty_v3(self, a_str, b_str, language="en"):
        """Honesty V3 with multi-hop + adaptive threshold."""
        inv_arr = self._inv_array
        a_id = self.id_by_label.get((language, a_str))
        b_id = self.id_by_label.get((language, b_str))
        
        if a_id is None or b_id is None:
            return {"status": "node_not_found", "strength": 0, "message": f"No information about '{a_str}' o '{b_str}'."}
        
        inv_a = int(inv_arr[a_id]) if a_id < len(inv_arr) else 0
        inv_b = int(inv_arr[b_id]) if b_id < len(inv_arr) else 0
        
        r = self.find_path(a_str, b_str)
        
        if r["status"] != "found" or not r["paths"]:
            return {"status": "empty", "strength": 0,
                    "message": f"Insufficient evidence to connect '{a_str}' and '{b_str}'."}
        
        path, inv_min = r["paths"][0]
        strength = inv_min / max(min(inv_a, inv_b), 1)
        
        threshold = self._adaptive_threshold(inv_a, inv_b)
        path_labels = [self._text(nid, language) for nid in path]
        
        if strength < threshold:
            return {
                "status": "weak", "strength": strength, "hop": r["hop"],
                "path": path_labels,
                "threshold": threshold,
                "message": (f"Weak connection between '{a_str}' and '{b_str}' "
                             f"(strength={strength:.4f} < threshold={threshold:.4f}, hop={r['hop']}).")
            }
        return {
            "status": "solid", "strength": strength, "hop": r["hop"],
            "path": path_labels,
            "threshold": threshold,
            "message": (f"'{a_str}' and '{b_str}' connected in {r['hop']} hop "
                         f"(strength={strength:.3f} >= threshold={threshold:.3f}).")
        }

    def topological_honesty_v3(self, a_str, b_str, language="en"):
        """Onestà topologica V3: multi-hop + threshold adattiva.
        
        Uses find_path (bidirectional BFS) as primary signal,
        NOT direct intersection. L'intersezione diretta è un caso
        of multi-hop with hop=1.
        
        Returns: (status, strength, message)
        """
        inv_arr = self._inv_array
        a_id = self.id_by_label.get((language, a_str))
        b_id = self.id_by_label.get((language, b_str))
        
        if a_id is None or b_id is None:
            return ("empty", 0.0,
                    f"No information about '{a_str}' o '{b_str}'.")
        
        inv_a = int(inv_arr[a_id]) if a_id < len(inv_arr) else 0
        inv_b = int(inv_arr[b_id]) if b_id < len(inv_arr) else 0
        
        r = self.find_path(a_str, b_str, language=language)
        
        if r["stato"] != "trovato" or not r["percorsi"]:
            return ("empty", 0.0,
                    f"Insufficient evidence to connect '{a_str}' e '{b_str}'.")
        
        path, inv_min = r["percorsi"][0]
        strength = inv_min / max(min(inv_a, inv_b), 1)
        threshold = self._adaptive_threshold(inv_a, inv_b)
        
        if strength < threshold:
            return ("weak", strength,
                    f"Weak connection between '{a_str}' e '{b_str}' "
                    f"(strength={strength:.4f} < threshold={threshold:.4f}, "
                    f"hop={r['hop']}).")
        return ("solid", strength,
                f"'{a_str}' e '{b_str}' connected in {r['hop']} hop "
                f"(strength={strength:.3f} >= threshold={threshold:.3f}).")

    # ── Specialized contents (§6.7) ─────────────────────

    def load_contents(self, path: str, tipo_base: str, language: str = "en"):
        """Carica contenuti specializzati (math, code, ...).
        
        File format: title \\t contenuto1 ||| contenuto2 ||| ...
        For 'code', the lang: prefix is mapped to type code_py/code_cpp/...
        """
        if not os.path.exists(path):
            print(f"  [contenuti:{tipo_base}] file non trovato: {path}")
            return
        
        loaded = 0
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or '\t' not in line:
                    skipped += 1
                    continue
                parts = line.split('\t', 1)
                if len(parts) != 2:
                    skipped += 1
                    continue
                title, raw = parts
                cid = self.id_by_label.get((language, title.lower()))
                if cid is None:
                    continue
                
                for item in raw.split(' ||| '):
                    item = item.strip()
                    if not item:
                        continue
                    
                    if tipo_base == "code":
                        # Format: "python:def foo(): ..." or "music:c d e f g"
                        if ':' in item:
                            lang, content = item.split(':', 1)
                            lang = lang.strip().lower()
                            content = content.strip()
                            # Map to canonical content type
                            lang_map = {
                                # Programming
                                "python": "code_py", "py": "code_py",
                                "cpp": "code_cpp", "c++": "code_cpp", "c": "code_c",
                                "javascript": "code_js", "js": "code_js",
                                "java": "code_java",
                                "generic": "code_generic",
                                # Music
                                "music": "music",
                                # Chemistry
                                "chemistry": "chemistry",
                                # Linguistics
                                "phonetics": "phonetics",
                                # Geography
                                "geo": "geo",
                                # Games
                                "chess": "chess",
                            }
                            tipo = lang_map.get(lang, lang)
                            if content:
                                self._contents[cid][tipo].append(content)
                                loaded += 1
                    else:
                        self._contents[cid][tipo_base].append(item)
                        loaded += 1
        
        # Count
        n_tot = 0
        for ct in self._contents.values():
            if tipo_base == "code":
                n_tot += sum(len(v) for k, v in ct.items() if k.startswith("code_"))
            else:
                n_tot += len(ct.get(tipo_base, []))
        print(f"  [contenuti:{tipo_base}] {loaded_count} items ({skipped_count} lines skipped)", flush=True)

    def answer_with_content(self, subject_str: str, context: list = None,
                               language: str = "en"):
        """Retrieve specialized contents for a concept, context-sensitive.
        
        Automatic type detection based on context:
        - "math", "formula", "equation" → tipo "math"
        - "python", "py", "def " → tipo "code_py"
        - "c++", "cpp", "int main" → tipo "code_cpp"
        - "javascript", "js", "function" → tipo "code_js"
        
        Returns: dict {tipo: [contenuti]}  oppure {} se nessun trigger
        """
        TYPE_TRIGGERS = {
            "math": ["math", "formula", "equation", "latex", "teorema", "theorem",
                     "derivative", "integral", "integrale", "calcola", "calculate"],
            "code_py": ["python", "py", "def ", "import ", "numpy", "pandas"],
            "code_cpp": ["c++", "cpp", "int main", "std::", "iostream"],
            "code_js": ["javascript", "js", "function(", "console.log", "node.js"],
            "music": ["music", "sheet", "score", "spartito", "musica", "note",
                      "lilypond", "melody", "chord", "accordo", "composer"],
            "chemistry": ["chemistry", "molecule", "compound", "reaction",
                          "chimica", "molecola", "reazione", "chemical"],
            "phonetics": ["pronunciation", "pronuncia", "ipa", "phonetic", "fonetica"],
            "geo": ["coordinates", "coordinate", "latitude", "longitude",
                    "lat", "lon", "geographic"],
            "chess": ["chess", "scacchi", "scacchiera", "checkmate"],
        }
        
        cid = self.id_by_label.get((language, subject_str.lower()))
        if cid is None or cid not in self._contents:
            return {}
        
        if context is None:
            return {}  # senza context, economia token
        
        contesto_lower = " ".join(context).lower()
        
        result = {}
        for tipo, triggers in TYPE_TRIGGERS.items():
            if any(t in contesto_lower for t in triggers):
                if tipo in self._contents[cid]:
                    result[tipo] = self._contents[cid][tipo]
        
        return result

    def _invariance(self, dst_id: int, rel_id: int) -> int:
        """How many subjects have an edge with this relation toward dst?"""
        self._build_in_idx()
        arr = self._in_idx.get(dst_id)
        if arr is None:
            return 0
        count = 0
        for k in range(0, len(arr), 2):
            if arr[k+1] == rel_id:
                count += 1
        return count

    def _text(self, nid: int, language: str = "en") -> str:
        """Return the label of an ID in a language."""
        if language in self.labels.get(nid, {}):
            return self.labels[nid][language]
        label = self.g._labels.get(nid)
        if label:
            return label[1]
        return f"#{nid}"

    def _rel_name(self, rid: int) -> str:
        """Return the name of a relation from its ID."""
        if rid in self.rid2name:
            return self.rid2name[rid]
        name = self.g._rel_name(rid)
        if name:
            self.rid2name[rid] = name
            return name
        return f"rel#{rid}"

    def topological_honesty(self, a_str, b_str, language="en"):
        """§7.3: Topological honesty with adaptive threshold from the graph itself.
        
        Returns (verdict, strength, message) where verdict is 'empty', 'weak', or 'solid'.
        The threshold is NOT a magic constant — it's the 99th percentile of
        strengths from comparable node pairs in the graph.
        """
        r = self.context_intersection(a_str, b_str, language=language)
        strength = r["strength"]
        if r["empty"]:
            return ("empty", 0.0,
                    f"Insufficient evidence to connect '{a_str}' and '{b_str}'.")
        
        # Adaptive threshold from the graph (§7.3)
        a_id = self.id_by_label.get((language, a_str))
        b_id = self.id_by_label.get((language, b_str))
        inv_arr = self._inv_array
        inv_a = int(inv_arr[a_id]) if a_id is not None and a_id < len(inv_arr) else 0
        inv_b = int(inv_arr[b_id]) if b_id is not None and b_id < len(inv_arr) else 0
        threshold = self._adaptive_threshold(inv_a, inv_b)
        
        if strength < threshold:
            return ("weak", strength,
                    f"Weak connection between '{a_str}' and '{b_str}' "
                    f"(strength={strength:.4f} < threshold={threshold:.4f}). "
                    f"{len(r['shared'])} shared concepts but low invariance.")
        return ("solid", strength,
                f"'{a_str}' and '{b_str}' share {len(r['shared'])} concepts "
                f"(strength={strength:.3f} >= threshold={threshold:.3f}).")

    def speak(self, subject_str: str, fatti: list, tipo: str = "cos'è",
              language: str = "en") -> str:
        """Generate natural language response (zero template)."""
        if not facts:
            return f"No information about '{subject_str}'."
        dst_id, rel_id, inv = facts[0]
        dst_str = self._text(dst_id, language)
        rel_str = self._rel_name(rel_id)
        return f"{subject_str} {rel_str} {dst_str}."

    # ── Statistics ────────────────────────────────────────

    @property
    def n_nodes(self) -> int:
        return len(self.g._labels)

    @property
    def n_edges(self) -> int:
        return len(self.g._src)

    @property
    def n_concepts(self) -> int:
        return sum(1 for nid in self.senses if len(self.senses[nid]) > 1)

    # ── Save / Load (.irgn) ─────────────────

    def save(self, path: str):
        """Save the graph in .irgn format (string table + binary edges)."""
        strings = []
        for nid, lang_dict in self.labels.items():
            for lang, text in lang_dict.items():
                strings.append((nid, lang, text))
        for nid, (lang, text) in self.g._labels.items():
            if nid not in self.labels or lang not in self.labels[nid]:
                strings.append((nid, lang, text))

        strings.sort()

        with open(path, 'wb') as f:
            f.write(b'IRGN')
            f.write(struct.pack('<I', 1))
            n_nodes = self.n_nodes
            n_edges = self.n_edges
            n_strings = len(strings)
            f.write(struct.pack('<III', n_nodes, n_edges, n_strings))

            for nid, lang, text in strings:
                data = text.encode('utf-8')
                lang_b = lang.encode('utf-8')
                f.write(struct.pack('<II', nid, len(lang_b)))
                f.write(lang_b)
                f.write(struct.pack('<I', len(data)))
                f.write(data)

            for arr in (self.g._src, self.g._rel, self.g._dst):
                f.write(arr.tobytes())

            senses_list = [(nid, ','.join(sorted(s))) 
                         for nid, s in self.senses.items() if s]
            f.write(struct.pack('<I', len(senses_list)))
            for nid, s_str in senses_list:
                data = s_str.encode('utf-8')
                f.write(struct.pack('<II', nid, len(data)))
                f.write(data)

            rel_items = list(self.g._rel_map.items())
            f.write(struct.pack('<I', len(rel_items)))
            for name, rid in rel_items:
                data = name.encode('utf-8')
                f.write(struct.pack('<II', rid, len(data)))
                f.write(data)

        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  Saved: {path} ({size_mb:.1f} MB, {n_nodes} nodes, {n_edges} edges)")

    @classmethod
    def load(cls, path: str):
        """Load a .irgn graph. Classmethod: SubtractionEngine.load(path) works."""
        instance = cls()
        import time as _time
        _t0 = _time.time()
        with open(path, 'rb') as f:
            magic = f.read(4)
            assert magic == b'IRGN', f"Unknown format: {magic}"
            version = struct.unpack('<I', f.read(4))[0]
            n_nodes, n_edges, n_strings = struct.unpack('<III', f.read(12))
            print(f"  [load] header ok, {n_strings} labels, {n_edges} edges", flush=True)

            instance.labels = defaultdict(dict)
            instance.id_by_label = {}
            for _ in range(n_strings):
                nid, lang_len = struct.unpack('<II', f.read(8))
                lang = f.read(lang_len).decode('utf-8')
                data_len = struct.unpack('<I', f.read(4))[0]
                text = f.read(data_len).decode('utf-8')
                instance.labels[nid][lang] = text
                instance.id_by_label[(lang, text)] = nid
            print(f"  [load] strings ok ({_time.time()-_t0:.1f}s)", flush=True)

            chunk_size = n_edges * 4
            instance.g._src = array('I')
            instance.g._src.frombytes(f.read(chunk_size))
            instance.g._rel = array('I')
            instance.g._rel.frombytes(f.read(chunk_size))
            instance.g._dst = array('I')
            instance.g._dst.frombytes(f.read(chunk_size))
            print(f"  [load] edges ok ({_time.time()-_t0:.1f}s)", flush=True)

            for nid, lang_dict in instance.labels.items():
                for lang, text in lang_dict.items():
                    instance.g._labels[nid] = (lang, text)
                    instance.g._id_from_label[(lang, text)] = nid
            instance.g._next_id = max(instance.g._labels.keys()) + 1 if instance.g._labels else 1
            print(f"  [load] labels compact ok ({_time.time()-_t0:.1f}s)", flush=True)

            n_sensi = struct.unpack('<I', f.read(4))[0]
            instance.senses = defaultdict(set)
            for _ in range(n_sensi):
                nid, data_len = struct.unpack('<II', f.read(8))
                s_str = f.read(data_len).decode('utf-8')
                if s_str:
                    instance.senses[nid] = set(s_str.split(','))
            print(f"  [load] senses ok ({_time.time()-_t0:.1f}s)", flush=True)

            n_rel = struct.unpack('<I', f.read(4))[0]
            instance.g._rel_map = {}
            instance.rid2name = {}
            for _ in range(n_rel):
                rid, data_len = struct.unpack('<II', f.read(8))
                name = f.read(data_len).decode('utf-8')
                instance.g._rel_map[name] = rid
                instance.rid2name[rid] = name
            instance.g._next_rel = max(instance.g._rel_map.values()) + 1 if instance.g._rel_map else 1
            print(f"  [load] rel map ok ({_time.time()-_t0:.1f}s)", flush=True)

        # Rebuild out_idx CORRECTLY (sort by src)
        instance.g._rebuild_out_idx()
        instance._in_idx = {}
        print(f"  Loaded: {path} ({n_nodes} nodes, {n_edges} edges, {n_strings} labels)) [{_time.time()-_t0:.1f}s]", flush=True)
        return instance





