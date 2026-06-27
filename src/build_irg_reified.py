"""
Reification Engine: triplets → abstract concepts → reified IRG.
Each triplet (S, rel, O) becomes 1 edge: S → concept → O.
The 15 abstract concepts are registered as regular nodes.
Relations are mapped through ABSTRACT_CONCEPTS dictionary.
"""
import json, sys, time
from collections import defaultdict
sys.path.insert(0, '.')
from subtraction_engine import SubtractionEngine

# ═══════════════════════════════════════════════════════════
# 15 Abstract Concepts — the semantic backbone
# Maps surface relations (verbs) → abstract concept categories
# ═══════════════════════════════════════════════════════════

ABSTRACT_CONCEPTS = {
    # ── LOCATION ──
    'was born in': 'location', 'died in': 'location', 'is in country': 'location',
    'is in location': 'location', 'is in city': 'location', 'is in state': 'location',
    'is in region': 'location', 'happened in': 'location', 'lived in': 'location',
    'was buried in': 'location', 'is headquartered in': 'location',
    'is spoken in': 'location', 'is awarded in': 'location',
    'is from country': 'location', 'lives in': 'location',
    
    # ── POSSESSION / ACQUISITION ──
    'received award': 'possession', 'won': 'possession', 'was awarded': 'possession',
    'received': 'possession', 'awarded': 'possession', 'received honor': 'possession',
    'held title': 'possession', 'held position': 'possession', 'held citizenship of': 'possession',
    'had nationality': 'possession', 'uses currency': 'possession',
    'has GDP': 'possession', 'has revenue': 'possession', 'has population': 'possession',
    'has area km2': 'possession', 'has elevation m': 'possession',
    'had budget': 'possession', 'had box office': 'possession',
    'has employees': 'possession', 'has students': 'possession',
    'has ISO code': 'possession', 'has postal code': 'possession',
    
    # ── CREATION ──
    'wrote': 'creation', 'created': 'creation', 'founded': 'creation', 'established': 'creation',
    'co-founded': 'creation', 'developed': 'creation', 'introduced': 'creation',
    'published': 'creation', 'produces': 'creation', 'was founded': 'creation',
    'was established': 'creation', 'was founded by': 'creation',
    'was written by': 'creation', 'was produced by': 'creation',
    'was directed by': 'creation', 'was released by label': 'creation',
    'was published by': 'creation', 'was published on': 'creation',
    'was released on': 'creation', 'was by artist': 'creation',
    'was produced by studio': 'creation', 'created notable work': 'creation',
    'composed': 'creation', 'designed': 'creation',
    
    # ── TEMPORAL ──
    'was born on': 'temporal', 'died on': 'temporal', 'happened on': 'temporal',
    'was published on': 'temporal', 'was released on': 'temporal',
    'was crowned on': 'temporal', 'was buried on': 'temporal',
    'reigned': 'temporal', 'was established on': 'temporal',
    'was established in year': 'temporal', 'was launched on': 'temporal',
    
    # ── MEMBERSHIP ──
    'member of': 'membership', 'was member of': 'membership', 'fellow of': 'membership',
    'elected to': 'membership', 'elected a': 'membership', 'joined': 'membership',
    'was part of movement': 'membership', 'belonged to school': 'membership',
    'belonged to house': 'membership', 'was member of party': 'membership',
    'signed with label': 'membership', 'performed with': 'membership',
    'is affiliated with': 'membership', 'served on board of': 'membership',
    'is subsidiary of': 'membership', 'owns subsidiary': 'membership',
    'is part of': 'membership', 'has member': 'membership',
    'had participants': 'membership', 'practiced religion': 'membership',
    
    # ── IDENTITY ──
    'was a': 'identity', 'was an': 'identity', 'was the': 'identity',
    'became a': 'identity', 'became the': 'identity', 'known as': 'identity',
    'is known for': 'identity', 'known for': 'identity', 'is type of': 'identity',
    'was born as': 'identity', 'is in genre': 'identity', 'is in language': 'identity',
    'is god of': 'identity', 'is type of mission': 'identity',
    'is equivalent to': 'identity', 'has name': 'identity', 'has abbreviation': 'identity',
    'has genitive': 'identity', 'is a': 'identity',
    
    # ── EDUCATION ──
    'educated at': 'education', 'studied at': 'education', 'graduated from': 'education',
    'enrolled in': 'education', 'attended': 'education', 'taught at': 'education',
    'had doctoral advisor': 'education', 'had academic advisor': 'education',
    'had doctoral student': 'education', 'had notable student': 'education',
    
    # ── PROFESSION ──
    'worked as': 'profession', 'worked at': 'profession', 'worked in field': 'profession',
    'worked on': 'profession', 'employed by': 'profession',
    'was professor at': 'profession', 'was professor of': 'profession',
    'was led by': 'profession', 'is led by': 'profession',
    'has president': 'profession', 'has key person': 'profession',
    'has mayor': 'profession', 'traded as': 'profession',
    
    # ── INFLUENCE ──
    'influenced': 'influence', 'was influenced by': 'influence',
    'was influenced': 'influence', 'influenced by': 'influence',
    
    # ── CAUSATION ──
    'discovered': 'causation', 'predicted': 'causation', 'demonstrated': 'causation',
    'proved': 'causation', 'solved': 'causation', 'derived': 'causation',
    'proposed': 'causation', 'formulated': 'causation', 'described': 'causation',
    'explained': 'causation', 'showed': 'causation', 'argued': 'causation',
    'led to': 'causation', 'based on': 'causation', 'applied to': 'causation',
    'had result': 'causation', 'had outcome': 'causation',
    
    # ── RELATION ── (personal)
    'was married to': 'relation', 'married': 'relation', 'had child': 'relation',
    'was related to': 'relation', 'had parent': 'relation',
    'divorced': 'relation', 'had sibling': 'relation',
    
    # ── MOTION ──
    'moved to': 'motion', 'visited': 'motion', 'traveled to': 'motion',
    'left': 'motion', 'ascended to': 'motion',
    
    # ── COMBAT ──
    'was fought by': 'combat', 'was fought against': 'combat',
    'was commanded by': 'combat', 'served under': 'combat',
    
    # ── SUCCESSION ──
    'had predecessor': 'succession', 'had successor': 'succession',
    'elected': 'succession', 'appointed': 'succession',
    'was crowned': 'succession', 'reigned as': 'succession',
    
    # ── COMMUNICATION ──
    'collaborated with': 'communication', 'corresponded with': 'communication',
    'named': 'communication', 'has website': 'communication',
    
    # ── other / catch-all ──
    'is presented by': 'possession', 'is awarded for': 'identity',
    'is held by': 'possession', 'was operated by': 'profession',
    'used spacecraft': 'possession', 'was launched from': 'location',
    'landed on': 'location', 'has brightest star': 'possession',
    'has nearest star': 'possession', 'plays instrument': 'profession',
    'has symbol': 'possession', 'rides': 'possession',
    'has purpose': 'identity', 'has board': 'possession',
    'has campus type': 'possession', 'has capital': 'possession',
    'has largest city': 'possession', 'has official language': 'possession',
    'is written in script': 'possession', 'descends from': 'causation',
    'is language family': 'identity', 'has proto-language': 'possession',
    'stars': 'profession', 'has music by': 'creation',
    'belongs to language family': 'membership', 'has speakers': 'possession',
    'honorary member of': 'membership',
    # ── ones we missed ──
    'died of': 'causation', 'authored': 'creation', 'co-authored': 'creation',
    'played genre': 'profession', 'played instrument': 'profession',
    'developed idea': 'creation', 'works in industry': 'profession',
    'studied under': 'education', 'was interested in': 'identity',
    'wrote in genre': 'creation', 'has gdp': 'possession',
    'was honored with': 'possession', 'has iso code': 'possession',
}


def build_reified_irg(jsonl_path, output_path, max_triplets=None):
    """Build IRG with relation abstraction: surface verb → concept → object.
    
    Each triplet becomes: subject --[concept]--> object
    Plus reverse index edges: concept --[instance]--> subject/object
    """
    m = SubtractionEngine()
    
    ALL_CONCEPT_VALUES = set(ABSTRACT_CONCEPTS.values())
    
    print(f"Mapping: {jsonl_path}")
    print(f"Concepts: {len(ALL_CONCEPT_VALUES)} | Verbs: {len(ABSTRACT_CONCEPTS)}")
    print(f"Model: subject → concept → object (1 edge/triplet)")
    print()
    
    t0 = time.time()
    count = 0
    unmapped = defaultdict(int)
    
    # Register all 15 concepts as NODES (not just relations)
    for c in ALL_CONCEPT_VALUES:
        m._get_or_create(c, 'en')
    print(f"  Registered {len(ALL_CONCEPT_VALUES)} concept nodes")
    print()
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_triplets and count >= max_triplets:
                break
            
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            subject = t.get('subject', '').lower().strip()
            relation = t.get('relation', '').lower().strip()
            obj = t.get('object', '').lower().strip()
            
            if not all([subject, relation, obj]):
                continue
            
            # First: check if relation is already an abstract concept (spaCy path)
            if relation in ALL_CONCEPT_VALUES:
                concept = relation
            else:
                # Map surface relation → abstract concept (infobox path)
                concept = ABSTRACT_CONCEPTS.get(relation, 'other')
                if concept == 'other':
                    unmapped[relation] += 1
            
            m.add(subject, concept, obj, language='en')
            # Make concept queryable: concept -> [all entities using it]
            m.add(concept, 'instance', subject, language='en')
            m.add(concept, 'instance', obj, language='en')
            
            count += 1
            if count % 2_000_000 == 0:
                elapsed = time.time() - t0
                print(f"  {count//1_000_000}M triplets | {elapsed:.0f}s", flush=True)
                m.compact()
    
    elapsed = time.time() - t0
    print(f"\nMapped: {count:,} triplets in {elapsed:.0f}s")
    
    if unmapped:
        print(f"\nUnmapped relations ({len(unmapped)}):")
        for rel, c in sorted(unmapped.items(), key=lambda x: -x[1])[:10]:
            print(f"  {rel:35s} {c:>8,d}")
    
    print(f"\nSaving...")
    m.save(output_path)
    
    size_mb = __import__('os').path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")
    
    return m


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build reified IRG with abstract concept mapping')
    parser.add_argument('--input', default='wiki_en_triplets.jsonl', help='Input triplets JSONL')
    parser.add_argument('--output', default='wiki_en_v4.irgn', help='Output IRG file')
    parser.add_argument('--max', type=int, default=None, help='Max triplets (for testing)')
    args = parser.parse_args()
    
    m = build_reified_irg(args.input, args.output, args.max)
    
    # Quick test
    print("\n=== Quick test ===")
    concepts_to_test = ['location', 'possession', 'creation', 'membership', 'temporal']
    for c in concepts_to_test:
        r = m.ask(c, language='en')
        found = r.get('concepts', {}).get(c.lower(), {})
        defs = found.get('top_definitions', [])
        print(f"  {c}: {len(defs)} definitions")
