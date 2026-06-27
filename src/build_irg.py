"""
Simple IRG builder: reads JSONL triplets and adds them directly to the graph.
No relation abstraction, no reification — just raw S → rel → O edges.
For relation abstraction, use build_irg_reified.py.
"""
import json, sys, time
sys.path.insert(0, '.')
from subtraction_engine import SubtractionEngine


def build_irg(jsonl_path, output_path, max_triplets=None):
    """Read JSONL triplets and build a raw IRG."""
    g = SubtractionEngine()
    
    print(f"Input: {jsonl_path}")
    t0 = time.time()
    count = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_triplets and count >= max_triplets:
                break
            
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            subject = t.get('subject', '').strip()
            relation = t.get('relation', '').strip()
            obj = t.get('object', '').strip()
            
            if not all([subject, relation, obj]):
                continue
            
            g.add(subject, relation, obj, language='en')
            count += 1
            
            if count % 1_000_000 == 0:
                g.compact()
                elapsed = time.time() - t0
                print(f"  {count//1_000_000}M | {elapsed:.0f}s", flush=True)
    
    elapsed = time.time() - t0
    print(f"Mapped: {count:,} triplets in {elapsed:.0f}s")
    
    g.compact()
    print(f"Nodes: {g.n_nodes}, Edges: {g.n_edges}, Concepts: {g.n_concepts}")
    
    print(f"Saving to {output_path}...")
    g.save(output_path)
    size_mb = __import__('os').path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")
    
    return g


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build raw IRG from JSONL triplets')
    parser.add_argument('--input', default='triplets.jsonl', help='Input triplets JSONL')
    parser.add_argument('--output', default='graph.irgn', help='Output IRG file')
    parser.add_argument('--max', type=int, default=None, help='Max triplets (for testing)')
    args = parser.parse_args()
    
    build_irg(args.input, args.output, args.max)


