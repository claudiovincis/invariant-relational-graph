"""ENGINE COMPACT per TRAINING — edges in array('I'), add O(1), RAM minima"""
import struct
from array import array
import numpy as np
from engine_compact import CompactEngine


class CompactEngineTrain(CompactEngine):
    """CompactEngine con add() per training. Usa array C, non dict Python."""
    
    def __init__(self):
        super().__init__()
        self._dedup = set()  # (src, rel, dst) per O(1) dedup
        self._dirty_nodes = set()  # nodi con out_idx da aggiornare
    
    def add(self, src_etichetta, rel_nome, dst_etichetta):
        """Aggiunge un arco src --[rel]--> dst. O(1) dedup."""
        # Trova/crea ID
        if isinstance(src_etichetta, str):
            src = self._id_from_label.get(("ita", src_etichetta))
            if src is None:
                src = self._next_id
                self._next_id += 1
                self._labels[src] = ("ita", src_etichetta)
                self._id_from_label[("ita", src_etichetta)] = src
        else:
            src = src_etichetta
        
        if isinstance(dst_etichetta, str):
            dst = self._id_from_label.get(("ita", dst_etichetta))
            if dst is None:
                dst = self._next_id
                self._next_id += 1
                self._labels[dst] = ("ita", dst_etichetta)
                self._id_from_label[("ita", dst_etichetta)] = dst
        else:
            dst = dst_etichetta
        
        rel = self._rel_id(rel_nome) if isinstance(rel_nome, str) else rel_nome
        
        # Dedup
        key = (src, rel, dst)
        if key in self._dedup:
            return
        self._dedup.add(key)
        
        # Appendi agli array
        self._src.append(src)
        self._rel.append(rel)
        self._dst.append(dst)
        self._flg.append(0)  # diretto
        
        self._dirty_nodes.add(src)
    
    def _rebuild_out_idx(self):
        """Ricostruisci indice uscente (chiamato prima di save)."""
        self._out_idx.clear()
        if len(self._src) == 0:
            return
        
        # Ordina per (src, rel, dst) — necessario per il formato .grf e per query
        n = len(self._src)
        dt = np.dtype([('s', np.uint32), ('r', np.uint32), ('d', np.uint32)])
        temp = np.empty(n, dtype=dt)
        temp['s'] = np.frombuffer(self._src, dtype=np.uint32)
        temp['r'] = np.frombuffer(self._rel, dtype=np.uint32)
        temp['d'] = np.frombuffer(self._dst, dtype=np.uint32)
        temp.sort(order=['s', 'r', 'd'])
        
        # Ricostruisci array ordinati
        new_src = array('I')
        new_rel = array('I')
        new_dst = array('I')
        new_flg = array('B')
        
        s_arr = temp['s']; r_arr = temp['r']; d_arr = temp['d']
        for i in range(n):
            new_src.append(int(s_arr[i]))
            new_rel.append(int(r_arr[i]))
            new_dst.append(int(d_arr[i]))
            new_flg.append(0)
        
        self._src = new_src
        self._rel = new_rel
        self._dst = new_dst
        self._flg = new_flg
        
        # Costruisci out_idx
        for i in range(n):
            s = self._src[i]
            if s not in self._out_idx:
                self._out_idx[s] = (i, i+1)
            else:
                start, _ = self._out_idx[s]
                self._out_idx[s] = (start, i+1)
        
        self._dirty_nodes.clear()
    
    def save(self, path):
        """save in formato .grf (compatibile con engine_compact.load)."""
        self._rebuild_out_idx()
        
        # Costruisci string table
        strings = []
        str_idx = {}
        def _aggiungi_str(s):
            if s not in str_idx:
                str_idx[s] = len(strings)
                strings.append(s)
            return str_idx[s]
        
        with open(path, "wb") as f:
            f.write(b"GRF\x00")
            f.write(struct.pack("<I", 1))  # version
            
            # Nodi (scritti in ordine di out_idx per compattezza)
            sorted_nodes = sorted(self._out_idx.keys())
            f.write(struct.pack("<I", len(sorted_nodes)))
            
            for nid in sorted_nodes:
                f.write(struct.pack("<I", nid))
                start, end = self._out_idx[nid]
                
                # Raccogli edges per questo nodo: (rel, [(dst, flags)])
                edges = {}
                for i in range(start, end):
                    r = self._rel[i]
                    d = self._dst[i]
                    fl = self._flg[i]
                    if r not in edges:
                        edges[r] = []
                    edges[r].append((d, fl))
                
                f.write(struct.pack("<I", len(edges)))
                for r_id, dests in edges.items():
                    r_nome = self._rel_name(r_id)
                    r_str_id = _aggiungi_str(r_nome)
                    f.write(struct.pack("<II", r_str_id, len(dests)))
                    for d, fl in dests:
                        f.write(struct.pack("<IB", d, fl))
            
            # Etichette (TUTTE, anche nodi senza edges uscenti)
            et_flat = []
            for nid, (lang, testo) in self._labels.items():
                et_flat.append((nid, lang, testo))
            
            f.write(struct.pack("<I", len(et_flat)))
            for nid, lang, testo in et_flat:
                f.write(struct.pack("<I", nid))
                lang_id = _aggiungi_str(lang)
                testo_id = _aggiungi_str(testo)
                f.write(struct.pack("<II", lang_id, testo_id))
            
            # String table
            f.write(struct.pack("<I", len(strings)))
            for s in strings:
                data = s.encode("utf-8") if isinstance(s, str) else s
                f.write(struct.pack("<H", len(data)))
                f.write(data)
    
    @property
    def n_nodes_train(self):
        return len(self._labels)
    
    @property
    def n_edges_train(self):
        return len(self._src)




