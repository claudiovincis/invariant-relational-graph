"""
ENGINE COMPACT — edges in array uint32, RAM minima
===================================================
Invece di Python dict→list→int, usa array C uint32.
9.2M edges → ~110 MB RAM (invece di 6 GB).

Query O(1) con lookup binario sull'indice.
"""

import struct
from array import array
import numpy as np


class CompactEngine:
    def __init__(self):
        # edges uscenti: array paralleli uint32 + byte flags
        self._src = array('I')
        self._rel = array('I')
        self._dst = array('I')
        self._flg = array('B')
        
        # edges entranti (ordinati per dst)
        self._src_in = array('I')
        self._rel_in = array('I')
        self._dst_in = array('I')
        
        # Indici O(1)
        self._out_idx = {}    # nid → (start, end) in _src
        self._in_idx = {}     # nid → (start, end) in _src_in
        
        # Etichette
        self._labels = {}  # nid → (lang, testo)
        self._id_from_label = {}  # (lang, testo) → nid
        
        # Relazioni
        self._rel_map = {}
        self._rel_rev = {}
        self._next_rel = 1
        
        self._sistema = 0
        self._next_id = 1
    
    # ═══════════════ CARICAMENTO BINARIO OTTIMIZZATO ═══════════════
    
    def load(self, path, build_incoming=True):
        """load da .grf direttamente in array compatti (RAM minima, due passate).
        Se build_incoming=False, salta l'indice entrante (risparmia ~50% RAM)."""
        import sys
        with open(path, "rb") as f:
            magic = f.read(4)
            assert magic == b"GRF\x00", "Formato non valido"
            version = struct.unpack("<I", f.read(4))[0]
            n_nodi = struct.unpack("<I", f.read(4))[0]
            print(f'  [1/4] {n_nodi:,} nodi, skip...', flush=True)
            
            # === PASSATA 1: salta nodi, leggi etichette + strings ===
            for i in range(n_nodi):
                f.read(4)  # nid
                n_archi = struct.unpack("<I", f.read(4))[0]
                for _ in range(n_archi):
                    f.read(4)  # r_str_id
                    n_dests = struct.unpack("<I", f.read(4))[0]
                    f.read(5 * n_dests)
                if (i+1) % 100000 == 0:
                    print(f'    skip {i+1:,}/{n_nodi:,}', flush=True)
            
            # Etichette raw
            n_etichette = struct.unpack("<I", f.read(4))[0]
            print(f'  [2/4] {n_etichette:,} etichette...', flush=True)
            etichette_raw = []
            for _ in range(n_etichette):
                nid = struct.unpack("<I", f.read(4))[0]
                lang_id = struct.unpack("<I", f.read(4))[0]
                testo_id = struct.unpack("<I", f.read(4))[0]
                etichette_raw.append((nid, lang_id, testo_id))
            
            # String table
            n_str = struct.unpack("<I", f.read(4))[0]
            strings = []
            for _ in range(n_str):
                l = struct.unpack("<H", f.read(2))[0]
                strings.append(f.read(l))
            
            print(f'    risoluzione etichette...', flush=True)
            # Risolvi etichette
            for nid, lang_id, testo_id in etichette_raw:
                lang = strings[lang_id].decode("utf-8")
                testo = strings[testo_id].decode("utf-8")
                self._labels[nid] = (lang, testo)
                self._id_from_label[(lang, testo)] = nid
            del etichette_raw
            
            for nid, (lang, testo) in self._labels.items():
                if lang == "sys" and testo == "sistema":
                    self._sistema = nid
                    break
            
            if n_nodi > 0:
                self._next_id = max(self._labels.keys()) + 1
            
            # === PASSATA 2: load nodi direttamente in array ===
            print(f'  [3/4] Caricamento edges in array...', flush=True)
            f.seek(12)
            
            for i in range(n_nodi):
                nid = struct.unpack("<I", f.read(4))[0]
                n_archi = struct.unpack("<I", f.read(4))[0]
                
                start = len(self._src)
                for _ in range(n_archi):
                    r_str_id = struct.unpack("<I", f.read(4))[0]
                    r_nome = strings[r_str_id].decode("utf-8")
                    r_id = self._rel_id(r_nome)
                    n_dests = struct.unpack("<I", f.read(4))[0]
                    for _ in range(n_dests):
                        d = struct.unpack("<I", f.read(4))[0]
                        flg = struct.unpack("<B", f.read(1))[0]
                        self._src.append(nid)
                        self._rel.append(r_id)
                        self._dst.append(d)
                        self._flg.append(flg)
                end = len(self._src)
                self._out_idx[nid] = (start, end)
                if (i+1) % 50000 == 0:
                    print(f'    {i+1:,}/{n_nodi:,} nodi, {len(self._src):,} edges', flush=True)
            
            # Costruisci indice entrante con numpy (opzionale, save ~50% RAM)
            if build_incoming:
                n = len(self._src)
                print(f'  [4/4] Indice entrante: numpy sort {n:,} edges ({n*12/1e9:.1f} GB)...', flush=True)
                dt_in = np.dtype([('d', np.uint32), ('r', np.uint32), ('s', np.uint32)])
                incoming = np.empty(n, dtype=dt_in)
                incoming['d'] = np.frombuffer(self._dst, dtype=np.uint32)
                incoming['r'] = np.frombuffer(self._rel, dtype=np.uint32)
                incoming['s'] = np.frombuffer(self._src, dtype=np.uint32)
                print(f'    sorting...', flush=True)
                incoming.sort(order=['d', 'r', 's'])
                print(f'    copia in array...', flush=True)
                
                d_arr = incoming['d']; r_arr = incoming['r']; s_arr = incoming['s']
                for i in range(n):
                    self._src_in.append(int(s_arr[i]))
                    self._rel_in.append(int(r_arr[i]))
                    self._dst_in.append(int(d_arr[i]))
                    if (i+1) % 50_000_000 == 0:
                        print(f'    {i+1:,}/{n:,} ({100*(i+1)/n:.0f}%)', flush=True)
                
                print(f'    indice...', flush=True)
                for i in range(n):
                    d = self._dst_in[i]
                    if d not in self._in_idx:
                        self._in_idx[d] = (i, i+1)
                    else:
                        s, e = self._in_idx[d]
                        self._in_idx[d] = (s, i+1)
            else:
                print(f'  [4/4] Indice entrante SALTATO (solo uscente)', flush=True)
                if (i+1) % 50_000_000 == 0:
                    print(f'    indice {i+1:,}/{n:,}', flush=True)
        
        return self
    
    # ═══════════════ ID / REL ═══════════════
    
    def _rel_id(self, nome):
        if nome not in self._rel_map:
            self._rel_map[nome] = self._next_rel
            self._rel_rev[self._next_rel] = nome
            self._next_rel += 1
        return self._rel_map[nome]
    
    def _rel_name(self, rid):
        return self._rel_rev.get(rid, f"rel_{rid}")
    
    def etichetta(self, nid):
        if nid in self._labels:
            return self._labels[nid][1]
        return f"#{nid}"
    
    def so(self, etichetta):
        return ("ita", etichetta) in self._id_from_label
    
    def _id(self, etichetta):
        return self._id_from_label.get(("ita", etichetta))
    
    # ═══════════════ QUERY ═══════════════
    
    def archi_uscenti(self, nid):
        """Restituisce [(rel_nome, dst_nome)] per il nodo."""
        if nid not in self._out_idx:
            return []
        start, end = self._out_idx[nid]
        risultato = []
        for i in range(start, end):
            rel = self._rel_name(self._rel[i])
            dst = self.etichetta(self._dst[i])
            risultato.append((rel, dst))
        return risultato
    
    def cerca(self, soggetto, relazione=None):
        """Cerca fatti: soggetto → [relazione] → ?"""
        nid = self._id(soggetto)
        if nid is None or nid not in self._out_idx:
            return []
        start, end = self._out_idx[nid]
        risultati = []
        for i in range(start, end):
            r_nome = self._rel_name(self._rel[i])
            if relazione is None or r_nome == relazione:
                risultati.append((r_nome, self.etichetta(self._dst[i])))
        return risultati
    
    def archi_entranti(self, nid):
        """Restituisce [(src_nome, rel_nome)] che puntano al nodo."""
        if nid not in self._in_idx:
            return []
        start, end = self._in_idx[nid]
        risultato = []
        for i in range(start, end):
            rel = self._rel_name(self._rel_in[i])
            src = self.etichetta(self._src_in[i])
            risultato.append((src, rel))
        return risultato
    
    # ═══════════════ STATS ═══════════════
    
    def __len__(self):
        return len(self._labels)
    
    @property
    def n_archi(self):
        return len(self._src)
    
    @property
    def n_nodi(self):
        return len(self._labels)


# ═══════════════ COMPATIBILITÀ CON PARLA ═══════════════

def crea_adattatore(mc):
    """
    Wrapper per far funzionare parla.py con CompactEngine.
    Simula l'interfaccia di MotoreBin.
    """
    class Adattatore:
        def __init__(self):
            self.g = mc
        
        @property
        def nodi(self):
            return {nid: {} for nid in mc._labels}
        
        @property
        def _sistema(self):
            return mc._sistema
        
        def etichetta(self, nid, lang="ita"):
            return mc.etichetta(nid)
        
        def _rel_name(self, rid):
            return mc._rel_name(rid)
        
        def so(self, etichetta, lang="ita"):
            return mc.so(etichetta)
        
        def _id(self, etichetta, lang="ita"):
            return mc._id(etichetta)
        
        def edges(self, nid):
            uscenti = {}
            entranti = {}
            for rel, dst in mc.archi_uscenti(nid):
                uscenti.setdefault(mc._rel_map.get(rel, 0), []).append(mc._id(dst))
            for src, rel in mc.archi_entranti(nid):
                entranti.setdefault(mc._rel_map.get(rel, 0), []).append(mc._id(src))
            return uscenti, entranti
    
    return Adattatore()




