# distutils: language = c++
# cython: language_level=3
include "busclique_h.pxi"

import homebase, os, pathlib, fasteners, threading
from pickle import dump, load
import networkx as nx, dwave_networkx as dnx
cdef int __cache_version = 1
cdef int __lru_size = 100
cdef dict __global_locks = {'clique': threading.Lock(),
                            'biclique': threading.Lock()}

def _num_nodes(nn, offset = 0):
    """
    Internal-use function to normalize inputs.

    Inputs:
        nn: int or iterable

    Outputs:
        (n, nodes)
            n: nn if n is an int, otherwise len(nodes)
            nodes: tuple(offset, offset + nn) if nn is an iterable, otherwise
            tuple(range(offset, offset + n))
    """
    try:
        nodes = tuple(range(offset, offset + nn))
        num = nn
    except TypeError:
        nodes = tuple(nn)
        num = len(nodes)
    return num, nodes

def find_clique_embedding(g, nodes, use_cache = True):
    """
    Finds a clique embedding in the graph g using a polynomial-time algorithm.
    
    Inputs:
        g: either a dwave_networkx.chimera_graph or dwave_networkx.pegasus_graph
        nodes: a number (indicating the size of the desired clique) or an 
            iterable (specifying the node labels of the desired clique)
        use_cache: bool, default True -- whether or not to compute / restore a
            cache of clique embeddings for g.  Note that this function only uses
            the filesystem cache, and does not maintain the cache in memory.  If
            many (or even several) embeddings are desired in a single session,
            it is recommended to use `busgraph_cache`

    Returns:
        emb: dict mapping node labels (either nodes, or range(nodes)) to chains
            of a clique embedding
    """
    try:
        graphdata = g.graph
        family = graphdata['family']
        busgraph = {'pegasus': _pegasus_busgraph,
                    'chimera': _chimera_busgraph}[family]
    except (AttributeError, KeyError):
        raise ValueError("g must be either a dwave_networkx.chimera_graph or"
                         "a dwave_networkx.pegasus_graph")
    if use_cache:
        return busgraph_cache(g).find_clique_embedding(nodes)
    else:
        return busgraph(g).find_clique_embedding(nodes)

class busgraph_cache:
    def __init__(self, g):
        """
        A cache class for chimera / pegasus graphs, and their associated cliques
        and bicliques.

        Input:
            g: a dwave_networkx.pegasus_graph or dwave_networkx.chimera_graph
        """
        self._family = g.graph['family']
        if(self._family == 'chimera'):
            self._graph = _chimera_busgraph(g)
        elif(self._family == 'pegasus'):
            self._graph = _pegasus_busgraph(g)
        else:
            raise ValueError(("input graph must either be a "
                              "dwave_networkx.pegasus_graph or a "
                              "dwave_networkx.chimera_graph"))
        self._cliques = None
        self._bicliques = None

    def _ensure_clique_cache(self):
        """
        Fetch / compute the clique cache, if it's not already in memory
        """
        if self._cliques is None:
            self._cliques = self._fetch_cache('clique', self._graph.cliques)


    def _ensure_biclique_cache(self):
        """
        Fetch / compute the clique cache, if it's not already in memory
        """
        if self._bicliques is None:
            self._bicliques = self._fetch_cache('biclique', self._graph.bicliques)

    def _fetch_cache(self, dirname, compute):
        """
        This is an ad-hoc implementation of a file-cache using a LRU strategy.
        It's intended to be platform independent, thread- and multiprocess-safe,
        and reasonably performant -- I couldn't find a ready-made solution that
        satisfies those requirements, so I had to roll my own.  TODO: keep 
        looking for something cleaner.

        We use `homebase` to locate a viable directory to store our caches.
        We create two subdirectories (for now): `cliques` and `bicliques`, and
        store respective caches in each.

        We derive a unique identifier from `self._graph`, and hash that
        identifier to produce a `shortcode`.  The cache associated with
        `self.graph` has the filename `str(shortcode)` -- this is preferable to
        using a single file to store the entire cache.

        Inside a cache directory, we have two special files and perhaps many
        cache files.  The `.lock` file is used by `fasteners.InterProcessLock`
        to provide process-level locking.  The `.lru` file is a list of up to
        `__lru_size` different cache filenames, sorted descending in the
        recentness of the last access of a given filename.  This enables us to
        automatically clean up the cache before it gets too large.
        """
        rootdir = homebase.user_data_dir('busclique', 'dwave', __cache_version)
        basedir = os.path.join(rootdir, dirname)
        pathlib.Path(basedir).mkdir(parents=True, exist_ok=True)
        lockfile = os.path.join(basedir, ".lock")
        lrufile = os.path.join(basedir, ".lru")
        identifier = self._graph.identifier
        shortcode = str(hash(identifier))
        #this solves thread-safety?
        with __global_locks[dirname]:
            #this solves inter-process safety?
            with fasteners.InterProcessLock(lockfile):
                cachefile = os.path.join(basedir, str(shortcode))
                if os.path.exists(cachefile):
                    with open(cachefile, 'rb') as filecache:
                        currcache = load(filecache)
                else:
                    currcache = {}
                cache = currcache.get(identifier)
                if cache is None:
                    cache = compute()
                    currcache[identifier] = cache
                    with open(cachefile, 'wb') as filecache:
                        dump(currcache, filecache)

                #now, update the LRU cache -- if this was being done in-memory,
                #there are more efficient algorithms... but since we're doing 
                #linear-time read&write, might as well do it the lazy way.
                try:
                    with open(lrufile, 'rb') as lru:
                        LRU = load(lru)
                except FileNotFoundError:
                    LRU = []
                newLRU = [shortcode]
                for x in LRU:
                    if x != shortcode:
                        newLRU.append(x)
                while len(newLRU) > __lru_size:
                    oldkey = newLRU.pop()
                    os.remove(os.path.join(basedir, oldkey))
                with open(lrufile, 'wb') as lru:
                    dump(newLRU, lru)
        return cache

    def largest_clique(self):
        """
        Returns the largest-found clique in the clique cache.  Keys of the
        embedding dict are from range(len(emb))
        """
        self._ensure_clique_cache()
        embs = self._cliques['raw']
        keys = self._cliques['size']
        return self._graph.relabel(dict(enumerate(embs[keys[max(keys)]])))

    def largest_clique_by_chainlength(self, chainlength):
        """
        Returns the largest-found clique in the clique cache, with a specified
        maximum chainlength.  Keys of the embedding dict are from range(len(emb)).
        """
        self._ensure_clique_cache()
        embs = self._cliques['raw']

        if 0 <= chainlength < len(embs):
            return self._graph.relabel(dict(enumerate(embs[chainlength])))
        else:
            return {}

    def find_clique_embedding(self, nn):
        """
        Returns a clique embedding, minimizing the maximum chainlength given its
        size.
        
        Inputs:
            nn: a number (indicating the size of the desired clique) or an 
                iterable (specifying the node labels of the desired clique)

        Returns:
            emb: dict mapping node labels (either nn, or range(nn)) to chains
                of a clique embedding
        """
        num, nodes = _num_nodes(nn)
        self._ensure_clique_cache()
        keys = self._cliques['size']
        key = keys.get(num)
        if key is None:
            return {}
        emb = dict(zip(nodes, self._cliques['raw'][key]))
        return self._graph.relabel(emb)

    def largest_balanced_biclique(self):
        """
        Returns the largest-size biclique where both sides have equal size.
        Nodes of the embedding dict are from range(len(emb)), where the nodes
        range(len(emb)//2) are completely connected to the nodes 
        range(len(emb)//2, len(emb)).        
        """
        self._ensure_biclique_cache()
        biggest = self._bicliques['max_side']
        embs = self._bicliques['raw']
        s0, s1 = key = max(embs, key=lambda x: min(x))
        raw_emb = embs[key]
        raw_emb0 = sorted(raw_emb[:s0], key=len)
        raw_emb1 = sorted(raw_emb[s0:s0+s1], key=len)
        s_min = min(s0, s1)
        emb0 = raw_emb0[:s_min]
        emb1 = raw_emb1[:s_min]
        return self._graph.relabel(dict(enumerate(emb0 + emb1)))

    def find_biclique_embedding(self, nn, mm):
        """
        Returns a biclique embedding, minimizing the maximum chainlength given 
        its size.
        
        Inputs:
            nn: int (indicating the size of one side of the desired biclique) or
                an iterable (specifying the node labels of one side the desired
                buclique)
            mm: int or iterable, as above.

        In the case that nn is a number, the first side will have nodes labeled
        from range(nn).  In the case that mm is a number, the second side will
        have nodes labeled from range(n, n + mm); where n is either nn or
        len(nn).        

        Returns:
            emb: dict mapping node labels (described above) to chains of a
                biclique embedding
        """
        self._ensure_biclique_cache()
        biggest = self._bicliques['max_side']
        by_size = self._bicliques['size']
        raw = self._bicliques['raw']
        n, N = _num_nodes(nn)
        m, M = _num_nodes(mm, offset = n)
        if(m < n):
            n, N, m, M = m, M, n, N
        if m == 0:
            emb = self._graph.independent_set(n)
            if emb is None:
                return {}
            return dict(zip(N, emb))
        nmax = biggest[None]
        for i in range(n, nmax + 1):
            mmax = biggest.get(i, 0)
            if mmax < m:
                continue
            row = by_size.get(i, {0:None})
            for j in range(m, mmax + 1):
                key = row.get(j)
                if key is None:
                    continue
                emb = raw[key]
                s0, s1 = key
                emb0 = emb[:s0]
                emb1 = emb[s0:]
                if s0 <= n and s1 <= m:
                    emb = dict(zip(N + M, emb0[:n] + emb1))
                else:
                    emb = dict(zip(N + M, emb1[:n] + emb0))
                return self._graph.relabel(emb)
        return {}

cdef dict _make_clique_cache(vector[embedding_t] &embs):
    """
    Process the raw clique cache produced by the c++ code.  Store both the raw
    list of embeddings (a list `raw` with `raw[i]` being the maximum-size 
    embedding with maximum chainlength `i`), and a size-indexed dictionary
    `by_size` where `embs[by_size[j]]` is a clique embedding of size `j` whose
    maximum chainlength is minimized.
    """
    cdef embedding_t emb
    cdef size_t maxsize = 0
    cdef size_t maxlength = 0
    cdef size_t i, j
    cdef dict by_size = {}
    cdef list raw = []
    for length, emb in enumerate(embs):
        raw.append(tuple(map(tuple, emb)))
        maxsize = max(emb.size(), maxsize)
        for i in range(maxsize, -1, -1):
            if by_size.setdefault(i, length) != length:
                break
    return {'raw': raw, 'size': by_size}

cdef _keep_biclique_key(tuple key0, tuple key1, dict chainlength):
    """
    Helper function for `_make_biclique_cache`.  Each "key" is a pair of ints
    (n, m) corresponding to a biclique K_{n, m}.  `chainlength` is a dictionary
    mapping these keys to the lengths of the longest, and shortest, chains in
    the known embedding of K_{n, m}.
 
    We return True if the embedding associated with key0 is "equal or better"
    than that of key1.  We say that one embedding is better than the other by
    first prioritizing the smallest maximum chainlength, and then prioritizing
    the largest minimum chainlength (e.g. embeddings with balanced chainlengths
    are preferred).
    """
    if key0 == key1:
        return True
    long0, short0 = chainlength[key0]
    long1, short1 = chainlength[key1]
    if long0 < long1 or (long0 == long1 and short0 <= short1):
        return True
    else:
        return False

cdef dict _make_biclique_cache(vector[pair[pair[size_t, size_t], embedding_t]] &embs):
    """
    Process the raw biclique cache produced by the c++ code.  Store the raw dict
    of embeddings (a dict `raw` with `raw[s0, s1]` being the minimum chainlength
    embedding of K_{s0, s1} where the order of s0, s1 matters; a 2-dimensional
    dict `size` where `size[s0][s1]` (given s0 >= s1) is a key into `raw` where         
    `raw[size[s0][s1]]` is the minimum chainlength embedding of K_{s0, s1}; and
    a dictionary `max_side` where `max_side[None]` gives the largest value of 
    `x` such that K_{x, y} exists for any `y > 0`; and otherwise `max_side[x]`
    gives the largest value y for which K_{x, y} exists.
    """
    cdef embedding_t emb
    cdef dict raw = {(s0, s1): emb for (s0, s1), emb in embs if emb.size()}
    cdef dict chainlength = {key: (max(map(len, emb)), min(map(len, emb)))
                             for key, emb in embs if emb.size()}

    cdef dict by_size = {}
    cdef dict by_length = {}
    for key, val in raw.items():
        s0, s1 = key
        emb0 = tuple(val[:s0])
        emb1 = tuple(val[s0:])
        if(s0 < s1):
            s1, s0 = s0, s1
        realkey = by_size.setdefault(s0, {}).setdefault(s1, key)
        if not _keep_biclique_key(realkey, key, chainlength):
            by_size[s0][s1] = key

    max_side = {None: max(by_size)}
    for key, val in by_size.items():
        max_side[key] = max(val)
    return {'raw': raw, 'size': by_size, 'max_side': max_side}

def _trivial_relabeler(emb):
    """This doesn't relabel anything"""
    return emb

def _make_relabeler(f):
    """
    This returns an embedding-relabeling function, which applies `f` to each
    chain in the embedding.
    """
    def _relabeler(emb):
        return {v: list(f(chain)) for v, chain in emb.items()}
    return _relabeler

cdef class _pegasus_busgraph:
    cdef topo_cache[pegasus_spec] *topo
    cdef nodes_t nodes
    cdef embedding_t emb_1
    cdef readonly object relabel
    cdef readonly object identifier
    def __cinit__(self, g):
        """
        This is a class which manages a single pegasus graph, and dispatches various
        structure-aware c++ embedding functions on it.
        """
        rows = g.graph['rows']
        voff = [o//2 for o in g.graph['vertical_offsets'][::2]]
        hoff = [o//2 for o in g.graph['horizontal_offsets'][::2]]
        cdef pegasus_spec *peg = new pegasus_spec(rows, voff, hoff)
        coordinates = dnx.pegasus_coordinates(rows)
        cdef edges_t edges
        if g.graph['labels'] == 'int':
            self.nodes = g.nodes()
            edges = g.edges()
            self.relabel = _trivial_relabeler
        elif g.graph['labels'] == 'coordinate':
            self.nodes = coordinates.iter_pegasus_to_linear(g.nodes())
            edges = coordinates.iter_pegasus_to_linear_pairs(g.edges())
            self.relabel = _make_relabeler(coordinates.iter_linear_to_pegasus)
        elif g.graph['labels'] == 'nice':
            self.nodes = coordinates.iter_nice_to_linear(g.nodes())
            edges = coordinates.iter_nice_to_linear_pairs(g.edges())
            self.relabel = _make_relabeler(coordinates.iter_linear_to_nice)
        else:
            raise ValueError("unrecognized graph labeling")

        self.topo = new topo_cache[pegasus_spec](peg[0], self.nodes, edges)
        short_clique(peg[0], self.nodes, edges, self.emb_1)

        #TODO replace this garbage with data from topo
        self.identifier = (rows, tuple(voff), tuple(hoff), tuple(sorted(self.nodes)),
                           tuple(sorted(map(tuple, map(sorted, edges)))))
        del peg

    def __dealloc__(self):
        del self.topo

    def bicliques(self):
        """
        Returns a biclique cache -- see _make_biclique_cache for more info.
        """
        cdef vector[pair[pair[size_t, size_t], embedding_t]] embs
        best_bicliques[pegasus_spec](self.topo[0], embs)
        return _make_biclique_cache(embs)

    def cliques(self):
        """
        Returns a clique cache -- see _make_clique_cache for more info.
        """
        cdef vector[embedding_t] embs
        best_cliques[pegasus_spec](self.topo[0], embs, self.emb_1)
        return _make_clique_cache(embs)

    def independent_set(self, size):
        """
        This is extremely silly, but what else should we do when a user requests
        a K_{0, n}?
        """
        cdef int i
        if size < self.nodes.size():
            emb = [self.nodes[i] for i in range(size)]
            return emb

    def find_clique_embedding(self, nn):
        """
        Perfoms a "one-shot" embedding procedure to find K_n where n is either
        int(nn) or len(tuple(nn)), without generating a cache.
        """
        num, nodes = _num_nodes(nn)
        cdef embedding_t emb
        if num <= self._emb_1.size():
            emb = self._emb_1
        elif not find_clique(self.topo[0], num, emb):
            return {}
        return self.relabel(dict(zip(nodes, emb)))

cdef class _chimera_busgraph:
    """
    This is a class which manages a single chimera graph, and dispatches various
    structure-aware c++ embedding functions on it.
    """
    cdef topo_cache[chimera_spec] *topo
    cdef embedding_t emb_1
    cdef nodes_t nodes
    cdef readonly object relabel
    cdef readonly object identifier
    def __cinit__(self, g):
        rows = g.graph['rows']
        cols = g.graph['columns']
        tile = g.graph['tile']
        if tile > 8:
            raise NotImplementedError(("this clique embedder only supports chimera "
                                       "graphs with a tile size of 8 or less"))
        cdef chimera_spec *chim = new chimera_spec(rows, cols, tile)
        cdef edges_t edges
        coordinates = dnx.chimera_coordinates(rows, cols, tile)
        if g.graph['labels'] == 'int':
            self.nodes = g.nodes()
            edges = g.edges()
            self.relabel = _trivial_relabeler
        elif g.graph['labels'] == 'coordinate':
            self.nodes = coordinates.iter_chimera_to_linear(g.nodes())
            edges = coordinates.iter_chimera_to_linear_pairs(g.edges())
            self.relabel = _make_relabeler(coordinates.iter_linear_to_chimera)
        else:
            raise ValueError("unrecognized graph labeling")

        self.topo = new topo_cache[chimera_spec](chim[0], self.nodes, edges)
        short_clique(chim[0], self.nodes, edges, self.emb_1)

        #TODO replace this garbage with data from topo
        self.identifier = (rows, cols, tile, tuple(sorted(self.nodes)),
                           tuple(sorted(tuple(sorted(e)) for e in edges)))

        del chim


    def __dealloc__(self):
        del self.topo

    def bicliques(self):
        """
        Returns a biclique cache -- see _make_biclique_cache for more info.
        """
        cdef vector[pair[pair[size_t, size_t], embedding_t]] embs
        best_bicliques(self.topo[0], embs)
        return _make_biclique_cache(embs)

    def cliques(self):
        """
        Returns a clique cache -- see _make_clique_cache for more info.
        """
        cdef vector[embedding_t] embs
        best_cliques(self.topo[0], embs, self.emb_1)
        return _make_clique_cache(embs)

    def independent_set(self, size):
        """
        This is extremely silly, but what else should we do when a user requests
        a K_{0, n}?
        """
        cdef int i
        if size < self.nodes.size():
            emb = [self.nodes[i] for i in range(size)]
            return emb

    def find_clique_embedding(self, nn):
        """
        Perfoms a "one-shot" embedding procedure to find K_n where n is either
        int(nn) or len(tuple(nn)), without generating a cache.
        """
        num, nodes = _num_nodes(nn)
        cdef embedding_t emb
        if num <= self._emb_1.size():
            emb = self._emb_1
        elif not find_clique(self.topo[0], num, emb):
            return {}
        return self.relabel(dict(zip(nodes, emb)))
