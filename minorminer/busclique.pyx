# Copyright 2020 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# distutils: language = c++
# cython: language_level=3
include "busclique_h.pxi"

import homebase, os, pathlib, fasteners, threading, random
from pickle import dump, load
import networkx as nx, dwave_networkx as dnx

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.bytes cimport PyBytes_FromStringAndSize

#increment this version any time there is a change made to the cache format,
#when yield-improving changes are made to clique algorithms, or when bugs are
#fixed in the same.
cdef int __cache_version = 6

cdef int __lru_size = 100
cdef dict __global_locks = {'clique': threading.Lock(),
                            'biclique': threading.Lock()}

def _num_nodes(nn, offset = 0):
    """Internal-use function to normalize inputs.

    Args:
        nn: int or iterable

    Returns:
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

_no_seed = object()
def find_clique_embedding(nodes, g, seed = _no_seed, use_cache = True):
    """Finds a clique embedding in the graph ``g`` using a polynomial-time 
    algorithm.

    Args:
        nodes (int/iterable): 
            A number (indicating the size of the desired clique) or an 
            iterable (specifying the node labels of the desired clique).
        
        g (NetworkX Graph): 
            The target graph that is either a :func:`dwave_networkx.chimera_graph` 
            or :func:`dwave_networkx.pegasus_graph`.
        
        use_cache (bool, optional, default=True):
            Whether or not to compute/restore a cache of clique embeddings for 
            ``g``. Note that this function only uses the filesystem cache, and 
            does not maintain the cache in memory. If many (or even several) 
            embeddings are desired in a single session, it is recommended to use 
            :class:`.busgraph_cache`.
            
        seed (int, optional):
            A seed for an internal random number generator.  If ``use_cache`` is
            True, then the seed defaults to an internally-defined value which
            is consistent between runs.  Otherwise, a seed is generated from the
            current python random state.

    Returns:
        dict: An embedding of node labels (either nodes, or range(nodes)) mapped 
        to chains of a clique embedding.

    Note:
        Due to internal optimizations, not all Chimera graphs are supported by
        this code. Specifically, the graphs :func:`dwave_networkx.chimera_graph(m, n, t)`
        are only supported for :math:`t<=8`. The code currently supports D-Wave 
        products, which have :math:`t=4`, but not all graphs. For graphs with 
        :math:`t>8`, use the legacy chimera-embedding package.

    Note:
        When the cache is used, clique embeddings of all sizes are computed
        and cached. This takes somewhat longer than a single embedding, but tends
        to pay off after a fairly small number of calls. An exceptional use case 
        is when there are a large number of missing internal couplers, where the 
        result is nondeterministic -- avoiding the cache in this case may be 
        preferable.
    
    """
    try:
        graphdata = g.graph
        family = graphdata['family']
        busgraph = {'pegasus': _pegasus_busgraph,
                    'zephyr': _zephyr_busgraph,
                    'chimera': _chimera_busgraph}[family]
    except (AttributeError, KeyError):
        raise ValueError(("input graph must either be a "
                          "dwave_networkx.pegasus_graph, "
                          "dwave_networkx.chimera_graph or "
                          "dwave_networkx.zephyr_graph"))

    if use_cache:
        if seed is _no_seed:
            seed = 0
        return busgraph_cache(g, seed=seed).find_clique_embedding(nodes)
    else:
        if seed is _no_seed:
            seed = None
        return busgraph(g, seed=seed).find_clique_embedding(nodes)

class busgraph_cache:
    """A cache class for Chimera, Pegasus and Zephyr graphs, and their 
    associated cliques and bicliques.

    The cache files are stored in a directory determined by `homebase` (use
    :meth:`busgraph_cache.cache_rootdir` to retrieve the path to this directory). 
    Subdirectories named `cliques` and `bicliques` are then created to store the
    respective caches in each.

    Args:
        g (NetworkX Graph):
            A :func:`dwave_networkx.pegasus_graph` or :func:`dwave_networkx.chimera_graph`.
                or :func:`dwave_networkx.zephyr_graph`.
    Note:
        Due to internal optimizations, not all Chimera graphs are supported by
        this code. Specifically, the graphs :func:`dwave_networkx.chimera_graph(m, n, t)`
        are only supported for :math:`t<=8`. The code currently supports D-Wave 
        products, which have :math:`t=4`, but not all graphs. For graphs with 
        :math:`t>8`, use the legacy chimera-embedding package.
    
    """
    def __init__(self, g, seed = 0):
        self._family = g.graph['family']
        graphclass = {'pegasus': _pegasus_busgraph,
                      'zephyr': _zephyr_busgraph,
                      'chimera': _chimera_busgraph}.get(self._family)
        if graphclass is None:
            raise ValueError(("input graph must either be a "
                              "dwave_networkx.pegasus_graph, "
                              "dwave_networkx.chimera_graph or "
                              "dwave_networkx.zephyr_graph"))
        self._graph = graphclass(g, seed=seed, compute_identifier=True)
        self._cliques = None
        self._bicliques = None

    def _ensure_clique_cache(self):
        """Fetch/compute the clique cache, if it's not already in memory."""
        if self._cliques is None:
            self._cliques = self._fetch_cache('clique', self._graph.cliques)


    def _ensure_biclique_cache(self):
        """Fetch/compute the clique cache, if it's not already in memory."""
        if self._bicliques is None:
            self._bicliques = self._fetch_cache('biclique', 
                                                self._graph.bicliques)

    @staticmethod
    def cache_rootdir(version=__cache_version):
        """Returns the directory corresponding to the provided cache version.

        Args:
            version (int, optional, default=current cache version):
                Cache version.

        Returns:
            str
        """
        return homebase.user_data_dir('busclique', 'dwave', version)

    @staticmethod
    def clear_all_caches():
        """Removes all caches created by this class, up to and including the
        current version.

        Returns:
            None
        """
        dirstack = []
        for i in range(__cache_version + 1):
            rootdir = pathlib.Path(busgraph_cache.cache_rootdir(i))
            if rootdir.exists():
                dirstack.append(rootdir)
        while dirstack:
            top = dirstack[-1]
            substack = []
            for item in top.iterdir():
                if item.is_dir():
                    substack.append(item)
                else:
                    item.unlink()
            if substack:
                dirstack.extend(substack)
            else:
                top.rmdir()
                dirstack.pop()

    def _fetch_cache(self, dirname, compute):
        """This is an ad-hoc implementation of a file-cache using a LRU strategy.
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
        using a single file to store the entire cache.  In the case of hash
        collisions, the databases of multiple graphs will be stored in a single
        cache file -- so that file contains a `pickle`'d dict mapping
        indentifiers (not shortcodes) to the corresponding clique/biclique
        caches.

        Inside a cache directory, we have two special files and perhaps many
        cache files.  The `.lock` file is used by `fasteners.InterProcessLock`
        to provide process-level locking.  The `.lru` file is a list of up to
        `__lru_size` different cache filenames, sorted descending in the
        recentness of the last access of a given filename.  This enables us to
        automatically clean up the cache before it gets too large.
        """
        rootdir = busgraph_cache.cache_rootdir()
        basedir = os.path.join(rootdir, dirname)
        pathlib.Path(basedir).mkdir(parents=True, exist_ok=True)
        lockfile = os.path.join(basedir, ".lock")
        lrufile = os.path.join(basedir, ".lru")
        identifier = self._graph.identifier
        shortcode = self._graph.short_identifier
        
        #this makes our writes to the filesystem atomic
        file_context = copy_on_close_context()

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
                    with file_context.open(cachefile, 'wb') as filecache:
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

                #this violates atomicity, but it's okay -- at worst, we'll need
                #to recompute a deleted file that still has an entry in the LRU
                while len(newLRU) > __lru_size:
                    oldkey = newLRU.pop()
                    os.remove(os.path.join(basedir, oldkey))
                with file_context.open(lrufile, 'wb') as lru:
                    dump(newLRU, lru)

        #perform the copy-on-write for the lrufile and cachefile
        file_context.close()
        return cache

    def largest_clique(self):
        """Returns the largest-found clique in the clique cache.
        
        This will compute the entire clique cache if it is missing from the 
        filesystem.

        Returns:
            dict: An embedding of node labels from ``range(len(embedding))`` 
            mapped to chains of the largest-found clique.
        
        """
        self._ensure_clique_cache()
        embs = self._cliques['raw']
        keys = self._cliques['size']
        return self._graph.relabel(dict(enumerate(embs[keys[max(keys)]])))

    def largest_clique_by_chainlength(self, chainlength):
        """Returns the largest-found clique in the clique cache, with a specified
        maximum ``chainlength``.

        This will compute the entire clique cache if it is missing from the 
        filesystem.

        Args:
            chainlength (int):
                Max chain length.

        Returns:
            dict: An embedding of node labels from ``range(len(embedding))`` 
            mapped to chains of the largest-found clique with maximum ``chainlength``.
        
        """
        self._ensure_clique_cache()
        embs = self._cliques['raw']

        if 0 <= chainlength < len(embs):
            return self._graph.relabel(dict(enumerate(embs[chainlength])))
        else:
            return {}

    def find_clique_embedding(self, nn):
        """Returns a clique embedding, minimizing the maximum chainlength given 
        its size.
        
        This will compute the entire clique cache if it is missing from
        the filesystem.
        
        Args:
            nn (int/iterable):
                A number (indicating the size of the desired clique) or an 
                iterable (specifying the node labels of the desired clique).

        Returns:
            dict: An embedding of node labels (either ``nn``, or ``range(nn)``) 
            mapped to chains of a clique embedding.
        
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
        """Returns the largest-size biclique where both sides have equal size.

        Nodes of the embedding dict are from ``range(len(embedding))``, where the 
        nodes ``range(len(embedding)/2)`` are completely connected to the nodes 
        ``range(len(embedding)/2, len(embedding))``.

        This will compute the entire biclique cache if it is missing from the
        filesystem.

        Returns:
            dict: An embedding of node labels (described above) mapped to chains 
            of the largest balanced biclique.

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
        """Returns a biclique embedding, minimizing the maximum chain length 
        given its size. 
        
        This will compute the entire biclique cache if it is missing from the 
        filesystem.
        
        Args:
            nn (int/iterable):
                A number (indicating the size of one side of the desired biclique) 
                or an iterable (specifying the node labels of one side the desired
                biclique).

            mm (int/iterable):
                Same as ``nn``, for the other side of the desired biclique.

        In the case that ``nn`` is a number, the first side will have nodes 
        labeled from ``range(nn)``. In the case that ``mm`` is a number, the 
        second side will have nodes labeled from ``range(n, n + mm)``; where 
        ``n`` is either ``nn`` or ``len(nn)``.

        Returns:
            dict: An embedding of node labels (described above) mapped to chains 
            of a biclique embedding.
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

cdef dict _make_biclique_cache(vector[pair[pair[size_t, size_t],
                               embedding_t]] &embs):
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

ctypedef fused topo_cache_t:
    topo_cache[zephyr_spec]
    topo_cache[pegasus_spec]
    topo_cache[chimera_spec]

cdef _serialize(topo_cache_t *topo, str family):
    cdef size_t nchars = topo.serialize(serialize_size_tag(), NULL)
    cdef char *tmp0 = <char *>PyMem_Malloc(nchars)
    cdef uint8_t *tmp = <uint8_t *>tmp0
    cdef uint64_t *tmp64 = <uint64_t*>(<void *>(tmp));
    cdef uint64_t shortcode_state = 0
    cdef uint64_t c
    cdef object ret, shortcode
    cdef size_t i, j, d, _

    #print(f"{<long int><void *>tmp0:x}")
    if tmp == NULL:
        raise MemoryError("could not allocate memory for topology serialization")

    topo.serialize(serialize_write_tag(), tmp);
    ret = PyBytes_FromStringAndSize(tmp0, nchars);

    # this is a hash-like operation.  It is *not* a good hash.  It's optimized
    # for speed and produces a 16-byte key which is reasonably sensitive to input.
    for i in range(nchars//8):
        shortcode_state += tmp64[i]
    for j in range(nchars-(nchars&7), nchars):
        shortcode_state += tmp[j] << (j&3)

    PyMem_Free(tmp0)
    #print(''.join('{:02x}'.format(x) for x in ret))
    shortcode = "{}_{:16x}".format(family, shortcode_state)
    return ret, shortcode

cdef class _zephyr_busgraph:
    cdef topo_cache[zephyr_spec] *topo
    cdef nodes_t nodes
    cdef embedding_t emb_1
    cdef readonly object relabel
    cdef readonly object identifier
    cdef readonly object short_identifier
    def __cinit__(self, g, seed = 0, compute_identifier = False):
        """
        This is a class which manages a single zephyr graph, and dispatches 
        various structure-aware c++ embedding functions on it.
        """
        cdef size_t rows = g.graph['rows']
        cdef size_t cols = g.graph['columns']
        cdef size_t tile = g.graph['tile']
        if tile > 4:
            raise NotImplementedError(("this clique embedder supports zephyr "
                                       "graphs with a tile size of 4 or less"))

        cdef uint32_t internal_seed
        if seed is None:
            seed_obj = os.urandom(sizeof(uint32_t))
            internal_seed = (<uint32_t *>(<void *>(<uint8_t *>(seed_obj))))[0]
        else:
            internal_seed = seed

        cdef zephyr_spec *zep = new zephyr_spec(rows, tile, internal_seed)
        coordinates = dnx.zephyr_coordinates(rows)
        cdef edges_t edges
        if g.graph['labels'] == 'int':
            self.nodes = g.nodes()
            edges = g.edges()
            self.relabel = _trivial_relabeler
        elif g.graph['labels'] == 'coordinate':
            self.nodes = coordinates.iter_zephyr_to_linear(g.nodes())
            edges = coordinates.iter_zephyr_to_linear_pairs(g.edges())
            self.relabel = _make_relabeler(coordinates.iter_linear_to_zephyr)
        else:
            raise ValueError("unrecognized graph labeling")

        self.topo = new topo_cache[zephyr_spec](zep[0], self.nodes, edges)
        short_clique(zep[0], self.nodes, edges, self.emb_1)

        if compute_identifier:
            self.identifier, self.short_identifier = _serialize(self.topo, 'zephyr')

        del zep

    def __dealloc__(self):
        del self.topo

    def bicliques(self):
        """
        Returns a biclique cache -- see _make_biclique_cache for more info.
        """
        cdef vector[pair[pair[size_t, size_t], embedding_t]] embs
        best_bicliques[zephyr_spec](self.topo[0], embs)
        return _make_biclique_cache(embs)

    def cliques(self):
        """
        Returns a clique cache -- see _make_clique_cache for more info.
        """
        cdef vector[embedding_t] embs
        best_cliques[zephyr_spec](self.topo[0], embs, self.emb_1)
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
        if num <= self.emb_1.size():
            emb = self.emb_1
        elif not find_clique(self.topo[0], num, emb):
            return {}
        return self.relabel(dict(zip(nodes, emb)))

cdef class _pegasus_busgraph:
    cdef topo_cache[pegasus_spec] *topo
    cdef nodes_t nodes
    cdef embedding_t emb_1
    cdef readonly object relabel
    cdef readonly object identifier
    cdef readonly object short_identifier
    def __cinit__(self, g, seed = 0, compute_identifier = False):
        """
        This is a class which manages a single pegasus graph, and dispatches 
        various structure-aware c++ embedding functions on it.
        """
        rows = g.graph['rows']
        voff = [o//2 for o in g.graph['vertical_offsets'][::2]]
        hoff = [o//2 for o in g.graph['horizontal_offsets'][::2]]
        
        cdef uint32_t internal_seed
        if seed is None:
            seed_obj = os.urandom(sizeof(uint32_t))
            internal_seed = (<uint32_t *>(<void *>(<uint8_t *>(seed_obj))))[0]
        else:
            internal_seed = seed

        cdef pegasus_spec *peg = new pegasus_spec(rows, voff, hoff, internal_seed)
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

        if compute_identifier:
            self.identifier, self.short_identifier = _serialize(self.topo, 'pegasus')

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
        if num <= self.emb_1.size():
            emb = self.emb_1
        elif not find_clique(self.topo[0], num, emb):
            return {}
        return self.relabel(dict(zip(nodes, emb)))

cdef class _chimera_busgraph:
    """Class for managing a single Chimera graph, and dispatches various 
    structure-aware C++ embedding functions on it.

    Note: 
        Due to internal optimizations, not all Chimera graphs are supported by
        this code. Specifically, the graphs :func:`dwave_networkx.chimera_graph(m, n, t)`
        are only supported for :math:`t<=8`. The code currently supports D-Wave 
        products, which have :math:`t=4`, but not all graphs. For graphs with 
        :math:`t>8`, use the legacy chimera-embedding package.
    """
    cdef topo_cache[chimera_spec] *topo
    cdef embedding_t emb_1
    cdef nodes_t nodes
    cdef readonly object relabel
    cdef readonly object identifier
    cdef readonly object short_identifier
    def __cinit__(self, g, seed = 0, compute_identifier = False):
        rows = g.graph['rows']
        cols = g.graph['columns']
        tile = g.graph['tile']
        if tile > 8:
            raise NotImplementedError(("this clique embedder supports chimera "
                                       "graphs with a tile size of 8 or less"))
        cdef uint32_t internal_seed
        if seed is None:
            seed_obj = os.urandom(sizeof(uint32_t))
            internal_seed = (<uint32_t *>(<void *>(<uint8_t *>(seed_obj))))[0]
        else:
            internal_seed = seed

        cdef chimera_spec *chim = new chimera_spec(rows, cols, tile, internal_seed)
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

        if compute_identifier:
            self.identifier, self.short_identifier = _serialize(self.topo, 'chimera')

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
        if num <= self.emb_1.size():
            emb = self.emb_1
        elif not find_clique(self.topo[0], num, emb):
            return {}
        return self.relabel(dict(zip(nodes, emb)))

class copy_on_close_context:
    def __init__(self):
        self.files = {}
        
    def open(self, file_name, mode):
        if file_name in self.files:
            self.files.clear()
            raise RuntimeError("copy_on_close_context can only open files once.  Discarding all files.")
        else:
            temp_name = file_name + '~'
            file_obj = open(temp_name, mode)
            self.files[file_name] = temp_name, file_obj
            return file_obj        

    def close(self):
        for file_name, (temp_name, file_obj) in self.files.items():
            file_obj.close()
            os.replace(temp_name, file_name)

