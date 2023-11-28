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

import homebase
import os
import pathlib
import fasteners
import threading
import random
import gzip
from pickle import dump, load
from json import dumps, loads
import networkx as nx
import dwave_networkx as dnx
from itertools import zip_longest
from hashlib import sha256

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.bytes cimport PyBytes_FromStringAndSize


#increment this version any time there is a change made to the cache format,
#when yield-improving changes are made to clique algorithms, or when bugs are
#fixed in the same.
cdef int __cache_version = 7

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

    def _ensure_clique_cache(self, blank=False):
        """Fetch/compute the clique cache, if it's not already in memory."""
        if self._cliques is None:
            if blank:
                self._cliques = _make_clique_cache([[]])
            else:
                self._cliques = self._fetch_cache(
                    'clique',
                    self._graph.cliques
                )

    def _ensure_biclique_cache(self):
        """Fetch/compute the clique cache, if it's not already in memory."""
        if self._bicliques is None:
            self._bicliques = self._fetch_cache('biclique', self._graph.bicliques)

    def topology_identifier(self):
        """Return a string identifying the busgraph basing this cache.

        Note that we're using sha256 to generate this.  If a collision is detected,
        the newsworthiness of that would be worth the hassle of dealing with the
        fallout."""
        s = sha256(self._graph.identifier)
        s.update(__cache_version.to_bytes(sizeof(__cache_version), 'little'))
        return s.hexdigest()

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

    def get_clique_cache(self, sparse=False, compress=True):
        """Returns an object representing the cliques stored in this cache.

        Args:
            sparse: (bool, optional, default=False)
                If True, the cache will be a dictionary mapping embedding sizes
                to lists of chains.  Otherwise, the cache will be a list of
                lists of chains; the index of the outermost list corresponding
                to embedding size.

            compress: (bool, optional, default=True)
                if True, the cache will be json-utf8-gzipped.  Otherwise, the
                cache will be a python object whose type is determined by the
                ``sparse`` argument.
        Returns:
            a cache object

        """
        self._ensure_clique_cache()
        raw = self._cliques['raw']
        size = self._cliques['size']
        if sparse:
            cache = {i: raw[s][:i] for i, s in size.items()}
        else:
            cache = [raw[size[i]][:i] if i in size else () for i in range(max(size))]
        if compress:
            cache = gzip.compress(dumps(cache).encode('utf8'))
        return cache

    def update_clique_cache(self, cache, write_to_disk = True):
        """Replaces or sets the cliques stored by this cache with the provided
        cache object.

        Args:
            cache: (bytes, str, dict or iterable)
                The cache object to replace.  Should be an object returned by
                :func:`self.get_clique_cache`.

            write_to_disk: (bool, optional, default=True)
                If True, the updated cache will be written to disk.
                Otherwise, the cache will only be updated in memory.
        """
        if isinstance(cache, bytes):
            cache = gzip.decompress(cache).decode('utf8')
        if isinstance(cache, str):
            cache = loads(cache)
        if isinstance(cache, dict):
            cache = cache.items()
        else:
            cache = enumerate(cache)

        empty = True
        for size, emb in cache:
            empty = False
            self.insert_clique_embedding(emb, write_to_disk=False, direct=True)

        if empty:
            self._ensure_clique_cache(blank=True)

        if write_to_disk:
            self._fetch_cache('clique', lambda: self._cliques, force_write=True)

    def insert_clique_embedding(self, emb, write_to_disk=True, quality_function=None, direct=False):
        """Inserts a clique embedding into the cache.

        Args:
            emb: (dict or list)
                The embedding to insert into the cache.

            write_to_disk: (bool, optional, default=True)
                If True, the updated cache will be written to disk.
                Otherwise, the cache will only be updated in memory.

            quality_function: (callable or None)
                A function used to evaluate the quality of an embedding before
                incorporating it into the cache.  If ``quality_function`` is
                None, preference is given to the embedding with shortest maximum
                chain length.  The embeddings evaluated by ``quality_function``
                will be provided as a list of list of integers, corresponding
                to the integer-labeled graph underlying ``self``.

            direct: (bool, optional, default=False)
                If True, the embedding will be stored directly into the cache,
                overwriting the current clique of that size if it exists.  In
                this case, ``quality_function`` must be ``None``.
                Otherwise, the embedding and embeddings derived by removing
                chains in decreasing order of length may be stored into the
                cache, if ``quality function`` evaluated on the new embedding is
                greater than the old.  If ``quality_function`` is None, new
                embeddings are retained when their chainlength is shorter than
                the old.
        """
        if not isinstance(emb, dict):
            emb = dict(enumerate(emb))
        emb = self._graph.delabel(emb)
        emb = sorted(emb.values(), key=len)
        if direct:
            if quality_function is not None:
                raise ValueError("quality_function is ignored for direct insertion")
            self._ensure_clique_cache(blank=True)
            raw = self._cliques['raw']
            key = len(raw)
            raw.append(emb)
            by_length = self._cliques['length']
            self._cliques['size'][len(emb)] = key
            length = len(emb[-1]) if emb else 0
            if length not in by_length or len(raw[by_length[length]]) < len(emb):
                by_length[length] = key
            if write_to_disk:
                self._fetch_cache('clique', lambda: self._cliques, force_write=True)
        else:
            max_length = max(map(len, emb)) if emb else 0
            emb_list = [[c for c in emb if len(c) <= i] for i in range(max_length+1)]
            cache = _make_clique_cache(emb_list)
            self.merge_clique_cache(cache, quality_function, write_to_disk)

    def merge_clique_cache(self, cache, quality_function=None, write_to_disk=True):
        """Merges the clique cache of another ``busgraph_cache`` object

        Args:
            cache: (busgraph_cache or object)
                If ``cache`` is a ``busgraph_cache``, its clique cache will be
                incorporated that of ``self``.  Otherwise, the cache object is
                an undocumented, internal cache representation (do not use).

            quality_function: (callable or None)
                A function used to evaluate the quality of an embedding before
                incorporating it into the cache.  If ``quality_function`` is
                None, preference is given to the embedding with shortest maximum
                chain length.  The embeddings evaluated by ``quality_function``
                will be provided as a list of list of integers, corresponding
                to the integer-labeled graph underlying ``self``.

            write_to_disk: (bool, optional, default=True)
                If True, the merged cache will be written to disk.
                Otherwise, the cache will only be updated in memory.
        """
        cdef vector[embedding_t] embs
        self._ensure_clique_cache(blank=True)
        if isinstance(cache, busgraph_cache):
            cache._ensure_clique_cache()
            cache = cache._cliques
        raw0 = cache['raw']
        raw1 = self._cliques['raw']
        used = {}

        size = {}
        size0 = cache['size']
        size1 = self._cliques['size']
        for i in sorted(set(size0).union(size1)):
            key0 = size0.get(i)
            key1 = size1.get(i)
            if key1 is None:
                key = used.setdefault((key0, 0), len(used))
            elif key0 is None:
                key = used.setdefault((key1, 1), len(used))
            else:
                if i == 0:
                    qual0 = qual1 = 0
                elif quality_function is None:
                    # by default, shortest chainlength is highest quality
                    qual0 = -len(raw0[key0][i-1])
                    qual1 = -len(raw1[key1][i-1])
                else:
                    qual0 = quality_function(raw0[key0][:i])
                    qual1 = quality_function(raw1[key1][:i])
                if qual0 > qual1:
                    key = used.setdefault((key0, 0), len(used))
                else:
                    # if the quality is the same, retain the old embedding
                    key = used.setdefault((key1, 1), len(used))
            size[i] = key

        length = {}
        length0 = cache['length']
        length1 = self._cliques['length']
        for i in sorted(set(length0).union(length1)):
            key0 = length0.get(i)
            key1 = length1.get(i)
            if key1 is None:
                key = used.setdefault((key0, 0), len(used))
            elif key0 is None:
                key = used.setdefault((key1, 1), len(used))
            elif len(raw0[key0]) > len(raw1[key1]):
                # pick the larger embedding
                key = used.setdefault((key0, 0), len(used))
            else:
                # if the embeddings have the same size, retain the old one
                key = used.setdefault((key1, 1), len(used))
            length[i] = key

        raw = [raw0[key] if i == 0 else raw1[key] for (key, i) in used]
        self._cliques = {"raw": raw, "size": size, "length": length}
        if write_to_disk:
            self._fetch_cache('clique', lambda: self._cliques, force_write=True)

    def _fetch_cache(self, dirname, compute, force_write=False):
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
                cache = None if force_write else currcache.get(identifier)
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
        if keys:
            return self._graph.relabel(dict(enumerate(embs[keys[max(keys)]])))
        else:
            return {}

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
        length = self._cliques['length'] or [0]

        if 0 < chainlength <= max(length):
            return self._graph.relabel(dict(enumerate(embs[length[chainlength]])))
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
        raw = self._cliques['raw']
        key = keys.get(num)
        if key is None:
            return {}
        emb = dict(zip(nodes, raw[key]))
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

    def draw_fragment_embedding(self, emb, **kwargs):
        m, n, t, nodes, edges = self._graph.fragment_graph_spec()

        f = dnx.chimera_graph(m, n=n, t=t, node_list=nodes, edge_list=edges)
        f_emb = {
            k : self._graph.fragment_nodes(c)
            for k, c in self._graph.delabel(emb).items()
        }
        dnx.draw_chimera_embedding(f, f_emb, **kwargs)

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
    cdef dict by_length = {}
    cdef list raw = []
    for length, emb in enumerate(embs):
        by_length[length] = length
        raw.append(tuple(map(tuple, emb)))
        maxsize = max(emb.size(), maxsize)
        for i in range(maxsize, -1, -1):
            if by_size.setdefault(i, length) != length:
                break
    return {'raw': raw, 'size': by_size, 'length': by_length}

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
    return {v: tuple(chain) for v, chain in emb.items()}

def _make_relabeler(f):
    """
    This returns an embedding-relabeling function, which applies `f` to each
    chain in the embedding.
    """
    def _relabeler(emb):
        return {v: tuple(f(chain)) for v, chain in emb.items()}
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
    shortcode = "{}_{:16x}".format(family, shortcode_state)
    return ret, shortcode

cdef class _zephyr_busgraph:
    cdef topo_cache[zephyr_spec] *topo
    cdef nodes_t nodes
    cdef embedding_t emb_1
    cdef readonly object relabel
    cdef readonly object delabel
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
            self.delabel = _trivial_relabeler
        elif g.graph['labels'] == 'coordinate':
            self.nodes = coordinates.iter_zephyr_to_linear(g.nodes())
            edges = coordinates.iter_zephyr_to_linear_pairs(g.edges())
            self.relabel = _make_relabeler(coordinates.iter_linear_to_zephyr)
            self.delabel = _make_relabeler(coordinates.iter_zephyr_to_linear)
        else:
            raise ValueError("unrecognized graph labeling")

        self.topo = new topo_cache[zephyr_spec](zep[0], self.nodes, edges)
        short_clique[zephyr_spec](zep[0], self.nodes, edges, self.emb_1)

        if compute_identifier:
            self.identifier, self.short_identifier = _serialize(self.topo, 'zephyr')

        del zep

    def __dealloc__(self):
        del self.topo

    def set_mask_bound(self, num_masks):
        """Sets the maximum number of masks to consider."""
        self.topo[0].set_mask_bound(num_masks)

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

    def fragment_graph_spec(self):
        m = coordinate_index(self.topo.topo.dim_y)
        n = coordinate_index(self.topo.topo.dim_x)
        t = self.topo.topo.shore
        nodes = self.topo.fragment_nodes()
        edges = self.topo.fragment_edges()
        return m, n, t, nodes, edges

    def fragment_nodes(self, nodes = None):
        if nodes is None:
            return self.topo.fragment_nodes()
        else:
            return self.topo.topo.fragment_nodes(<nodes_t&>nodes)

cdef class _pegasus_busgraph:
    cdef topo_cache[pegasus_spec] *topo
    cdef nodes_t nodes
    cdef embedding_t emb_1
    cdef readonly object relabel
    cdef readonly object delabel
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
            self.delabel = _trivial_relabeler
        elif g.graph['labels'] == 'coordinate':
            self.nodes = coordinates.iter_pegasus_to_linear(g.nodes())
            edges = coordinates.iter_pegasus_to_linear_pairs(g.edges())
            self.relabel = _make_relabeler(coordinates.iter_linear_to_pegasus)
            self.delabel = _make_relabeler(coordinates.iter_pegasus_to_linear)
        elif g.graph['labels'] == 'nice':
            self.nodes = coordinates.iter_nice_to_linear(g.nodes())
            edges = coordinates.iter_nice_to_linear_pairs(g.edges())
            self.relabel = _make_relabeler(coordinates.iter_linear_to_nice)
            self.delabel = _make_relabeler(coordinates.iter_nice_to_linear)
        else:
            raise ValueError("unrecognized graph labeling")

        self.topo = new topo_cache[pegasus_spec](peg[0], self.nodes, edges)
        short_clique[pegasus_spec](peg[0], self.nodes, edges, self.emb_1)

        if compute_identifier:
            self.identifier, self.short_identifier = _serialize(self.topo, 'pegasus')

        del peg

    def __dealloc__(self):
        del self.topo

    def set_mask_bound(self, num_masks):
        """Sets the maximum number of masks to consider."""
        self.topo[0].set_mask_bound(num_masks)

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

    def fragment_graph_spec(self):
        m = coordinate_index(self.topo.topo.dim_y)
        n = coordinate_index(self.topo.topo.dim_x)
        t = self.topo.topo.shore
        nodes = self.topo.fragment_nodes()
        edges = self.topo.fragment_edges()
        return m, n, t, nodes, edges

    def fragment_nodes(self, nodes = None):
        if nodes is None:
            return self.topo.fragment_nodes()
        else:
            return self.topo.topo.fragment_nodes(<nodes_t&>nodes)

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
    cdef readonly object delabel
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
            self.delabel = _trivial_relabeler
        elif g.graph['labels'] == 'coordinate':
            self.nodes = coordinates.iter_chimera_to_linear(g.nodes())
            edges = coordinates.iter_chimera_to_linear_pairs(g.edges())
            self.relabel = _make_relabeler(coordinates.iter_linear_to_chimera)
            self.delabel = _make_relabeler(coordinates.iter_chimera_to_linear)
        else:
            raise ValueError("unrecognized graph labeling")

        self.topo = new topo_cache[chimera_spec](chim[0], self.nodes, edges)
        short_clique[chimera_spec](chim[0], self.nodes, edges, self.emb_1)

        if compute_identifier:
            self.identifier, self.short_identifier = _serialize(self.topo, 'chimera')

        del chim


    def __dealloc__(self):
        del self.topo

    def set_mask_bound(self, num_masks):
        """Sets the maximum number of masks to consider."""
        self.topo[0].set_mask_bound(num_masks)

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

    def fragment_graph_spec(self):
        m = coordinate_index(self.topo.topo.dim_y)
        n = coordinate_index(self.topo.topo.dim_x)
        t = self.topo.topo.shore
        nodes = self.topo.fragment_nodes()
        edges = self.topo.fragment_edges()
        return m, n, t, nodes, edges

    def fragment_nodes(self, nodes = None):
        if nodes is None:
            return self.topo.fragment_nodes()
        else:
            return self.topo.topo.fragment_nodes(<nodes_t&>nodes)

def _default_quality_function(emb):
    """A function that returns a tuple corresponds to the "quality" of the embedding.

    The "quality" of the embedding is measured by the maximum chainlength (to be
    minimized), and the number of chains with that length (to be maximized).

    Args:
        emb: (list/tuple of lists/tuples)
            The embedding.

    Returns:
        quality: (tuple)

    Usage:
        better_emb = max(emb0, emb1, key = _default_quality_function)

    """
    maxlen = max(map(len, emb))
    return -maxlen, sum(len(chain) == maxlen for chain in emb)

def _regularize_embedding(g, emb):
    """Best-effort augmented embedding, with hopefully uniform chainlegnths.

    We augment ``emb`` by adding nodes to chains that are shorter than the chain
    with maximum length.

    Args:
        emb: (dict with list/tuple values)
            The embedding

    Returns:
        emb: (dict with tuple values)
            The augmented embedding.

    """
    progress = len(set(map(len, emb.values()))) != 1
    max_length = max(map(len, emb.values()))
    emb = {i: list(chain) for i, chain in emb.items()}
    remainder = set(g) - set().union(*emb.values())
    while progress:
        progress = False
        matching_graph = nx.Graph()
        top_nodes = []
        for i, chain in emb.items():
            if len(chain) == max_length:
                continue
            boundary = nx.node_boundary(g, chain, remainder)
            if not boundary:
                continue
            for j in range(len(chain), max_length):
                chain_node = (i, 'chain', j)
                top_nodes.append(chain_node)
                matching_graph.add_edges_from((chain_node, q) for q in boundary)
        matching = nx.bipartite.maximum_matching(matching_graph, top_nodes)
        for chain_node in top_nodes:
            matched = matching.get(chain_node)
            if matched is not None:
                remainder.discard(matched)
                progress = True
                emb[chain_node[0]].append(matched)
        if progress:
            progress = len(set(map(len, emb.values()))) != 1
    return {i: tuple(chain) for i, chain in emb.items()}

def mine_clique_embeddings(
        g,
        quality_function=None,
        regularize=True,
        num_seeds=16,
        mask_bound=64,
        heuristic_bound=None,
        heuristic_args=None,
        heuristic_runs=10,
    ):
    """Use heuristic and polynomial methods to construct a cache of clique embeddings.

    This function uses both the heuristic :func:`minorminer.find_embedding` and
    the polynomial algorithm in :func:`busclique.find_clique_embedding` to find
    clique embeddings, and collects the best results of the two methods into a
    single :class:`busgraph_cache` object.  One purpose of this function is to
    identify optimized embeddings for hardware graphs, to be provided through
    Ocean structured solvers and samplers.

    If a disk-cache for ``g`` already exists, it will be augmented with the
    embeddings discovered during execution.  The resulting cache will be writte
    to disk afterwards, so repeated calls to this function can only improve the
    quality of the cache.

    The method operates by
        (1) Load or construct a cache of embeddings for ``g`` with
            :class:`busgraph_cache`.
        (2) For 0 <= n < num_seeds, compute a :class:`busgraph_cache` with a
            random seed, and insert the corresponding embeddings into the cache.
        (3) For 3 <= n <= heuristic_bound, find clique embeddings of size n
            into ``g`` with :func:`minorminer.find_embedding`, and insert them
            into the cache if they improve the quality.

    Args:
        g: (networkx.Graph)
            The graph to compute a clique cache for.

        quality_function: (callable, optional, default=None)
            A function that assesses the "quality" of an embedding and returns
            an object that can be compared to other values returned by
            ``quality_function``.  The embeddings are represented as lists of
            lists of ints corresponding to the integer-labeled topology of ``g``.
            If None, we use :func:`_default_quality_function`.

        regularize: (bool, optional, default=True)
            If True, we attempt to augment the found embeddings by adding extra
            nodes in order to produce embeddings with all-equal chainlengths.

        num_seeds: (int, optional, default=16)
            The number of random seeds to compute :class:`busgraph_cache`.

        mask_bound: (int, optional, default=64)
            The number of topologies considered internal to :class:`busgraph_cache`.

        heuristic_bound: (int, optional, default=None)
            The maximum clique size to attempt heuristic embedding.  If None,
            a default value is computed based on the topology parameters of ``g``.

        heuristic_args: (dict, optional, default=None)
            Keyword arguments to be used when calling the heuristic embedder.
            See :func:`minorminer.find_embedding` for more details.

        heurisitc_runs: (int, optional, default=10)
            The number of times to attempt heuristic embedding per size.

    Returns:
        cache: (busgraph_cache)

    """

    from minorminer import miner
    import logging
    logger = logging.getLogger(__name__)
    bgc = busgraph_cache(g)
    for i in range(num_seeds):
        logger.info("polynomial embedder run %d of %d", i+1, num_seeds)
        seed = random.randint(0, 2**32-1)
        bgc_i = busgraph_cache(g, seed=seed)
        bgc_i._graph.set_mask_bound(mask_bound)
        bgc.merge_clique_cache(bgc_i, write_to_disk=False, quality_function=quality_function)
        if regularize:
            for n in range(3, len(bgc_i.largest_clique())+1):
                emb = _regularize_embedding(g, bgc_i.find_clique_embedding(n))
                bgc.insert_clique_embedding(emb, write_to_disk=False, quality_function=quality_function)

    if heuristic_bound is None:
        if bgc._family == "zephyr":
            tile = g.graph['tile']
            rows = g.graph['rows']
            heuristic_bound = min(8*tile, 2*tile*(rows - 1))
        elif bgc._family == "pegasus":
            rows = g.graph['rows']
            heuristic_bound = min(36, 12*(rows-1))
        elif bgc._family == "chimera":
            rows = g.graph['rows']
            cols = g.graph['cols']
            tile = g.graph['tile']
            heuristic_bound = (min(tile*rows, tile*cols)+3)/4
        else:
            raise NotImplementedError("graph family not supported")

    if quality_function is None:
        quality_function = _default_quality_function

    if heuristic_args is None:
        heuristic_args = dict(chainlength_patience=100, max_beta=3, tries=1)

    for size in range(3, heuristic_bound+1):
        logger.info("running heuristic embeddings for size %d", size)
        hm = miner(nx.complete_graph(size), g, **heuristic_args)
        for emb in hm.find_embeddings(heuristic_runs, force=False):
            if not emb:
                continue
            bgc.insert_clique_embedding(emb, write_to_disk=False, quality_function=quality_function)
            if regularize:
                emb1 = _regularize_embedding(g, emb)
                bgc.insert_clique_embedding(emb1, write_to_disk=False, quality_function=quality_function)

    logger.info("writing embedding cache to disk")
    bgc.insert_clique_embedding({}, write_to_disk=True)
    return bgc

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

