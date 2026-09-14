# Copyright 2026 D-Wave
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

"""Name-based quality criteria for ranking embeddings.

An embedding is a collection of chains; a chain is a collection of nodes
(``len(chain)`` is its length). "Quality" ranks embeddings so a search can keep
the best one(s) per size. The convention is fixed and load-bearing: **smaller is
better**.

The user specifies quality by NAME(s), never by writing functions::

    "max_chain_length"                        # single criterion
    ("max_chain_length", "mean_chain_length", "percentile_93_chain_length")
    # a hierarchy: rank by the first, ties broken by the next, and so on

:func:`build_quality` turns that into a callable ``embedding -> key``, where
``key`` is a tuple with one scalar per criterion. Python's lexicographic tuple
ordering then gives the hierarchy for free. An empty embedding -- or one that
contains an empty (zero-length) chain -- maps to ``+inf`` on every criterion, so
it can never rank above a real one.

The embedding passed to a quality function may be either a sequence of chains
(``[chain, ...]``) or a mapping of chain-id to chain (``{cid: chain, ...}``); a
mapping is scored over its values. A chain is any object with a length and
iterable nodes.

This module has NO dependency on any embedder or on graph/topology code. Node
labels are opaque (typically ``int``, ``str``, or a tuple of those) and are only
ever hashed, never inspected or compared, so they need only be hashable.

**Criterion names**

Length metrics:

``max_chain_length``
    Length of longest chain.
``mean_chain_length``
    Average chain length.

Parametric:

``percentile_<p>_chain_length``
    The ``p``-th percentile of chain lengths, e.g. ``percentile_93_chain_length``.
    ``p`` in ``[0, 100]``.

Fault-weighted (needs ``fault_map``, see :func:`build_quality`):

``faultiness``
    Total fault weight the embedding sits on: the sum of per-node and
    per-present-edge fault weights (in ``[0, 1]``) from ``fault_map``. See
    :func:`faultiness`.
"""

__all__ = ["build_quality", "faultiness"]

import re
from typing import Any, Callable, Iterable, Mapping, TypedDict


class FaultMap(TypedDict, total=False):
    """Fault weights for a graph, as consumed by the ``faultiness`` criterion.

    ``total=False``: either key may be absent (a missing key means "no faults of
    that kind"). Weights are in ``[0, 1]`` (0 = healthy, 1 = worst); they are the
    caller's responsibility and are not validated. An edge key is an unordered
    node pair: either endpoint order is accepted, and an edge listed under both
    orders is one edge, counted once.
    """

    nodes: dict[Any, float]  # node label -> fault weight
    edges: dict[tuple[Any, Any], float]  # (node, node) -> fault weight


# ==========================================================================
# Length metrics: (list of chain lengths) -> smaller is better
# ==========================================================================
# NB: these return a Python number that is int OR float (max of ints is int;
# mean is float). That is intentional: quality keys only ever compare a
# criterion against itself across embeddings, and int/float compare and sort
# together fine, so the annotations use the conventional `float` == "a number".
LENGTH_METRICS: dict[str, Callable[[list[int]], float]] = {
    "max_chain_length": lambda lengths: max(lengths),
    "mean_chain_length": lambda lengths: sum(lengths) / len(lengths),
}

# "percentile_<p>_chain_length" -> p
_PERCENTILE_RE = re.compile(r"^percentile_(\d+)_chain_length$")

# When _resolve looks at a criterion name, it decides which input that metric
# needs -- the raw chain lengths, the sorted lengths, or the chains themselves --
# and returns that choice as one of these three tags alongside the metric
# function. build_quality's `quality` then reads the tag to know which argument
# to pass. Defining the tags as named constants (instead of writing the strings
# "lengths"/"sorted_lengths"/"chains" directly in both places) means a typo
# becomes an immediate NameError rather than a silent mismatch that would feed a
# metric the wrong argument.
_LENGTHS = "lengths"
_SORTED_LENGTHS = "sorted_lengths"
_CHAINS = "chains"


def _percentile_sorted(sorted_values: list[float], percentile: float) -> float:
    """``p``-th percentile (``p`` in ``[0, 100]``) of an ALREADY-SORTED, non-
    empty list, by linear interpolation between the two closest ranks.

    :func:`build_quality`'s inner function sorts the length list at most once
    per embedding and passes it here, so several percentile criteria in one
    hierarchy share a single sort.
    """
    num_values = len(sorted_values)
    if num_values == 1:
        return sorted_values[0]
    rank = (percentile / 100) * (num_values - 1)  # fractional rank in [0, n-1]
    lo = int(rank)  # floor of rank -> the lower index
    frac = rank - lo  # how far past lo we are, in [0, 1)
    if lo + 1 >= num_values:  # p == 100 (or rounding to the top rank)
        return sorted_values[-1]
    return sorted_values[lo] + frac * (
        sorted_values[lo + 1] - sorted_values[lo]
    )


def _canonical_edge(u: Any, v: Any) -> frozenset[Any]:
    """Canonical (order-independent) key for the undirected edge {u, v}.

    A frozenset is canonical by construction -- {u, v} and {v, u} are equal --
    and needs only hashability of the labels, no ordering.
    """
    return frozenset((u, v))


def _prep_faultiness(
    fault_map: FaultMap | None,
    missing_edges: Iterable[tuple[Any, Any]] | None,
) -> tuple[dict[Any, float], dict[frozenset[Any], float]]:
    """Pre-process a fault map into the form :func:`_faultiness` consumes.

    Does all the once-only work -- extracting the node/edge sub-dicts,
    canonicalizing both the fault edges and the missing edges, and dropping the
    latter from the former -- so this runs once when :func:`build_quality`
    resolves the metric, not once per embedding scored. Returns
    ``(node_weights, edge_weights)`` where:

    * ``node_weights`` maps node -> weight (used as-is);
    * ``edge_weights`` maps a canonical (frozenset) edge key -> weight, with
      every missing edge already removed.

    Keying edges canonically means either endpoint order works in either input,
    and that a fault map listing an edge under BOTH orientations contributes its
    weight once, not twice. Two such entries collide on one key, so the later
    one in iteration order wins -- plain dict semantics; giving one edge two
    different weights is a caller error, not something this resolves.
    """
    if fault_map is None:
        fault_map = {}
    node_weights = fault_map.get("nodes", {})

    missing = set()
    if missing_edges:
        missing = {_canonical_edge(u, v) for (u, v) in missing_edges}

    edge_weights = {}
    for (u, v), weight in fault_map.get("edges", {}).items():
        edge = _canonical_edge(u, v)
        if edge in missing:
            continue  # edge absent -> cannot be faulty; drop it once, here
        edge_weights[edge] = weight

    return node_weights, edge_weights


def _faultiness(
    chains: Iterable[Iterable[Any]],
    node_weights: dict[Any, float],
    edge_weights: dict[frozenset[Any], float],
) -> float:
    """Sum fault weight over the embedding, against PRE-PROCESSED inputs from
    :func:`_prep_faultiness` (node weights, and canonical edge weights already
    stripped of missing edges).

    Iterating the (small) fault entries rather than all used-node pairs keeps
    this ``O(faults)``, not ``O(nodes^2)``. An edge counts when its canonical
    key is a subset of the used nodes, i.e. both endpoints are used.
    """
    used = {node for c in chains for node in c}

    total = 0
    for node, weight in node_weights.items():
        if node in used:
            total += weight

    for edge, weight in edge_weights.items():
        if edge <= used:
            total += weight

    return total


def faultiness(
    chains: Iterable[Iterable[Any]],
    fault_map: FaultMap | None = None,
    missing_edges: Iterable[tuple[Any, Any]] | None = None,
) -> float:
    """Total fault weight the embedding sits on, summed over its nodes and its
    present edges. Smaller is better::

        faultiness = sum(node fault weight for each used node)
                   + sum(edge fault weight for each present used edge)

    Any node or edge NOT in ``fault_map`` has weight 0 -- absence means "no known
    fault", treated as healthy. With no ``fault_map`` the embedding scores 0. The
    edge term counts a fault weight only when BOTH endpoints are used AND the edge
    actually exists (see ``missing_edges``).

    This is the standalone entry point (score one embedding once). The
    ``faultiness`` criterion of :func:`build_quality` does the same computation
    but pre-processes the fault map once at build time via the same helper, so a
    repeatedly-evaluated quality function pays the setup cost only once.

    Args:
        chains: the embedding, an iterable of chains (each an iterable of node
            labels).
        fault_map: fault weights, shape ``{"nodes": {...}, "edges": {...}}`` (see
            :class:`FaultMap`). Weights are the caller's responsibility --
            out-of-range values are summed as given, not validated. An edge key
            is an unordered node pair, so either order works and an edge listed
            under both orders is counted once.
        missing_edges: unordered node pairs whose edge is ABSENT between two
            present nodes. Such an edge is skipped in the sum, since an edge that
            isn't there cannot contribute fault. ``None`` means "assume every edge
            between present nodes exists".

    Returns:
        The summed fault weight (smaller is better).

    Note:
        Only node/edge FAULT weights are consulted; this module has no view of
        the graph, so it cannot (and need not) verify a used node is physically
        present. Absent NODES therefore have no analog of ``missing_edges`` -- a
        node you do not use contributes nothing regardless, and a node in the
        weight map is assumed to be a real, usable qubit.
    """
    node_weights, edge_weights = _prep_faultiness(fault_map, missing_edges)
    return _faultiness(chains, node_weights, edge_weights)


def _resolve(
    name: str,
    fault_map: FaultMap | None,
    missing_edges: Iterable[tuple[Any, Any]] | None,
) -> tuple[str, Callable]:
    """Resolve a criterion NAME to a ``(kind, func)`` pair (smaller is better).

    ``kind`` is one of the module constants ``_LENGTHS`` / ``_SORTED_LENGTHS``
    / ``_CHAINS``, telling :func:`build_quality`'s inner function which
    argument to pass ``func``: the raw length list, the sorted length list, or
    the chains themselves. This lets the quality function compute lengths (and
    sort them) at most once per embedding and share them across criteria.
    Raises :exc:`ValueError` on an unknown name.
    """
    if name in LENGTH_METRICS:
        return (_LENGTHS, LENGTH_METRICS[name])

    m = _PERCENTILE_RE.match(name)
    if m:
        p = int(m.group(1))
        if not (0 <= p <= 100):
            raise ValueError(
                f"percentile must be in [0, 100], got {p} in '{name}'"
            )
        return (_SORTED_LENGTHS, lambda sl, _p=p: _percentile_sorted(sl, _p))

    if name == "faultiness":
        # pre-process the fault map ONCE here; the metric closes over the result
        node_weights, edge_weights = _prep_faultiness(fault_map, missing_edges)
        return (
            _CHAINS,
            lambda chains: _faultiness(chains, node_weights, edge_weights),
        )

    known = sorted(LENGTH_METRICS) + ["faultiness"]
    raise ValueError(
        f"unknown quality criterion '{name}'. Known: {known} "
        f"or 'percentile_<p>_chain_length'."
    )


def build_quality(
    criteria: str | Iterable[str],
    fault_map: FaultMap | None = None,
    missing_edges: Iterable[tuple[Any, Any]] | None = None,
) -> Callable[
    [Mapping[Any, Iterable[Any] | Iterable[Iterable[Any]]]], tuple[float, ...]
]:
    """Build a quality function from a criterion name or an ordered collection.

    Args:
        criteria: a single criterion name (``str``) or an ordered tuple/list of
            names. The order defines the hierarchy: embeddings are compared by
            the first criterion, ties broken by the second, and so on.
        fault_map: fault weights (see :class:`FaultMap`), used only by the
            ``faultiness`` criterion (ignored otherwise).
        missing_edges: set of unordered node pairs whose edge is absent, used
            only by the ``faultiness`` criterion (ignored otherwise). Consumed
            once here, so a one-shot iterable (e.g. a generator) is fine.

    Returns:
        A function ``embedding -> key`` where ``key`` is a tuple with one number
        per criterion, smaller is better. The embedding may be a sequence of
        chains or a ``{chain_id: chain}`` mapping (scored over its values). An
        empty embedding, or one containing an empty (zero-length) chain, yields
        ``(+inf, ...)`` of the right arity, so it never ranks above a real one.

    Raises:
        ValueError: if ``criteria`` is empty or names an unknown criterion.

    Examples::

        build_quality("max_chain_length")
        build_quality(("max_chain_length", "percentile_93_chain_length"))
        build_quality("faultiness", fault_map=fault_map)
        build_quality(("max_chain_length", "faultiness"), fault_map=fault_map,
                      missing_edges=missing_edges)
    """
    if isinstance(criteria, str):
        names = (criteria,)
    else:
        names = tuple(criteria)
    if not names:
        raise ValueError(
            "criteria must name at least one criterion"
        )

    metrics = [_resolve(n, fault_map, missing_edges) for n in names]
    worst = (float("inf"),) * len(metrics)
    need_sorted = any(kind == _SORTED_LENGTHS for kind, _ in metrics)

    def quality(
        embedding: Mapping[Any, Iterable[Any]] | Iterable[Iterable[Any]],
    ) -> tuple[float, ...]:
        chains = list(
            embedding.values() if isinstance(embedding, Mapping) else embedding
        )
        if not chains:  # empty embedding: worst possible
            return worst

        lengths = [len(c) for c in chains]
        if 0 in lengths:  # a zero-length chain is malformed: worst possible
            return worst
        sorted_lengths = sorted(lengths) if need_sorted else None

        key = []
        for kind, func in metrics:
            if kind == _LENGTHS:
                key.append(func(lengths))
            elif kind == _SORTED_LENGTHS:
                key.append(func(sorted_lengths))
            else:  # _CHAINS
                key.append(func(chains))
        return tuple(key)

    quality.__name__ = "quality[" + ",".join(names) + "]"
    quality.__doc__ = (
        "Quality key (smaller better) for criteria, in priority order: "
        + ", ".join(names)
        + ". Empty embedding (or one with an empty chain) -> +inf per criterion."
    )
    return quality
