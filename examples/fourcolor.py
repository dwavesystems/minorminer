def ParseBlocks(G, blocks, block_good, eblock_good):
    """
    Extract the blocks from a graph, and returns a
    block-quotient graph according to the acceptability
    functions block_good and eblock_good

    Inputs:
        G: a networkx graph
        blocks: a partition of G.nodes()
        block_good: a function
                f(blocks[i], G)
            which is true if the node `i` should be
            represented in the quotient, and false otherwise
        eblock_good: a function
                f(blocks[i], blocks[j], G)
            which is true if the edge `(i,j)` should be
            represented in the quotient, and false otherwise

    """
    from networkx import Graph

    blockid = {}
    for i,b in enumerate(blocks):
        if not block_good(b, G):
            continue
        for q in b:
            if q in blockid:
                raise(RuntimeError, "two blocks overlap")
            blockid[q] = i

    BG = Graph()
    for q,u in blockid.items():
        for p in G[q]:
            if p not in blockid:
                continue
            v = blockid[p]
            if BG.has_edge(u,v):
                continue
            if eblock_good(blocks[u], blocks[v], G):
                BG.add_edge(u,v)

    return BG


def FullBlock(block, G):
    """
    true if every node in `block` is present in `G`
    """
    return all(G.has_node(x) for x in block)

def BiCliqueBlocks(block0, block1, G):
    """
    true if every edge `(u,v)` is present in `G`,
        for all `u` in block0 and
                `v` in block1
    """
    return all(G.has_edge(x,y) for x in block0 for y in block1)

def MatchBlocks(block0, block1, G):
    """
    true if every edge `(block0[i], block1[i])` is present in `G`
        for all `i`
    """
    return all(G.has_edge(x,y) for x,y in zip(block0, block1))

def ChimeraBlockEdge(block0, block1, G):
    """
    Chimera block edge quotient:
        if block0 and block1 are the same orientation
            in adjacent tiles, check with MatchBlocks
        if block0 and block1 are opposite orientation
            in the same tile, check with BlCliqueBlocks
    """
    if block0[0][2] == block1[0][2]:
        return MatchBlocks(block0, block1, G)
    else:
        return BiCliqueBlocks(block0, block1, G)

def ChimeraBlocks(M=16,N=16,L=4):
    """
    Generator for blocks for a chimera block quotient
    """
    for x in xrange(M):
        for y in xrange(N):
            for u in (0,1):
                yield tuple((x,y,u,k) for k in xrange(L))

def FourColor(source_graph,target_graph,M=16,N=16,L=4):
    """
    Produce an embedding in target_graph suitable to
    check if source_graph is 4-colorable.  More generally,
    if target_graph is a (M,N,L) Chimera subgraph, the
    test is for L-colorability.  This depends heavily upon
    the Chimera structure

    Inputs:
        source_graph, target_graph: networkx graphs
        M,N,L: integers defining the base chimera topology

    Outputs:
        emb: a dictionary mapping (v,i)

    """


    from random import sample
    from networkx import Graph
    from minorminer import find_embedding
    blocks = list(ChimeraBlocks(M,N,L))

    BG = ParseBlocks(target_graph, blocks, FullBlock, ChimeraBlockEdge)

    ublocks = {block:(block[0][2],i) for (i,block) in enumerate(blocks) if BG.has_node(i) }
    source_e = list(source_graph.edges())
    source_n = {x for e in source_e for x in e}
    fabric_e = list(BG.edges())

    #Construct the hints:
    # Goal: each source node must be connected to one horizontal block and one
    #       vertical block (by Chimera structure, each source node will
    #       contain a full (horizontal and vertical) unit cell
    # Construction:
    #   0. for each source node `z`, construct two dummy nodes (z,0) and (z,1)
    #     in both the source and target graphs used by the embedder
    #   1. fix the embedding `(z,u) -> (z,u)` for all dummy nodes
    #   2. for each target block `i` with orientation `u`, and for each source
    #     node `z`, add the target edge `((z,u), i)`
    #   3. for each source node `z` and each orientation `u`, add the source
    #     edge `((z,u), z)`

    fix_chains = {}
    for z in source_n:
        for u in (0,1):
            source_e.append(((z,u), z))
            fix_chains[z,u] = [(z,u)]
        for u,i in ublocks.values():
            fabric_e.append(((z,u), i))

    #first, build up a pool of embeddings
    embeddings = []
    while len(embeddings) < 10:
        emb = find_embedding(source_e, fabric_e, fixed_chains = fix_chains, verbose=1,
                            chainlength_patience=10)
        emb = {v:c for v,c in emb.items() if v not in fix_chains}
        if emb:
            embeddings.append(emb)


    #now, pick the best, and pass it through a few chainlength improvements
    emb = min(embeddings, key=lambda emb:sorted((len(c) for c in emb.values()), reverse=True))
    for _ in range(10):
        emb = find_embedding(source_e, fabric_e, fixed_chains = fix_chains, verbose=1,
                                initial_chains=emb,
                                chainlength_patience=100, skip_initialization=True)
        emb = {v:c for v,c in emb.items() if v not in fix_chains}

    #finally, translate the block-embedding to a qubit-embedding
    newemb = {}
    for v in source_n:
        for k in range(L):
            newemb[v,k] = [blocks[i][k] for i in emb[v]]

    return newemb


#first, we construct a Chimera graph
import dwave_networkx, networkx
from dwave_networkx.generators import chimera
G = dwave_networkx.generators.chimera.chimera_graph(16)
labs = {i:d['chimera_index'] for i,d in G.nodes(data=True)}
unlab = {d:i for i,d in labs.items()}
H = networkx.relabel_nodes(G,labs)

#Next, we delete some nodes and edges to simulate a low-yield processor
from random import sample
bad_q = sample(H.nodes(), 50)
for q in bad_q:
    H.remove_node(q)
bad_e = sample(H.edges(), 50)
for e in bad_e:
    H.remove_edge(*e)

#Now we take a graph to be colored
graph = networkx.generators.complete_multipartite_graph(2,3,4,2)

#we embed it using FourColor,
emb = FourColor(graph, H, 16,16,4)
#and then translate back to integer indices
newemb = {v:[unlab[q] for q in c] for v, c in emb.items()}
print(newemb)