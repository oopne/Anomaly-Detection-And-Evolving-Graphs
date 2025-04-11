import networkx as nx
import numpy as np

from abc import ABC, abstractmethod


class Embedding(ABC):
    @abstractmethod
    def update(self, snapshot: nx.MultiDiGraph) -> None:
        pass

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        pass


class PageRank(Embedding):
    '''
    Maintains PageRank's for evolving graph
    '''
    def __init__(self, init_graph: nx.MultiDiGraph, alpha: float = 0.85) -> None:
        '''
        Initialises PageRank's for nodes of the initial graph.
        :param init_graph: initial graph
        :param alpha: damping factor for PageRank
        '''
        self.max_node = len(init_graph)
        self.alpha = alpha
        self.pageranks = nx.pagerank(init_graph, alpha=self.alpha)

    def update(self, snapshot: nx.MultiDiGraph) -> None:
        '''
        Updates nodes attributes for the new snapshot of a graph.
        :param snapshot: new snapshot of a graph
        '''
        for v in range(self.max_node, len(snapshot)):
            self.pageranks[v] = (1 - self.alpha) / len(snapshot)

        self.max_node = len(snapshot)
        self.pageranks = nx.pagerank(snapshot, alpha=self.alpha, nstart=self.pageranks)

    def to_numpy(self) -> np.ndarray:
        '''
        Constructs numpy array of vectors with nodes attributes
        '''
        return np.array([[self.pageranks[v]] for v in range(self.max_node)])


def _add_edge_with_weight(weighted_digraph: nx.DiGraph, u: int, v: int) -> None:
    if not weighted_digraph.has_edge(u, v):
        weighted_digraph.add_edge(u, v, weight=0)
    weighted_digraph.edges[(u, v)]['weight'] += 1


def _convert_to_weighted_digraph(multidigraph: nx.MultiDiGraph) -> nx.DiGraph:
    weighted_digraph = nx.DiGraph()
    for u, v in multidigraph.edges():
        _add_edge_with_weight(weighted_digraph, u, v)

    return weighted_digraph


class PageRankClustering(Embedding):
    '''
    Provides 2D embeddings for evolving graph's nodes, consisting of pageranks and clustering coefficients
    '''
    def __init__(self, init_graph: nx.MultiDiGraph, alpha: float = 0.85) -> None:
        '''
        Initialises nodes attributes for initial graph.
        :param init_graph: initial graph
        :param alpha: damping factor for PageRank
        '''
        self.graph = init_graph.copy()
        self.weighted_digraph = _convert_to_weighted_digraph(self.graph)

        self.alpha = alpha
        self.pageranks = nx.pagerank(self.graph, alpha=self.alpha)
        self.clusterings = nx.clustering(self.weighted_digraph)

    def update(self, snapshot: nx.MultiDiGraph) -> None:
        '''
        Updates nodes attributes for the new snapshot of a graph.
        :param snapshot: new snapshot of a graph
        '''
        nodes_to_update: set[int] = set()
        for u, v, idx in snapshot.edges:
            if self.graph.has_edge(u, v, idx):
                continue

            _add_edge_with_weight(self.weighted_digraph, u, v)

            nodes_to_update |= set(snapshot.predecessors(u)) |\
                               set(snapshot.predecessors(v)) |\
                               set(snapshot.successors(u)) |\
                               set(snapshot.successors(v))

            for w in [u, v]:
                if not self.graph.has_node(w):
                    self.pageranks[w] = (1 - self.alpha) / len(snapshot)

        self.graph = snapshot.copy()
        
        self.pageranks = nx.pagerank(self.graph, alpha=self.alpha, nstart=self.pageranks)
        self.clusterings |= nx.clustering(self.weighted_digraph, nodes_to_update)

    def to_numpy(self) -> np.ndarray:
        '''
        Constructs numpy array of vectors with nodes attributes
        '''
        return np.array([[self.pageranks[v] * len(self.graph), self.clusterings[v]] for v in range(len(self.graph))])


class PageRankInDegree(Embedding):
    '''
    Provides 2D embeddings for evolving graph's nodes, consisting of pageranks and in-degrees
    '''
    def __init__(self, init_graph: nx.MultiDiGraph, alpha: float = 0.85) -> None:
        '''
        Initialises PageRanks and in-degrees for nodes of the initial graph.
        :param init_graph: initial graph
        :param alpha: damping factor for PageRank
        '''
        self.max_node = len(init_graph)
        self.alpha = alpha
        self.pageranks = nx.pagerank(init_graph, alpha=self.alpha)
        self.in_degrees = init_graph.in_degree()

    def update(self, snapshot: nx.MultiDiGraph) -> None:
        '''
        Updates nodes attributes for the new snapshot of a graph.
        :param snapshot: new snapshot of a graph
        '''
        for v in range(self.max_node, len(snapshot)):
            self.pageranks[v] = (1 - self.alpha) / len(snapshot)

        self.max_node = len(snapshot)
        self.pageranks = nx.pagerank(snapshot, alpha=self.alpha, nstart=self.pageranks)
        self.in_degrees = snapshot.in_degree()

    def to_numpy(self) -> np.ndarray:
        '''
        Constructs numpy array of vectors with nodes attributes
        '''
        return np.array([[self.pageranks[v] * self.max_node, self.in_degrees[v]] for v in range(self.max_node)])


class PageRankLogInDegree(Embedding):
    '''
    Provides 2D embeddings for evolving graph's nodes, consisting of pageranks and log of in-degrees
    '''
    def __init__(self, init_graph: nx.MultiDiGraph, alpha: float = 0.85) -> None:
        '''
        Initialises PageRanks and log of in-degrees for nodes of the initial graph.
        :param init_graph: initial graph
        :param alpha: damping factor for PageRank
        '''
        self.max_node = len(init_graph)
        self.alpha = alpha
        self.pageranks = nx.pagerank(init_graph, alpha=self.alpha)
        self.in_degrees = init_graph.in_degree()

    def update(self, snapshot: nx.MultiDiGraph) -> None:
        '''
        Updates nodes attributes for the new snapshot of a graph.
        :param snapshot: new snapshot of a graph
        '''
        for v in range(self.max_node, len(snapshot)):
            self.pageranks[v] = (1 - self.alpha) / len(snapshot)

        self.max_node = len(snapshot)
        self.pageranks = nx.pagerank(snapshot, alpha=self.alpha, nstart=self.pageranks)
        self.in_degrees = snapshot.in_degree()

    def to_numpy(self) -> np.ndarray:
        '''
        Constructs numpy array of vectors with nodes attributes
        '''
        return np.array([[self.pageranks[v] * self.max_node,
                          np.log(self.in_degrees[v] + 1)] for v in range(self.max_node)])


class PageRankMLM(Embedding):
    '''
    Provides 2D embeddings for evolving graph's nodes, consisting of pageranks and MLMs
    '''
    def __init__(self, init_graph: nx.MultiDiGraph, alpha: float = 0.85) -> None:
        '''
        Initialises PageRanks and MLMs for nodes of the initial graph.
        :param init_graph: initial graph
        :param alpha: damping factor for PageRank
        '''
        self.max_node = len(init_graph)
        self.alpha = alpha
        self.pageranks = nx.pagerank(init_graph, alpha=self.alpha)
        self.pageranks = {v: pr * self.max_node for v, pr in self.pageranks.items()}
        self.mlms = {v: self._mlm(init_graph, v) for v in range(self.max_node)}

    def update(self, snapshot: nx.MultiDiGraph) -> None:
        '''
        Updates nodes attributes for the new snapshot of a graph.
        :param snapshot: new snapshot of a graph
        '''
        for v in range(self.max_node, len(snapshot)):
            self.pageranks[v] = (1 - self.alpha) / len(snapshot)

        self.max_node = len(snapshot)
        self.pageranks = nx.pagerank(snapshot, alpha=self.alpha, nstart=self.pageranks)
        self.pageranks = {v: pr * self.max_node for v, pr in self.pageranks.items()}
        self.mlms = {v: self._mlm(snapshot, v) for v in range(self.max_node)}

    def _mlm(self, graph: nx.MultiDiGraph, v: int) -> float:
        neighbours = [self.alpha * self.pageranks[u] / graph.out_degree(u) for u in graph.predecessors(v)]
        return max(neighbours + [(1 - self.alpha)])

    def to_numpy(self) -> np.ndarray:
        '''
        Constructs numpy array of vectors with nodes attributes
        '''
        return np.array([[self.pageranks[v], self.mlms[v]] for v in range(self.max_node)])