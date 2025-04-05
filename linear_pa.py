import networkx as nx
import random


class ScaleFreeDiGraph:
    '''
    Scale free digraph, evolving according to alpha-, beta-, gamma- schemas
    '''
    def __init__(self, init_graph: nx.MultiDiGraph | None = None) -> None:
        '''
        :param init_graph: initial graph to start with
        '''
        # If initial_graph is None, start with self-loop
        self.graph = init_graph if init_graph is not None else nx.MultiDiGraph([(0, 0),])
        
        self._indeg_nodes = sum(([node] * indeg for node, indeg in self.graph.in_degree()), [])
        self._outdeg_nodes = sum(([node] * outdeg for node, outdeg in self.graph.out_degree()), [])
        self._nodes = list(self.graph)
        self._next_node = len(self._nodes)

    def get_graph(self) -> nx.MultiDiGraph:
        '''
        Returns current graph.
        '''
        return self.graph

    def __choose_node(self, deg_nodes: list, nodes: list, delta: float) -> int:
        if delta > 0:
            delta_sum = delta * len(nodes)
            delta_prob = delta_sum / (len(deg_nodes) + delta_sum)
            if random.random() < delta_prob:
                return random.choice(nodes)
        return random.choice(deg_nodes)

    def grow(self, steps: int = 1, alpha: float = 0.41, beta: float = 0.54,
             gamma: float = 0.05, delta_in: float = 0.2, delta_out: float = 0) -> None:
        '''
        Makes specified number of evolution steps
        :param steps: number of evolution steps
        :param alpha: probability for adding new node connected to existing node,
                      chosen according to in-degree distribution
        :param beta: probability for adding new edge between existing nodes,
                     which are chosen according to in- and out-degree distributions
        :param gamma: probability for adding new node connected to existing node,
                      chosen according to out-degree distribution
        :param delta_in: bias for choosing nodes from in-degree distribution
        :param delta_out: bias for choosing nodes from out-degree distribution
        '''
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        if beta < 0:
            raise ValueError("beta must be >= 0")
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
    
        if abs(alpha + beta + gamma - 1.0) >= 1e-9:
            raise ValueError("alpha+beta+gamma must equal 1")
    
        if delta_in < 0:
            raise ValueError("delta_in must be >= 0.")
        if delta_out < 0:
            raise ValueError("delta_out must be >= 0.")

        for step in range(steps):
            coin = random.random()

            if coin < alpha:
                node_from = self._next_node
                node_to = self.__choose_node(self._indeg_nodes, self._nodes, delta_in)
                
                self._nodes.append(self._next_node)
                self._next_node += 1
            elif coin < alpha + beta:
                node_from = self.__choose_node(self._outdeg_nodes, self._nodes, delta_out)
                node_to = self.__choose_node(self._indeg_nodes, self._nodes, delta_in)
            else:
                node_from = self.__choose_node(self._outdeg_nodes, self._nodes, delta_out)
                node_to = self._next_node

                self._nodes.append(self._next_node)
                self._next_node += 1

            self.graph.add_edge(node_from, node_to)
            self._indeg_nodes.append(node_to)
            self._outdeg_nodes.append(node_from)