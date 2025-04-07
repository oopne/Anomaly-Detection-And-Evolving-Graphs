import estimators
from embeddings import Embedding
from gpdc import DiscreteGPDC

import numpy as np
import networkx as nx
import typing as tp

from sklearn.preprocessing import StandardScaler


class GPDCCommunities:
    '''
    Community search on evolving random graphs using GPDC
    '''
    def __init__(self, init_graph: nx.MultiDiGraph,
                 embedding_class: Embedding,
                 **gpdc_kwargs: tp.Any) -> None:
        '''
        Initialises base partition of an initial graph.
        :param init_graph: initial graph (nodes are expected to be integers 0, 1, ..., len(init_graph) - 1)
        '''
        self.max_node = len(init_graph)

        self.base_community = list(range(self.max_node))
        self.other_communities: list[set[int]] = []

        self.embeddings = embedding_class(init_graph)
        self.gpdc_kwargs = gpdc_kwargs

    def update(self, snapshot: nx.MultiDiGraph) -> None:
        '''
        Updates the partition.
        :param snapshot: new snapshot of a graph
        '''
        self.embeddings.update(snapshot)
        embeddings = self.embeddings.to_numpy()

        test_nodes = np.arange(self.max_node, len(snapshot))

        scaler = StandardScaler()
        train_embeddings = embeddings[self.base_community]
        test_embeddings = embeddings[test_nodes]

        # train_embeddings = scaler.fit_transform(train_embeddings)
        # test_embeddings = scaler.transform(test_embeddings)
        model = DiscreteGPDC(**self.gpdc_kwargs).fit(train_embeddings)
        
        test_results = model.predict(test_embeddings)

        new_normals = test_nodes[test_results == 1]
        new_community = test_nodes[test_results == -1]

        self.base_community.extend(new_normals)
        self.other_communities.append(set(new_community))

        self.max_node = len(snapshot)

    def get_communities(self) -> list[set[int]]:
        return [set(self.base_community)] + self.other_communities