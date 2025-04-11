from embeddings import Embedding
from gpdc import DiscreteGPDC

import numpy as np
import networkx as nx
import typing as tp

from sklearn.preprocessing import StandardScaler


class GPDCAnomalies:
    '''
    Anomaly detection on evolving random graphs using GPDC
    '''
    def __init__(self, init_graph: nx.MultiDiGraph,
                 embedding_class: Embedding,
                 tail_size_ratio: float = 0.01, alpha: float = 0.05) -> None:
        '''
        :param init_graph: initial graph (nodes are expected to be integers 0, 1, ..., len(init_graph) - 1)
        '''
        self.max_node = len(init_graph)

        self.embeddings = embedding_class(init_graph)
        self.tail_size_ratio = tail_size_ratio
        self.alpha = alpha

    def update(self, snapshot: nx.MultiDiGraph) -> np.ndarray:
        '''
        Returns anomalies in the new snapshot
        :param snapshot: new snapshot of a graph
        '''
        old_embeddings = self.embeddings.to_numpy()
        self.embeddings.update(snapshot)
        new_embeddings = self.embeddings.to_numpy()

        scaler = StandardScaler()
        old_embeddings = scaler.fit_transform(old_embeddings)
        new_embeddings = scaler.transform(new_embeddings)

        model = DiscreteGPDC(tail_size_ratio=self.tail_size_ratio, alpha=self.alpha)
        model.fit(old_embeddings)
        test_results = model.predict(new_embeddings)

        anomalies = np.where(test_results == -1)[0]

        self.max_node = len(snapshot)

        return anomalies