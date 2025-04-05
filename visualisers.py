import matplotlib.pyplot as plt
import networkx as nx


def visualise_communities(graph: nx.MultiDiGraph, communities: list[set[int]]):
    colour_indices = {}
    for i, community in enumerate(communities):
        colour_indices.update((node, i) for node in community)

    colour_map = [plt.cm.viridis(colour_indices[node] / len(communities)) for node in range(len(graph))]
    pos = nx.spring_layout(graph, seed=42)

    nx.draw(graph, pos, node_color=colour_map, node_size=10, width=0.15)