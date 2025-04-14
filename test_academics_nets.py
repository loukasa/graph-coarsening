import glob
import os
import pprint
import pandas as pd
import matplotlib.pyplot as plt
from graph_coarsening import coarsening_utils
from graph_coarsening import graph_lib

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

def setPropertiesToNodes(graph, Call, G):
    print("Setting properties to nodes")
    for node in graph.nodes():
        graph.nodes[node]['sizeSuperNode'] = 0
        graph.nodes[node]['nodesInSuperNode'] = []

    node_mapping = {i: [i] for i in range(len(Call[0].indices))}
    
    for level in range(len(Call)):
        new_node_mapping = {}
        for vi in range(len(Call[level].indices)):
            index = Call[level].indices[vi].real
            if index not in new_node_mapping:
                new_node_mapping[index] = []
            new_node_mapping[index].extend(node_mapping[vi])
        
        node_mapping = new_node_mapping

    for node, members in node_mapping.items():
        graph.nodes[node]['sizeSuperNode'] = len(members)
        graph.nodes[node]['nodesInSuperNode'] = members

    # print("Nodes and properties:", graph.nodes(data=True))

    # Create a new reduced graph with updated vertex indices
    new_graph = nx.Graph()

    node_mapping = {}
    for node in graph.nodes():
        members = graph.nodes[node]['nodesInSuperNode']
        
        if len(members) > 1:
            # Find the node with the highest degree in the induced subgraph
            subgraph = nx.Graph(G.W).subgraph(members)
            max_degree_node = max(subgraph.degree, key=lambda x: x[1])[0]
            new_graph.add_node(max_degree_node)
            new_graph.nodes[max_degree_node]['label'] = max_degree_node
            new_graph.nodes[max_degree_node]['sizeSuperNode'] = len(members)
            new_graph.nodes[max_degree_node]['nodesInSuperNode'] = members
            node_mapping[node] = max_degree_node
        elif len(members) == 1:
            new_graph.add_node(members[0])
            new_graph.nodes[members[0]]['label'] = members[0]
            new_graph.nodes[members[0]]['sizeSuperNode'] = 0
            new_graph.nodes[members[0]]['nodesInSuperNode'] = []
            node_mapping[node] = members[0]

    # Add edges to the new graph using the node mapping
    for u, v in graph.edges():
        new_u = node_mapping[u]
        new_v = node_mapping[v]
        new_graph.add_edge(new_u, new_v)

    # print("New graph with updated vertex indices:", new_graph.nodes(data=True))
    
    # Convert 'nodesInSuperNode' property to a list format
    for node in new_graph.nodes():
        new_graph.nodes[node]['nodesInSuperNode'] = str(new_graph.nodes[node]['nodesInSuperNode'])

    return graph, new_graph

def testAcademicsNets():
    """
    Test the academics networks
    """

    # Parámetros globales
    r = 0.6  # coarsening ratio
    methods = [
        # "variation_neighborhoods",
        # "variation_edges",
        # "variation_cliques",
        # "heavy_edge",
        "algebraic_JC",
        "affinity_GS",
        # "kron",
    ]

    # Evaluar cada grafo académico
    # graph_names = ["karate", "dolphins", "polbooks", "football"]
    graph_names = ["karate"]

    spectral_metrics_all = []

    for graph_name in graph_names:
        print(f"\nEvaluando grafo: {graph_name}")

        # Cargar el grafo utilizando realAcademic
        G = graph_lib.realAcademic(N=-1, graph_name=graph_name, connected=True)

        nx_graph = nx.from_scipy_sparse_array(G.W)

        # Calcular base espectral para operaciones avanzadas
        G.compute_fourier_basis()


        for method in methods:
            print(f"    - Método: {method}")
            run_i = 0
            for run_i in range(5):

                # Aplicar coarsening
                _, Gc, Call, _ = coarsening_utils.coarsen(G, r=r, method=method) # en Call está lo que necesito, propiedad indices
                # Gc es el grafo reducido, Call es la lista de coarsening
                nx_graph_H = nx.from_scipy_sparse_array(Gc.W)

                # acá tengo que crear las propiedades 'sizeSuperNode' y 'nodesInSuperNode' en nx_graph_H 
                _, G_reducido_update = setPropertiesToNodes(nx_graph_H, Call, G)

                # # Imprimir el grafo reducido con indexación y sus adyacentes
                # i = 0
                # print(f"Grafo reducido ({graph_name} - {method}):")
                # for node in G_reducido_update.nodes():
                #     neighbors = list(G_reducido_update.neighbors(node))
                #     print(f"item {i} -> Node {node}: {neighbors}")
                #     i+=1

                # Guardar los grafos en formato GML para visualización en Gephi
                # nx.write_gml(G_reducido_update, f'/home/darian/graph-coarsening/final_result/{graph_name}_{method}_reduced.gml')

                # Convertir los grafos a matrices de adyacencia
                adj_matrix_G = nx.to_numpy_array(nx_graph, weight=None)
                adj_matrix_H = nx.to_numpy_array(nx_graph_H, weight=None)

                # Guardar la matriz de adyacencia adj_matrix_H en un archivo .txt iterando fila por fila
                adj_matrix_H_filename = os.path.join('/home/darian/graph-coarsening/final_result/', f'{graph_name}_{method}_{run_i}_reduced_adj_matrix.txt')
                with open(adj_matrix_H_filename, 'w') as f:
                    for row in adj_matrix_H:
                        f.write(' '.join(map(str, row.astype(int))) + '\n')
                print(f"Matriz de adyacencia guardada en: {adj_matrix_H_filename}")

                # # Imprimir la matriz de adyacencia adj_matrix_H
                # print("Matriz de adyacencia (adj_matrix_H):")
                # print(adj_matrix_H)
                
                # Calcular las métricas espectrales para el grafo original y el grafo reducido
                spectral_metrics = analyze_spectral_properties(adj_matrix_G, adj_matrix_H)

                spectral_metrics['Network'] = graph_name
                spectral_metrics['Method'] = method

                spectral_metrics_all.append(spectral_metrics)

                save_metrics_to_excel(spectral_metrics_all, f'/home/darian/graph-coarsening/final_result/metrics_results_{method}_{run_i}.xlsx')
                spectral_metrics_all = []

if __name__ == "__main__":
    print("Starting...")

    testAcademicsNets()
    
    print("Done!")