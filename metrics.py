import numpy as np
from scipy.linalg import lu, qr, expm, fractional_matrix_power
from scipy.sparse.linalg import eigsh
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.isomorphism import GraphMatcher

# Para importar la función communicability_exp
from networkx.algorithms.communicability_alg import communicability_exp

# Para importar la función communicability_betweenness_centrality
from networkx.algorithms.centrality.subgraph_alg import communicability_betweenness_centrality

import pandas as pd
from scipy.linalg import eigh
# from scipy.sparse.csgraph import laplacian


#------------------------------------------
# Metricas de reducción de grafos
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian


def get_graph_from_laplacian(L):
    H = nx.Graph()
    n = L.shape[0]
    H.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1,n):
            if (L[i,j] != 0):
                H.add_edge(i, j)

    # print("Adjacency list of the graph:")
    # for node in sorted(H.nodes()):
    #     print(f"{node}: {sorted(list(H.adj[node]))}")
    
    # print("Nodes of the graph:")
    # print(sorted(H.nodes()))

    return H

def analyze_spectral_properties(graph, reduced_graph):

    """
    Calculate and compare the spectral properties between two graphs.
    
    Parameters:
    - graph: numpy.ndarray, adjacency matrix of the original graph.
    - reduced_graph: numpy.ndarray, adjacency matrix of the reduced graph.
    - k: int, number of top eigenvalues to consider.
    
    Returns:
    - metrics: spectral ratios, eigenratios, algebraic connectivity, and spectral gaps.
    """
    # print("Begin analyze_spectral_properties")
    # if isinstance(graph, nx.Graph):
    #     print("Nodos y atributos de graph:", graph.nodes(data=True))
    #     print("Aristas y atributos de graph:", graph.edges(data=True))
    #     print("number of nodes in graph:", graph.number_of_nodes())
    #     print("number of edges in graph:", graph.number_of_edges())

    # if isinstance(reduced_graph, nx.Graph):
    #     print("Nodos y atributos de reduced_graph:", reduced_graph.nodes(data=True))
    #     print("Aristas y atributos de reduced_graph:", reduced_graph.edges(data=True))
    #     print("number of nodes in reduced graph:", reduced_graph.number_of_nodes())
    #     print("number of edges in reduced graph:", reduced_graph.number_of_edges())
    
    # Convert NetworkX graphs to adjacency matrices if they are not already numpy arrays
    if isinstance(graph, nx.Graph):
        graph = nx.to_numpy_array(graph)
    if isinstance(reduced_graph, nx.Graph):
        reduced_graph = nx.to_numpy_array(reduced_graph)

    G = nx.from_numpy_array(graph, edge_attr=None)
    G_r = nx.from_numpy_array(reduced_graph, edge_attr=None)
    L = nx.laplacian_matrix(G).toarray()
    L_prime = nx.laplacian_matrix(G_r).toarray()

    # Calculate the adjacency matrices of both graphs
    A = nx.to_numpy_array(G)
    A_prime = nx.to_numpy_array(G_r)

    # Calculate the eigenvalues of both adjacency matrices
    eigenvalues_A = np.linalg.eigvals(A)
    eigenvalues_A_prime = np.linalg.eigvals(A_prime)

    # Sort the eigenvalues in ascending order
    eigenvalues_A = np.sort(eigenvalues_A)
    eigenvalues_A_prime = np.sort(eigenvalues_A_prime)

    # Calculate the eigenvalues of both Laplacian matrices
    eigenvalues_L = eigh(L, eigvals_only=True)
    # print("Eigenvalues of the Laplacian matrix of the original graph:", eigenvalues_L)

    eigenvalues_L_prime = eigh(L_prime, eigvals_only=True)
    # print("Eigenvalues of the Laplacian matrix of the reduced graph:", eigenvalues_L_prime)

    # Sort the eigenvalues in ascending order
    eigenvalues_L = np.sort(eigenvalues_L)
    eigenvalues_L_prime = np.sort(eigenvalues_L_prime)
  

    # Calculate the spectral ratio (largest eigenvalue / second smallest eigenvalue ratio)
    larget_eigenvalue_L = eigenvalues_L[-1]
    secound_smallest_eigenvalue_L = eigenvalues_L[1]

    larget_eigenvalue_L_prime = eigenvalues_L_prime[-1]
    secound_smallest_eigenvalue_L_prime = eigenvalues_L_prime[1]

    spectral_ratio_L = larget_eigenvalue_L / secound_smallest_eigenvalue_L if len(eigenvalues_L) > 1 else 0
    spectral_ratio_L_prime = larget_eigenvalue_L_prime / secound_smallest_eigenvalue_L_prime if len(eigenvalues_L_prime) > 1 and secound_smallest_eigenvalue_L_prime != 0 else 0

    # Calculate the eigenratio (smallest non-zero to largest eigenvalue ratio)
    # Filter out very small eigenvalues close to zero
    tolerance = 1e-10  # Adjust tolerance to filter out very small eigenvalues close to zero
    # tolerance = 1e-50  # Remove redundant tolerance assignment
    non_zero_eigenvalues_L = eigenvalues_L[eigenvalues_L > tolerance]
    non_zero_eigenvalues_L_prime = eigenvalues_L_prime[eigenvalues_L_prime > tolerance]
    smallestNoZero = non_zero_eigenvalues_L[0]
    larget_eigenvalue_L = non_zero_eigenvalues_L[-1]
    smallestNoZero_prime = non_zero_eigenvalues_L_prime[0]
    larget_eigenvalue_L_prime = non_zero_eigenvalues_L_prime[-1]

    eigenratio_L = smallestNoZero / larget_eigenvalue_L
    eigenratio_L_prime = smallestNoZero_prime / larget_eigenvalue_L_prime

    # Calculate algebraic connectivity
    algebraic_connectivity_L = smallestNoZero
    algebraic_connectivity_L_prime = smallestNoZero_prime

    # Calculate the spectral gap in matriz A (difference between first and second largest eigenvalue)
    spectral_gap_A = eigenvalues_A[-1] - eigenvalues_A[-2] if len(eigenvalues_A) > 1 else 0
    spectral_gap_A_prime = eigenvalues_A_prime[-1] - eigenvalues_A_prime[-2] if len(eigenvalues_A_prime) > 1 else 0

    # Compile metrics with 4 decimal places and remove complex part if present
    metrics = {
        'Spectral Ratio (Original)': round(spectral_ratio_L.real, 4),
        'Spectral Ratio (Reduced)': round(spectral_ratio_L_prime.real, 4),
        'Eigenratio (Original)': round(eigenratio_L.real, 4),
        'Eigenratio (Reduced)': round(eigenratio_L_prime.real, 4),
        'Spectral Gap (Original)': round(spectral_gap_A.real, 4),
        'Spectral Gap (Reduced)': round(spectral_gap_A_prime.real, 4),
        'Algebraic Connectivity (Original)': round(algebraic_connectivity_L.real, 4),
        'Algebraic Connectivity (Reduced)': round(algebraic_connectivity_L_prime.real, 4),
        'Number of Nodes (Original)': graph.shape[0],
        'Number of Nodes (Reduced)': reduced_graph.shape[0],
        'Number of Edges (Original)': np.count_nonzero(graph) // 2,
        'Number of Edges (Reduced)': np.count_nonzero(reduced_graph) // 2
    }

    return metrics

def save_metrics_to_excel(metrics_list, filename):
    data = []
    for metrics in metrics_list:
        network_name = metrics['Network']
        method = metrics['Method']
        data.append([network_name, method, 'Spectral Ratio', metrics['Spectral Ratio (Original)'], metrics['Spectral Ratio (Reduced)']])
        data.append([network_name, method, 'Eigenratio', metrics['Eigenratio (Original)'], metrics['Eigenratio (Reduced)']])
        data.append([network_name, method, 'Spectral Gap', metrics['Spectral Gap (Original)'], metrics['Spectral Gap (Reduced)']])
        data.append([network_name, method, 'Algebraic Connectivity', metrics['Algebraic Connectivity (Original)'], metrics['Algebraic Connectivity (Reduced)']])
        data.append([network_name, method, 'Number of Nodes', metrics['Number of Nodes (Original)'], metrics['Number of Nodes (Reduced)']])
        data.append([network_name, method, 'Number of Edges', metrics['Number of Edges (Original)'], metrics['Number of Edges (Reduced)']])

    df = pd.DataFrame(data, columns=['Network', 'Method', 'Metric', 'Original', 'Reduced'])
    df.to_excel(filename, index=False)
