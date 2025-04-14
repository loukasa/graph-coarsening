import networkx as nx
from pygsp import graphs

import graph_coarsening.graph_utils as graph_utils
import graph_coarsening.coarsening_utils as coarsening_utils
import graph_coarsening.graph_lib as graph_lib
from metrics import *

# Función para agregar pesos a los grafos
def apply_weighting(graph, method='betweenness'):
    if method == 'betweenness':
        add_edge_betweenness_weights(graph)
    elif method == 'common_neighbors':
        add_common_neighbors_weights(graph)
    elif method == 'reciprocal_distance':
        add_reciprocal_distance_weights(graph)
    elif method == 'pagerank':
        add_edge_pagerank_weights(graph)
    else:
        raise ValueError("Método no reconocido. Usa 'betweenness', 'common_neighbors', 'reciprocal_distance' o 'pagerank'.")

# Métodos de ponderación

def add_edge_betweenness_weights(graph):
    """Agrega pesos basados en la centralidad de intermediación de las aristas."""
    betweenness = nx.edge_betweenness_centrality(graph)
    nx.set_edge_attributes(graph, betweenness, "weight")

def add_common_neighbors_weights(graph):
    """Agrega pesos basados en el número de vecinos comunes entre nodos."""
    for u, v in graph.edges():
        common_neighbors = len(list(nx.common_neighbors(graph, u, v)))
        graph[u][v]["weight"] = common_neighbors

def add_reciprocal_distance_weights(graph):
    """Agrega pesos basados en la distancia recíproca entre nodos (1/distancia)."""
    for u, v in graph.edges():
        try:
            distance = nx.shortest_path_length(graph, source=u, target=v)
            graph[u][v]["weight"] = 1 / distance if distance > 0 else 0
        except nx.NetworkXNoPath:
            graph[u][v]["weight"] = 0

def add_edge_pagerank_weights(graph):
    """Agrega pesos basados en el PageRank de los nodos extremos de cada arista."""
    pagerank = nx.pagerank(graph)
    for u, v in graph.edges():
        graph[u][v]["weight"] = pagerank[u] + pagerank[v]

# Parámetros globales
r = 0.6  # coarsening ratio
methods = [
    "variation_neighborhoods",
    "variation_edges",
    "variation_cliques",
    "heavy_edge",
    "algebraic_JC",
    "affinity_GS",
    "kron",
]
weighting_method = "betweenness"  # Cambiar a 'common_neighbors', 'reciprocal_distance', o 'pagerank'

# Diccionario para almacenar resultados
results = {}

# Evaluar cada grafo académico
graph_names = ["karate_club", "dolphins", "political_books", "football"]

for graph_name in graph_names:
    print(f"\nEvaluando grafo: {graph_name}")

    # Cargar el grafo utilizando realAcademic
    G = graph_lib.realAcademic(N=-1, graph_name=graph_name, connected=True)

    # Verificar si el grafo es ponderado
    if not nx.is_weighted(nx.from_scipy_sparse_array(G.W)):
        print(f"  - {graph_name} no es ponderado. Aplicando ponderación '{weighting_method}'.")
        nx_graph = nx.from_scipy_sparse_array(G.W)
        apply_weighting(nx_graph, method=weighting_method)
        W = nx.to_scipy_sparse_array(nx_graph, weight="weight", format="csr")
        G = graphs.Graph(W)  # Reconstruir el objeto PyGSP Graph con los pesos aplicados

    # Calcular base espectral para operaciones avanzadas
    G.compute_fourier_basis()

    # Resultados para cada método en el grafo actual
    results[graph_name] = {}

    for method in methods:
        print(f"  - Método: {method}")

        # Aplicar coarsening
        C, Gc, Call, Gall = coarsening_utils.coarsen(G, r=r, method=method)

        # Evaluar calidad del coarsening
        metrics = coarsening_utils.coarsening_quality(G, C)
        results[graph_name][method] = {
            "original_nodes": G.N,
            "reduced_nodes": Gc.N,
            "reduction_ratio": metrics["r"],
            "error_eigenvalue": metrics["error_eigenvalue"],
            "error_subspace": metrics["error_subspace"],
            "error_sintheta": metrics["error_sintheta"],
        }

        # Graficar el coarsening
        fig = coarsening_utils.plot_coarsening(Gall, Call, title=f"{graph_name} | {method}", size=2)
        fig.show()

# Mostrar resultados resumidos
for graph_name, metrics in results.items():
    print(f"\nResultados para el grafo: {graph_name}")
    for method, data in metrics.items():
        print(f"  Método: {method}")
        print(f"    Nodos Originales: {data['original_nodes']}")
        print(f"    Nodos Reducidos: {data['reduced_nodes']}")
        print(f"    Proporción de Reducción: {data['reduction_ratio']}")
        print(f"    Error Eigenvalores: {data['error_eigenvalue'][:5]}...")  # Solo primeros valores