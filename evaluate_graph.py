import networkx as nx
from pygsp import graphs

import graph_coarsening.graph_utils as graph_utils
import graph_coarsening.coarsening_utils as coarsening_utils
import graph_coarsening.graph_lib as graph_lib
from metrics import *
import pprint
import pandas as pd
import os


def plot_graph(G, title="Grafo"):
    """
    Función para graficar un grafo.

    Parámetros:
    - G: Un grafo de NetworkX.
    - title: Título del gráfico.
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)  # Posiciones de los nodos
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=15, font_color='black')
    plt.title(title)
    plt.show()
    plt.savefig(f'/home/darian/Graph-Reduction-Project/{title.replace(" ", "_")}.png')

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

def main():
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

    # Diccionario para almacenar resultados
    results = {}

    # Evaluar cada grafo académico
    graph_names = ["karate", "dolphins", "polbooks", "football"]
    spectral_metrics_all = []

    # Lista para almacenar todos los resultados
    all_rows = []

    for graph_name in graph_names:
        print(f"\nEvaluando grafo: {graph_name}")

        # Cargar el grafo utilizando realAcademic
        G = graph_lib.realAcademic(N=-1, graph_name=graph_name, connected=True)

        nx_graph = nx.from_scipy_sparse_array(G.W)
        # W = nx.to_scipy_sparse_array(nx_graph, weight="weight", format="csr")
        # G = graphs.Graph(W)  # Reconstruir el objeto PyGSP Graph con los pesos aplicados

        # Calcular base espectral para operaciones avanzadas
        G.compute_fourier_basis()

        spectral_metrics_all = []

        for method in methods:
            print(f"    - Método: {method}")

            # Aplicar coarsening
            C, Gc, Call, Gall = coarsening_utils.coarsen(G, r=r, method=method)

            nx_graph_H = nx.from_scipy_sparse_array(Gc.W)

            # Convertir los grafos a matrices de adyacencia
            adj_matrix_G = nx.to_numpy_array(nx_graph)
            adj_matrix_H = nx.to_numpy_array(nx_graph_H)

            # Evaluar calidad del coarsening
            # Calcular las métricas espectrales para el grafo original y el grafo reducido
            spectral_metrics = analyze_spectral_properties(adj_matrix_G, adj_matrix_H)
            
            spectral_metrics_all.append(spectral_metrics)

            # plot_graph(nx_graph_H, title=f"{graph_name}_reduced_{method}")    

        # Nombres de las métricas espectrales
        metrics_names = list(spectral_metrics_all[0].keys())
        x_labels = [method for method in methods]

        # Crear un diccionario para almacenar los valores de las métricas
        data = {}
        for method in methods:
            data[method] = {}
            for metric_name in metrics_names:
                base_metric_name = metric_name.split(' (')[0]
                if base_metric_name not in data[method]:
                    data[method][base_metric_name] = {'Original': [], 'Reduced': []}

        # Rellenar el diccionario con los valores de las métricas
        for j, method in enumerate(methods):
            for metric_name in metrics_names:
                base_metric_name = metric_name.split(' (')[0]
                if 'Original' in metric_name:
                    data[method][base_metric_name]['Original'] = float(spectral_metrics_all[j].get(metric_name, 0))
                elif 'Reduced' in metric_name:
                    data[method][base_metric_name]['Reduced'] = float(spectral_metrics_all[j].get(metric_name, 0))

        # Crear una lista de diccionarios para construir el DataFrame
        for method in methods:
            for metric_name in metrics_names:
                base_metric_name = metric_name.split(' (')[0]
                original_value = data[method][base_metric_name]['Original']
                reduced_value = data[method][base_metric_name]['Reduced']
                row = {
                    'Network': graph_name,
                    'Method': method,
                    'Metric': base_metric_name,
                    'Original': original_value,
                    'Reduced': reduced_value
                }
                if row not in all_rows:
                    all_rows.append(row)

    # Crear el DataFrame con todos los resultados
    df = pd.DataFrame(all_rows)

    # Mostrar el DataFrame
    print(df)

    # Guardar el DataFrame en un archivo Excel
    file_name = '/home/darian/Graph-Reduction-Project/coarseningMethods_all_networks.xlsx'
    df.to_excel(file_name, index=False)

def loadNetworks():
    folder_path = '/home/darian/graph-coarsening/All_Networks/'
    graph_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    print(f"Found {len(graph_files)} graph files in {folder_path}")

    for file_name in graph_files:
        file_path = os.path.join(folder_path, file_name)
        print(f"Loading graph from {file_path}")

        # Leer la matriz de adyacencia desde el archivo
        adj_matrix = pd.read_csv(file_path, header=None, delim_whitespace=True).values

        # Crear el grafo desde la matriz de adyacencia
        G = nx.from_numpy_array(adj_matrix)

        # Guardar el grafo en formato GML
        gml_file_path = os.path.join(folder_path, file_name.replace('.txt', '.gml'))
        nx.write_gml(G, gml_file_path)
        print(f"Graph saved to {gml_file_path}")
        print("Done")



if __name__ == "__main__":
    # main()
    loadNetworks()



