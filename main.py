
from fuction_reduction import *
from metrics import *

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import glob
import random
from TwiceRamanujan import *
from LeeSunSpectralSparsifier import *
from SpectralSparsificationbyEffectiveResistance import *


def create_indexed_graph(edges):
    """
    Crea un grafo de NetworkX con índices de nodos explícitos antes de añadir las aristas.
    
    Parámetros:
    - edges: Una lista de tuplas representando las aristas (v, w).
    
    Retorna:
    - G: Un grafo de NetworkX con nodos indexados.
    - node_indices: Un diccionario que asigna índices a los nodos.
    """
    G = nx.Graph()
    nodes = set()
    
    # Recopila todos los nodos únicos a partir de las aristas
    for v, w in edges:
        nodes.add(v)
        nodes.add(w)
    
    # Crea índices para los nodos
    node_indices = {node: idx for idx, node in enumerate(nodes)}
    
    # Añade nodos al grafo
    G.add_nodes_from(node_indices.values())
    
    # Añade aristas al grafo usando los índices
    indexed_edges = [(node_indices[v], node_indices[w]) for v, w in edges]
    G.add_edges_from(indexed_edges)
    
    return G, node_indices

def create_experiment_graphs():
    """
        Crea una lista de grafos para utilizar en experimentos.

        Los grafos corresponden a los utilizados en el artículo "Communicability cosine distance: similarity and symmetry in
        graphs/networks" de Ernesto Estrada.

        Retorna:
        - Una lista de grafos de NetworkX.
        """
    # Definimos las aristas de los grafos
    Fig_7a_edges = [(1, 3), (1, 8), (2, 3), (2, 4), (2, 5), (3, 7), (3, 8), (4, 6), (4, 7), (4, 8), (5, 6), (5, 7), (5, 8), (6, 7), (6, 8)]
    Fig_7b_edges = [(1, 4), (1, 5), (1, 7), (1, 8), (2, 3), (2, 6), (2, 7), (2, 8), (3, 4), (3, 5), (3, 6), (4, 6), (4, 7), (5, 6), (5, 8), (6, 7), (6, 8), (7, 8)]
    Fig_7c_edges = [(1, 7), (1, 8), (2, 3), (2, 4), (2, 5), (3, 6), (3, 7), (3, 8), (4, 6), (4, 7), (4, 8), (5, 6), (5, 7), (5, 8), (6, 8)]
    Fig_7d_edges = [(1, 7), (1, 8), (2, 5), (2, 6), (2, 7), (2, 8), (3, 5), (3, 6), (3, 7), (3, 8), (4, 5), (4, 6), (4, 7), (4, 8), (5, 8), (6, 8)]
    Fig_7e_edges = [(1, 3), (1, 6), (1, 7), (1, 8), (2, 3), (2, 6), (2, 7), (2, 8), (3, 4), (3, 5), (4, 5), (4, 6), (4, 7), (4, 8), (5, 6), (5, 7), (5, 8), (6, 7), (6, 8), (7, 8)]
    Fig_7f_edges = [(1, 8), (2, 8), (3, 8), (4, 8), (5, 7), (6, 7), (6, 8)]
    Fig_7g_edges = [(1, 6), (2, 3), (2, 7), (3, 8), (4, 5), (4, 8), (5, 7), (6, 7), (6, 8)]
    Fig_7h_edges = [(1, 3), (1, 4), (1, 5), (2, 6), (2, 7), (2, 8), (3, 4), (3, 5), (3, 8), (4, 6), (4, 7), (5, 6), (5, 7), (6, 8), (7, 8)]

    edges = [Fig_7a_edges, Fig_7b_edges, Fig_7c_edges, Fig_7d_edges, Fig_7e_edges, Fig_7f_edges, Fig_7g_edges, Fig_7h_edges]

    graphs = []
    for graph_edges in edges:
        G, _ = create_indexed_graph(graph_edges)
        graphs.append(G)

    return graphs

def run_experimentEqualNodesGraph():
    # Crear una lista de grafos para el experimento
    graphList = create_experiment_graphs()
    spectral_metrics_all = []

    # Iterar sobre cada grafo en la lista
    for i in range(len(graphList)):
        print(f"Grafo {chr(97 + i)}")
        print("Nodos del grafo original:", [node + 1 for node in graphList[i].nodes()])
        
        # Calcular las puntuaciones y la matriz CCC
        matrix_ccc = compute_score(graphList[i])[2]

        # Reducir el grafo utilizando la matriz CCC
        H = coarsenet_EqualsNodes(graphList[i], matrix_ccc)

        # Ajustar los nodos del grafo reducido
        adjusted_nodes = []
        for node in H.nodes():
            if isinstance(node, str) and '+' in node:
                parts = node.split('+')
                adjusted_node = '+'.join(str(int(part) + 1) for part in parts)
                adjusted_nodes.append(adjusted_node)
            else:
                adjusted_nodes.append(node + 1)

        print("Nodos ajustados de H:", adjusted_nodes)
        
        # Convertir los grafos a matrices de adyacencia
        adj_matrix_G = nx.to_numpy_array(graphList[i])
        adj_matrix_H = nx.to_numpy_array(H)
        
        # Calcular las métricas espectrales para el grafo original y el grafo reducido
        spectral_metrics = analyze_spectral_properties(adj_matrix_G, adj_matrix_H, k=5)
        
        spectral_metrics_all.append(spectral_metrics)
        
        # Imprimir y graficar las métricas espectrales del grafo original
        print("Métricas espectrales de G:")
        for metric_name, metric_value in spectral_metrics.items():
            if 'Original' in metric_name:
                print(f"{metric_name}: {metric_value:.6f}")
        plot_graph(graphList[i], title="Grafo Original")
        plt.savefig(f'/home/darian/Graph-Reduction-Project/graphOriginalNotEqual_{chr(97 + i)}.png')

        # Imprimir y graficar las métricas espectrales del grafo reducido
        print("Métricas espectrales de H:")
        for metric_name, metric_value in spectral_metrics.items():
            if 'Reduced' in metric_name:
                print(f"{metric_name}: {metric_value:.6f}")
        plot_graph(H, title="Grafo Reducido")
        plt.savefig(f'/home/darian/Graph-Reduction-Project/graphReducedNotEqual_{chr(97 + i)}.png')

    
    # Nombres de las métricas espectrales
    metrics_names = list(spectral_metrics_all[0].keys())
    num_metrics = len(metrics_names)
    x_labels = [chr(97 + i) for i in range(len(graphList))]

    # Crear un diccionario para almacenar los valores de las métricas
    data = {}
    for metric_name in metrics_names:
        base_metric_name = metric_name.split(' (')[0]
        if base_metric_name not in data:
            data[base_metric_name] = {'Original': [], 'Reduced': []}

    # Rellenar el diccionario con los valores de las métricas
    for i in range(len(x_labels)):
        for metric_name in metrics_names:
            base_metric_name = metric_name.split(' (')[0]
            if 'Original' in metric_name:
                data[base_metric_name]['Original'].append(spectral_metrics_all[i].get(metric_name, 0))
            elif 'Reduced' in metric_name:
                data[base_metric_name]['Reduced'].append(spectral_metrics_all[i].get(metric_name, 0))

    # Crear un DataFrame con MultiIndex en el encabezado
    tuples = [(metric_name, sub_metric) for metric_name in data.keys() for sub_metric in ['Original', 'Reduced']]
    index = pd.MultiIndex.from_tuples(tuples, names=['Metric', 'Type'])
    df = pd.DataFrame({x_labels[i]: [data[metric_name][sub_metric][i] for metric_name, sub_metric in tuples] for i in range(len(x_labels))}, index=index).T

    # Mostrar el DataFrame
    print(df)

    # Guardar el DataFrame en un archivo Excel
    df.to_excel('/home/darian/Graph-Reduction-Project/spectral_metricsNotEqual.xlsx')
    

    # Graficar las métricas espectrales
    for i in range(0, len(metrics_names), 2):
        metric_name = metrics_names[i].split(' (')[0]
        plt.figure(figsize=(10, 8))
        
        G_values = [metrics[metrics_names[i]] for metrics in spectral_metrics_all]
        H_values = [metrics[metrics_names[i + 1]] for metrics in spectral_metrics_all]
        
        plt.plot(x_labels, G_values, label=f'Grafo Original - {metric_name}', marker='o')
        plt.plot(x_labels, H_values, label=f'Grafo Reducido - {metric_name}', marker='x')
        
        plt.xlabel('Grafos Evaluados')
        plt.ylabel(metric_name)
        plt.title(f'Métrica Espectral: {metric_name}')
        plt.legend()
        plt.savefig(f'/home/darian/Graph-Reduction-Project/spectral_metricNotEqual_{metric_name}.png')
        plt.show()


def run_experimentAcademicGraph():
    listGraphs = []
    # Definir las aristas de los grafos académicos
    academic_graphs_edges = {
        "karate_club": nx.karate_club_graph().edges(),
        "dolphins": nx.read_gml('/home/darian/Graph-Reduction-Project/dolphins.gml').edges(),
        "political_books": nx.read_gml('/home/darian/Graph-Reduction-Project/polbooks.gml').edges(),
        "football": nx.read_gml('/home/darian/Graph-Reduction-Project/football.gml').edges()
    }

    # Crear los grafos y añadirlos a la lista
    for graph_name, edges in academic_graphs_edges.items():
        G, _ = create_indexed_graph(edges)
        listGraphs.append((graph_name, G))

    spectral_metrics_all = []

    # Continuar con el procesamiento de los grafos
    for graph_name, G in listGraphs:
        
        print(f"Grafo {graph_name}")
        print("Nodos del grafo original:", [node + 1 for node in G.nodes()])
        
        # Calcular las puntuaciones y la matriz CCC
        scores = compute_score(G)[0]

        H = coarsenet(G, scores, 0.25)

        # Ajustar los nodos del grafo reducido
        adjusted_nodes = []
        for node in H.nodes():
            if isinstance(node, str) and '+' in node:
                parts = node.split('+')
                adjusted_node = '+'.join(str(int(part) + 1) for part in parts)
                adjusted_nodes.append(adjusted_node)
            else:
                adjusted_nodes.append(node + 1)

        print("Nodos ajustados de H:", adjusted_nodes)
        
        # Convertir los grafos a matrices de adyacencia
        adj_matrix_G = nx.to_numpy_array(G)
        adj_matrix_H = nx.to_numpy_array(H)
        
        # Calcular las métricas espectrales para el grafo original y el grafo reducido
        spectral_metrics = analyze_spectral_properties(adj_matrix_G, adj_matrix_H)
        
        spectral_metrics_all.append(spectral_metrics)

        # # Imprimir y graficar las métricas espectrales del grafo original
        # print("Métricas espectrales de G:")
        # for metric_name, metric_value in spectral_metrics.items():
        #     if 'Original' in metric_name:
        #         print(f"{metric_name}: {metric_value:.6f}")
        # plot_graph(G, title=f"Grafo Original - {graph_name}")
        # plt.savefig(f'/home/darian/Graph-Reduction-Project/graphOriginal_{graph_name}.png')


        # # Imprimir y graficar las métricas espectrales del grafo reducido
        # print("Métricas espectrales de H:")
        # for metric_name, metric_value in spectral_metrics.items():
        #     if 'Reduced' in metric_name:
        #         print(f"{metric_name}: {metric_value:.6f}")
        # plot_graph(H, title=f"Grafo Reducido - {graph_name}")
        # plt.savefig(f'/home/darian/Graph-Reduction-Project/graphReduced_{graph_name}.png')
    
    # Nombres de las métricas espectrales
    metrics_names = list(spectral_metrics_all[0].keys())
    num_metrics = len(metrics_names)
    x_labels = [graph_name for graph_name, _ in listGraphs]

    # Crear un diccionario para almacenar los valores de las métricas
    data = {}
    for metric_name in metrics_names:
        base_metric_name = metric_name.split(' (')[0]
        if base_metric_name not in data:
            data[base_metric_name] = {'Original': [], 'Reduced': []}

    # Rellenar el diccionario con los valores de las métricas
    for i in range(len(x_labels)):
        for metric_name in metrics_names:
            base_metric_name = metric_name.split(' (')[0]
            if 'Original' in metric_name:
                data[base_metric_name]['Original'].append(spectral_metrics_all[i].get(metric_name, 0))
            elif 'Reduced' in metric_name:
                data[base_metric_name]['Reduced'].append(spectral_metrics_all[i].get(metric_name, 0))

    # Crear un DataFrame con MultiIndex en el encabezado
    tuples = [(metric_name, sub_metric) for metric_name in data.keys() for sub_metric in ['Original', 'Reduced']]
    index = pd.MultiIndex.from_tuples(tuples, names=['Metric', 'Type'])
    df = pd.DataFrame({x_labels[i]: [data[metric_name][sub_metric][i] for metric_name, sub_metric in tuples] for i in range(len(x_labels))}, index=index).T

    # Mostrar el DataFrame
    print(df)

    # Guardar el DataFrame en un archivo Excel
    df.to_excel('/home/darian/Graph-Reduction-Project/coarseNetSpectral_metricsAcademicNets.xlsx')


    # # Graficar las métricas espectrales
    # for i in range(0, len(metrics_names), 2):
    #     metric_name = metrics_names[i].split(' (')[0]
    #     plt.figure(figsize=(10, 8))
        
    #     G_values = [metrics[metrics_names[i]] for metrics in spectral_metrics_all]
    #     H_values = [metrics[metrics_names[i + 1]] for metrics in spectral_metrics_all]
        
    #     plt.plot(x_labels, G_values, label=f'Grafo Original - {metric_name}', marker='o')
    #     plt.plot(x_labels, H_values, label=f'Grafo Reducido - {metric_name}', marker='x')
        
    #     plt.xlabel('Grafos Evaluados')
    #     plt.ylabel(metric_name)
    #     plt.title(f'Métrica Espectral: {metric_name}')
    #     plt.legend()
    #     plt.savefig(f'/home/darian/Graph-Reduction-Project/coarseNetSpectral_metricAcademicNets_{metric_name}.png')
    #     plt.show()


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
  
def find_most_similar_pairs(ccc_values):

    """
    Encuentra los pares de vértices más similares según el valor de CCC.

    Parámetros:
    - ccc_values: Una lista de tuplas (nodo, ccc).

    Retorna:
    - Una lista de tuplas ((nodo1, nodo2), ccc_diff) ordenada de menor a mayor según ccc_diff.
    """
    similar_pairs = []
    num_nodes = len(ccc_values)

    # Comparar cada par de nodos
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            node1, ccc1 = ccc_values[i]
            node2, ccc2 = ccc_values[j]
            ccc_diff = abs(ccc1 - ccc2)
            similar_pairs.append(((node1, node2), ccc_diff))

    # Ordenar las tuplas por ccc_diff en orden ascendente
    similar_pairs_sorted = sorted(similar_pairs, key=lambda x: x[1])

    return similar_pairs_sorted

def generate_random_graph(n, p):
    return nx.erdos_renyi_graph(n, p)

def coarsenet_reduction(G, threshold):
    """
    Reduce un grafo utilizando el método Coarsenet.

    Parámetros:
    - G: Un grafo de NetworkX.
    - threshold: Umbral para la reducción.

    Retorna:
    - H: Un grafo reducido.
    """
    # Calcular las puntuaciones y la matriz CCC
    scores = compute_score(G)[0]

    # Reducir el grafo utilizando las puntuaciones
    H = coarsenet(G, scores, threshold)

    return H

# Función para añadir peso basado en la centralidad de intermediación
def add_edge_betweenness_weights(graph):
    betweenness = nx.edge_betweenness_centrality(graph)
    for u, v, data in graph.edges(data=True):
        data['weight'] = 1 + betweenness[(u, v)] * (len(graph.edges()) - 1)

# Función para añadir peso basado en el número de vecinos comunes
def add_common_neighbors_weights(graph):
    for u, v in graph.edges():
        common_neighbors = len(list(nx.common_neighbors(graph, u, v)))
        graph[u][v]['weight'] = 1 + common_neighbors

# Función para añadir peso basado en la distancia recíproca
def add_reciprocal_distance_weights(graph):
    shortest_paths = dict(nx.shortest_path_length(graph))
    for u, v in graph.edges():
        distance = shortest_paths[u][v]
        graph[u][v]['weight'] = 1 + (1 / distance if distance > 0 else 0)

# Función para añadir peso basado en el PageRank de los nodos
def add_edge_pagerank_weights(graph):
    pagerank = nx.pagerank(graph)
    for u, v in graph.edges():
        graph[u][v]['weight'] = 1 + pagerank[u] + pagerank[v]

# Aplicar la ponderación a cada grafo académico usando un indicador específico
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

def main():
    # Evaluar los 4 grafos académicos
    academic_graphs_edges = {
        "karate": nx.karate_club_graph().edges(),
        "dolphins": nx.read_gml('/home/darian/Graph-Reduction-Project/dolphins.gml').edges(),
        "polbooks": nx.read_gml('/home/darian/Graph-Reduction-Project/polbooks.gml').edges(),
        "football": nx.read_gml('/home/darian/Graph-Reduction-Project/football.gml').edges()
    }

    print("Setear pesos en los grafos académicos")
    # Aplicar pesos a cada grafo con el método deseado
    for graph_name, edges in academic_graphs_edges.items():
        graph, _ = create_indexed_graph(edges)
        # graph = nx.to_numpy_array(graph)
        print(f"Graph: {graph_name}")
        apply_weighting(graph, method='pagerank')  # Cambia 'betweenness' por el método que prefieras
        for u, v, data in graph.edges(data=True):
            print(f"Arista ({u}, {v}) - Peso: {data['weight']}")
        
        print(f"Nodos y atributos de {graph_name}:", graph.nodes(data=True))
        print(f"Aristas y atributos de {graph_name}:", graph.edges(data=True))

        # # Aplicar el método de Lee
        # H_reduced_graph_LeeSun = spectral_sparsifier_reduction(graph)
        # print("Graph reduced with Lee Method")

        # # Evaluar las métricas espectrales para el método de Lee
        # metrics_lee = analyze_spectral_properties(graph, H_reduced_graph_LeeSun)

        # print(f"\nMetrics for {graph_name} - Lee Method:")
        # for metric, value in metrics_lee.items():
        #     print(f"{metric}: {value}")

    
    print("Experimento con grafos académicos")
    for graph_name, edges in academic_graphs_edges.items():
        graph, _ = create_indexed_graph(edges)
        # graph = nx.to_numpy_array(graph)
        print(f"Graph: {graph_name}")

        print("Nodos y atributos de G1:", graph.nodes(data=True))
        print("Aristas y atributos de G1:", graph.edges(data=True))


        # Aplicar el método de reducción CoreNet
        H_reduced_graph_Coarsenet = coarsenet_reduction(graph, 0.25)
        print("Graph reduced with CoreNet")

        # # Instanciar y aplicar el método de reducción TwiceRamanujan
        # tr = TwiceRamanujan(graph, d=2, verbose=2)
        # H_reduced_graph_TR = tr.sparsify_Parallel()
        # print("Graph reduced with TwiceRamanujan")

        # # Aplicar el método de Lee
        # H_reduced_graph_LeeSun = spectral_sparsifier_reduction(graph)
        # print("Graph reduced with Lee Method")

        # # Aplicar el método de Sparsification by Effective Resistance
        # H_reduced_graph_EffectiveResistance = effective_resistance_reduction(graph)
        # print("Graph reduced with Sparsification by Effective Resistance")

        # Evaluar las métricas espectrales para CoreNet
        metrics_cn = analyze_spectral_properties(graph, H_reduced_graph_Coarsenet)

        # # Evaluar las métricas espectrales para TwiceRamanujan
        # metrics_tr = analyze_spectral_properties(graph, H_reduced_graph_TR)

        # # Evaluar las métricas espectrales para el método de Lee
        # metrics_lee = analyze_spectral_properties(graph, H_reduced_graph_LeeSun)

        # # Evaluar las métricas espectrales para Sparsification by Effective Resistance
        # metrics_sr = analyze_spectral_properties(graph, H_reduced_graph_EffectiveResistance)
        
        # Imprimir las métricas
        # print(f"Metrics for {graph_name} - TwiceRamanujan:")
        # for metric, value in metrics_tr.items():
        #     print(f"{metric}: {value}")

        # print(f"\nMetrics for {graph_name} - Lee Method:")
        # for metric, value in metrics_lee.items():
        #     print(f"{metric}: {value}")

        # print(f"\nMetrics for {graph_name} - Sparsification by Effective Resistance:")
        # for metric, value in metrics_sr.items():
        #     print(f"{metric}: {value}")

        print(f"\nMetrics for {graph_name} - CoreNet:")
        for metric, value in metrics_cn.items():
            print(f"{metric}: {value}")

        # Graficar el grafo original
        # plot_graph(graph, title=f"Original Graph - {graph_name}")

        # Graficar el grafo reducido por TwiceRamanujan
        # plot_graph(H_reduced_graph_TR, title=f"Reduced Graph (TwiceRamanujan) - {graph_name}")

        # # Graficar el grafo reducido por el método de Lee
        # plot_graph(H_reduced_graph_LeeSun, title=f"Reduced Graph (Lee Method) - {graph_name}")

        # # Graficar el grafo reducido por Sparsification by Effective Resistance
        # plot_graph(H_reduced_graph_EffectiveResistance, title=f"Reduced Graph (Sparsification by Effective Resistance) - {graph_name}")

        # Graficar el grafo reducido por CoreNet
        # plot_graph(H_reduced_graph_Coarsenet, title=f"Reduced Graph (CoreNet) - {graph_name}")

    





if __name__ == "__main__":
    # main()
    run_experimentAcademicGraph()
    


    
