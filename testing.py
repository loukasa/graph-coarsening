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


def compare_adjacencies(original_graph, reduced_graph):
    original_edges = set(original_graph.edges())
    reduced_edges = set(reduced_graph.edges())

    new_edges = reduced_edges - original_edges

    print(f"Número de nuevas aristas en el grafo reducido: {len(new_edges)}")
    print(f"Nuevas aristas en el grafo reducido: {new_edges}")

    return new_edges


# Función para leer archivos .xlsx
def read_excel_file(filepath):
    df = pd.read_excel(filepath, engine='openpyxl')
    return df

# Función principal
def main():
    # Directorio donde se encuentran los archivos .xlsx
    directory = '/home/darian/Graph-Reduction-Project/'

    # Leer archivos
    coarse_filepath = os.path.join(directory, 'coarseNetSpectral_metricsAcademicNets.xlsx')
    coarsening_filepath = os.path.join(directory, 'coarseningMethods_all_networks.xlsx')

    coarse_df = read_excel_file(coarse_filepath)
    coarsening_df = read_excel_file(coarsening_filepath)
    print(coarse_df)
    print(coarsening_df)

    # Unificar los DataFrames
    combined_df = pd.concat([coarse_df, coarsening_df], ignore_index=True)
    print(combined_df)

    # Guardar el DataFrame combinado en un archivo testall.xlsx
    combined_df.to_excel(os.path.join(directory, 'testall.xlsx'), index=False)

    # Crear gráficos para cada red evaluada y métrica
    combined_df.dropna(subset=['Method'], inplace=True)
    combined_df.fillna(0, inplace=True)

    print(combined_df)

    networks = combined_df['Network'].unique()
    metrics = combined_df['Metric'].unique()

    for network in networks:
        for metric in metrics:
            plt.figure()
            network_df = combined_df[combined_df['Network'] == network]
            print(network_df)

            # Filtrar los datos para la red y métrica actuales
            metric_df = network_df[network_df['Metric'] == metric]
            print(metric_df)
           
            # Obtener el valor original
            original_value = metric_df.loc[metric_df['Method'] == 'coarseNet', 'Original']
            print(original_value)

            # Obtener los valores de los métodos reducidos
            reduced_methods = metric_df['Method'].unique()
            print(reduced_methods)
          
            values = [metric_df[metric_df['Method'] == method]['Reduced'].values[0] for method in reduced_methods]
            print(values)

            # Crear el gráfico
            plt.figure()
            plt.bar(range(len(reduced_methods)), values, label='Reduced Methods')
            
            plt.axhline(y=original_value.values[0], color='r', linestyle='--', label='Original Value')

            plt.xlabel('Methods')
            plt.ylabel(metric)
            plt.title(f'{network} - {metric}')
            plt.xticks(range(len(reduced_methods)), reduced_methods, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(directory, f'{network}_{metric}.png'))
            plt.close()

def testBigNetsAndSaveToGML(pathBignets, output_path):
    
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

    
    # Obtener la lista de archivos .gml en el directorio pathBignets
    graph_files = glob.glob(os.path.join(pathBignets, '*.gml'))

    for graph_file in graph_files:
        graph_name = os.path.splitext(os.path.basename(graph_file))[0]
        print(f"\nEvaluando grafo: {graph_name}")

        # Cargar el grafo desde el archivo .gml
        nx_graph = nx.read_gml(graph_file)

        # Convertir el grafo de NetworkX a la estructura esperada por graph_lib
        G = graph_lib.to_pygsp_graph(nx_graph)

        # Imprimir el grafo original con indexación y sus adyacentes
        # print(f"Grafo original ({graph_name}):")
        # for node in nx_graph.nodes():
        #     neighbors = list(nx_graph.neighbors(node))
        #     print(f"Node {node}: {neighbors}")

        # Calcular base espectral para operaciones avanzadas
        G.compute_fourier_basis()

        for method in methods:
            print(f"    - Método: {method}")

            # Aplicar coarsening
            _, Gc, Call, _ = coarsening_utils.coarsen(G, r=r, method=method) # en Call está lo que necesito, propiedad indices

            nx_graph_H = nx.from_scipy_sparse_array(Gc.W)
            # acá tengo que crear las propiedades 'sizeSuperNode' y 'nodesInSuperNode' en nx_graph_H 
            _, G_reducido_update = setPropertiesToNodes(nx_graph_H, Call, G)

            # Imprimir el grafo reducido con indexación y sus adyacentes
            # print(f"Grafo reducido ({graph_name} - {method}):")
            # for node in G_reducido_update.nodes():
            #     neighbors = list(G_reducido_update.neighbors(node))
            #     print(f"Node {node}: {neighbors}")

            # Guardar los grafos en formato GML para visualización en Gephi
            nx.write_gml(G_reducido_update, f'/home/darian/graph-coarsening/results/{graph_name}_{method}_reduced.gml')
            print(f"Archivo GML guardado: {graph_name}_{method}_reduced.gml")

    

def testAcademicsNetsAndSaveToGML():
    
    # Parámetros globales
    r = 0.6  # coarsening ratio
    methods = [
        "variation_neighborhoods",
        # "variation_edges",
        # "variation_cliques",
        # "heavy_edge",
        # "algebraic_JC",
        # "affinity_GS",
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

        # # Imprimir el grafo original con indexación y sus adyacentes
        # print(f"Grafo original ({graph_name}):")
        # for node in nx_graph.nodes():
        #     neighbors = list(nx_graph.neighbors(node))
        #     print(f"Node {node}: {neighbors}")

        # Calcular base espectral para operaciones avanzadas
        G.compute_fourier_basis()


        for method in methods:
            print(f"    - Método: {method}")

            # Aplicar coarsening
            _, Gc, Call, _ = coarsening_utils.coarsen(G, r=r, method=method) # en Call está lo que necesito, propiedad indices
            # TODO acá es el problema.... con esos pesos me da otra matrix de adyacencia....
            nx_graph_H = nx.from_scipy_sparse_array(Gc.W)

            # acá tengo que crear las propiedades 'sizeSuperNode' y 'nodesInSuperNode' en nx_graph_H 
            _, G_reducido_update = setPropertiesToNodes(nx_graph_H, Call, G)

            # # Imprimir el grafo reducido con indexación y sus adyacentes
            # print(f"Grafo reducido ({graph_name} - {method}):")
            # for node in G_reducido_update.nodes():
            #     neighbors = list(G_reducido_update.neighbors(node))
            #     print(f"Node {node}: {neighbors}")

            # Guardar los grafos en formato GML para visualización en Gephi
            # nx.write_gml(nx_graph, f'/home/darian/Graph-Reduction-Project/{graph_name}_{method}_original.gml')
            nx.write_gml(G_reducido_update, f'/home/darian/graph-coarsening/results/{graph_name}_{method}_reduced.gml')

            # saveToGMLwithCommunityLabels(graph_name, method=method)

            # Convertir los grafos a matrices de adyacencia
            adj_matrix_G = nx.to_numpy_array(nx_graph, weight=None)
            adj_matrix_H = nx.to_numpy_array(nx_graph_H, weight=None)

            # Guardar la matriz de adyacencia adj_matrix_H en un archivo .txt iterando fila por fila
            adj_matrix_H_filename = os.path.join('/home/darian/graph-coarsening/results/', f'{graph_name}_{method}_adj_matrix_H.txt')
            with open(adj_matrix_H_filename, 'w') as f:
                for row in adj_matrix_H:
                    f.write(' '.join(map(str, row.astype(int))) + '\n')
            print(f"Matriz de adyacencia guardada en: {adj_matrix_H_filename}")

            # Imprimir la matriz de adyacencia adj_matrix_H
            # print("Matriz de adyacencia (adj_matrix_H):")
            # print(adj_matrix_H)
            
            # Calcular las métricas espectrales para el grafo original y el grafo reducido
            spectral_metrics = analyze_spectral_properties(adj_matrix_G, adj_matrix_H)

            spectral_metrics['Network'] = graph_name
            spectral_metrics['Method'] = method

            spectral_metrics_all.append(spectral_metrics)

    save_metrics_to_excel(spectral_metrics_all, '/home/darian/graph-coarsening/results/metrics_results.xlsx')

def setPropertiesToNodes(graph, Call, G):
    print("Setting properties to nodes")
    for node in graph.nodes():
        graph.nodes[node]['sizeSuperNode'] = 0
        graph.nodes[node]['nodesInSuperNode'] = []

    print("Call:", Call)
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

def saveToGMLwithCommunityLabels(networkName, method):

    # Leer los archivos GML
    print(f"Reading GML files for {networkName} and {method}")
    original_file = glob.glob(f'/home/darian/graph-coarsening/AcademicsNets/{networkName}.gml')[0]
    graph_original = nx.read_gml(original_file)
    reduced_file = glob.glob(f'/home/darian/graph-coarsening/results/*{networkName}*{method}*_reduced.gml')[0]
    graph_reduced = nx.read_gml(reduced_file)
    # graph_original_with_communities = nx.read_gml(f'/home/darian/Graph-Reduction-Project/netsToGephiWithComm/{networkName}.gml')

    # Obtener las etiquetas de comunidad de los nodos en G
    # community_labels = nx.get_node_attributes(graph_original_with_communities, 'gt')
    community_labels = nx.get_node_attributes(graph_original, 'gt')

    # Imprimir las etiquetas de comunidad como un diccionario por vértices
    print("Etiquetas de comunidad por vértices:")
    pprint.pprint(community_labels)
    
    print("Nodos y atributos de G original:", graph_original.nodes(data=True))

    # Asignar las etiquetas de comunidad a los nodos en graph_reduced
    for node in graph_reduced.nodes():
        if graph_reduced.nodes[node]['sizeSuperNode'] != 0:
            nodes_in_supernode = eval(graph_reduced.nodes[node]['nodesInSuperNode'])
            community_counts = {}
            for subnode in nodes_in_supernode:
                community = community_labels.get(str(subnode), None)
                if community:
                    if community in community_counts:
                        community_counts[community] += 1
                    else:
                        community_counts[community] = 1
            if not community_counts:
                print('La estructura community_counts está vacía')
            most_common_community = max(community_counts, key=community_counts.get)
            graph_reduced.nodes[node]['gt'] = most_common_community
            
        else:
            if node in community_labels:
                graph_reduced.nodes[node]['gt'] = community_labels[node]

    print("Nodos y atributos de G reducido:", graph_reduced.nodes(data=True))
    print("Done!!!!!")

    _, changed = calculate_accuracy(graph_original, graph_reduced)
    for node in graph_reduced.nodes():
        if node in changed:
            graph_reduced.nodes[node]['nodesChanged'] = str(changed[node])
        else:
            graph_reduced.nodes[node]['nodesChanged'] = "[]"

    print("Nodos y atributos de G reducido con nodos cambiados:", graph_reduced.nodes(data=True))
    print("Done!!!!!")

    # Eliminar la propiedad 'weight' de los nodos en graph_original
    for u, v, d in graph_original.edges(data=True):
        if 'weight' in d:
            del d['weight']

    # Eliminar la propiedad 'weight' de los nodos en graph_reduced
    for u, v, d in graph_reduced.edges(data=True):
        if 'weight' in d:
            del d['weight']

    # Guardar los grafos en formato GML para visualización en Gephi
    nx.write_gml(graph_reduced, f'/home/darian/graph-coarsening/results/{networkName}_{method}_reduced_with_communities.gml')

def calculate_accuracy(original_graph, reduced_graph):
    total_nodes = 0
    matching_nodes = 0
    changes = {}

    for supernode in reduced_graph.nodes():
        supernode_gt = reduced_graph.nodes[supernode]['gt']
        nodes_in_supernode = eval(str(reduced_graph.nodes[supernode]['nodesInSuperNode']))

        for node in nodes_in_supernode:
            original_gt = original_graph.nodes[str(node)]['gt']
            total_nodes += 1
            if original_gt == supernode_gt:
                matching_nodes += 1
            else:
                if supernode not in changes:
                    changes[supernode] = []
                changes[supernode].append((str(node), original_gt, supernode_gt))

    accuracy = matching_nodes / total_nodes if total_nodes > 0 else 0
    return accuracy, changes

def evaluate_graphs(original_graph, reduced_graph):
    results = []
    accuracy, changed = calculate_accuracy(original_graph, reduced_graph)

    for supernode in reduced_graph.nodes():
        supernode_gt = reduced_graph.nodes[supernode]['gt']
        nodes_in_supernode = eval(str(reduced_graph.nodes[supernode]['nodesInSuperNode']))

        for node in nodes_in_supernode:
            original_gt = original_graph.nodes[str(node)]['gt']
            if original_gt != supernode_gt:
                results.append({
                    'Supernode': supernode,
                    'Node': str(node),
                    'Supernode_GT': supernode_gt,
                    'Original_GT': original_gt
                })

    return results, accuracy, changed

def evlauateGT():
    # Directorio donde se encuentran los archivos GML
    directory = '/home/darian/Graph-Reduction-Project/networkToGML/'

    # Obtener todos los archivos GML en el directorio
    gml_files = glob.glob(os.path.join(directory, '*_reduced_with_communities.gml'))

    # Diccionario para almacenar los resultados por red
    network_results = {}

    # Iterar sobre los archivos GML
    for gml_file in gml_files:
        print(f"Evaluando archivo: {gml_file}")
        
        # Obtener el nombre de la red y el método
        filename = os.path.basename(gml_file)
        network_name = filename.split('_')[0]
        method = filename.split('_', 1)[1].split('_reduced')[0].split('_original')[0]

        # Leer los grafos
        original_graph = nx.read_gml(gml_file.replace('_reduced_with_communities.gml', '_original_with_communities.gml'))
        reduced_graph = nx.read_gml(gml_file)

        # Evaluar la eficacia
        results, accuracy, changed = evaluate_graphs(original_graph, reduced_graph)

        # Crear un DataFrame con los resultados
        results_df = pd.DataFrame(results)
        results_df['Method'] = method

        # Agregar los resultados al diccionario
        if network_name not in network_results:
            network_results[network_name] = {'results': results_df, 'accuracy': []}
        else:
            network_results[network_name]['results'] = pd.concat([network_results[network_name]['results'], results_df], ignore_index=True)

        # Agregar la precisión al diccionario
        network_results[network_name]['accuracy'].append({'Method': method, 'Accuracy': accuracy})

    print("Resultados por red:")
    # Guardar los resultados por red en archivos CSV y Excel
    for network_name, data in network_results.items():
        results_df = data['results']
        accuracy_df = pd.DataFrame(data['accuracy'])

        # Guardar los resultados de precisión
        accuracy_excel_filename = os.path.join(directory, f'{network_name}_accuracy_results.xlsx')
        accuracy_df.to_excel(accuracy_excel_filename, index=False)

        # Guardar los resultados de los nodos que no cumplieron con el mismo 'gt'
        results_excel_filename = os.path.join(directory, f'{network_name}_evaluation_results.xlsx')
        results_df.to_excel(results_excel_filename, index=False)

        # Imprimir los resultados
        print(f"Resultados para la red {network_name}:")
        print(results_df)
        print(f"Precisión para la red {network_name}:")
        print(accuracy_df)

def update_gt_property(filepath):
    # Leer el archivo GML
    graph = nx.read_gml(filepath)

    # Actualizar la propiedad 'gt' de cada nodo
    for node in graph.nodes():
        if 'gt' in graph.nodes[node]:
            value = graph.nodes[node]['gt']
            graph.nodes[node]['gt'] = str(value)

            # "{value}".format(value=value)

    # Guardar el archivo GML actualizado
    nx.write_gml(graph, filepath)

def update_idNode_property(filepath):
    # Leer el archivo GML
    graph = nx.read_gml(filepath)

    # Crear un mapeo de los nombres de los equipos a números
    node_mapping = {node: str(i) for i, node in enumerate(graph.nodes())}

    # Actualizar la propiedad 'label' de cada nodo
    for node in graph.nodes():
        graph.nodes[node]['label'] = node_mapping[node]

    # Renombrar los nodos en el grafo
    graph = nx.relabel_nodes(graph, node_mapping)

    # Guardar el archivo GML actualizado
    new_filepath = filepath.replace('.gml', '_updateIdNode.gml')
    nx.write_gml(graph, new_filepath)

def create_latex_tables(results):
    # Convertir los resultados en un DataFrame
    df = pd.DataFrame(results)

    # Crear un archivo .tex para todas las tablas
    with open('all_results.tex', 'w') as f:
        for (network, method), group in df.groupby(['Network', 'Method']):
            latex_table = group.to_latex(index=False)
            f.write(f"\\section*{{{network} - {method}}}\n")
            f.write(latex_table)
            f.write("\n\n")

    print("Todas las tablas LaTeX guardadas en all_results.tex")


import networkx as nx
import json

def update_dolphins_network(input_file, output_gml_file, output_mapping_file):
    """
    Actualiza la red de delfines en formato GML:
    - Cambia la propiedad 'community' a 'gt'.
    - Asigna el 'label' al valor del 'id' del nodo.
    - Crea un mapeo de los 'labels' antiguos al nuevo valor del 'id'.
    
    Args:
        input_file (str): Ruta al archivo GML de entrada.
        output_gml_file (str): Ruta donde se guardará el archivo GML actualizado.
        output_mapping_file (str): Ruta donde se guardará el archivo JSON con el mapeo de etiquetas.

    Returns:
        None
    """
    # Cargar la red desde el archivo GML
    G = nx.read_gml(input_file)

    # Definir las comunidades según los nombres de los nodos

    community_1 = {
        "Beescratch", "DN16", "DN21", "DN63", "Feather", "Gallatin", "Jet", "Knit", "MN23",
        "Mus", "Notch",  "Number1", "Quasi", "Ripplefluke", "SN90", "TR82", "Upbang", "Wave", "Web", "Zig",
    }

    community_2 = {
        "Beak", "Bumper", "Fish",  "Oscar", "PL", "SN96", "TR77", 
    }

    community_3 = {
        "CCL", "Double", "Fork", "Grin",  "Hook", "Kringel", "Scabs", "Shmuddel","SN100", "SN4", "SN63", "SN89",
        "SN9", "Stripes", "Thumper", "TR120", "TR88", "TR99", "TSN103", "TSN83", "Whitetip", "Zap", "Zipfel"
    }

    community_4 = {
        "Cross", "Five", "Haecksel", "Jonah",  "MN105", "MN60", "MN83", "Patchback", "SMN5", "Topless", "Trigger",  "Vau",
    }

    # Crear un diccionario para almacenar la relación entre labels antiguos y nuevos
    label_mapping = {}

    # Asignar comunidades y cambiar labels
    for i, (node_id, node_data) in enumerate(G.nodes(data=True)):
        old_label = node_id  # Label original es el ID del nodo

        # Asignar comunidades como 'gt'
        if old_label in community_1:
            G.nodes[node_id]['gt'] = "1"
        elif old_label in community_2:
            G.nodes[node_id]['gt'] = "2"
        elif old_label in community_3:
            G.nodes[node_id]['gt'] = "3"
        elif old_label in community_4:
            G.nodes[node_id]['gt'] = "4"
        else:
            G.nodes[node_id]['gt'] = 0  # Nodo sin comunidad

        # Guardar el mapeo del label antiguo al nuevo
        label_mapping[old_label] = str(i)

        # Cambiar el label al valor del id
        G.nodes[node_id]['label'] = str(i)

    # Renombrar los nodos en el grafo
    G = nx.relabel_nodes(G, label_mapping)

    # Guardar el archivo actualizado
    nx.write_gml(G, output_gml_file)

    # Guardar el diccionario de mapeo en un archivo JSON
    with open(output_mapping_file, "w") as file:
        json.dump(label_mapping, file, indent=4)

    print(f"Archivo GML actualizado guardado como '{output_gml_file}'")
    print(f"Mapeo de labels guardado como '{output_mapping_file}'")

def assign_karate_gt(input_file, output_file):
    """
    Asigna las comunidades ground truth a la red de Karate en el atributo 'gt'.
    
    Args:
        input_file (str): Ruta al archivo GML de entrada.
        output_file (str): Ruta donde se guardará el archivo GML actualizado.

    Returns:
        None
    """
    # Cargar la red desde el archivo GML
    G = nx.read_gml(input_file)

    # Ground truth: Asignar comunidades conocidas
    community_0 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21}  # Comunidad 0
    community_1 = {9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}  # Comunidad 1

    for node in G.nodes():
        if int(node) in community_0:
            G.nodes[node]['gt'] = 0
        elif int(node) in community_1:
            G.nodes[node]['gt'] = 1
        else:
            G.nodes[node]['gt'] = -1  # Nodo sin comunidad (si aplica)

    # Guardar el archivo actualizado
    nx.write_gml(G, output_file)
    print(f"Archivo GML actualizado con comunidades ground truth guardado como '{output_file}'")





def process_gml_files(directory):
    # Obtener la lista de archivos .gml que contienen la subcadena 'reduced'
    gml_files = [f for f in os.listdir(directory) if f.endswith('.gml') and 'reduced' in f]

    results = []

    for gml_file in gml_files:
        # Extraer el nombre de la red y el método del nombre del archivo
        parts = gml_file.split('_')
        network_name = parts[0]
        method_name = '_'.join(parts[1:parts.index('reduced')])

        # Leer el archivo .gml
        filepath = os.path.join(directory, gml_file)
        graph = nx.read_gml(filepath)

        for node in graph.nodes():
            if graph.nodes[node].get('sizeSuperNode', 0) != 0:
                label = graph.nodes[node].get('label', node)
                nodes_in_supernode = str(graph.nodes[node].get('nodesInSuperNode', ''))
                gt = graph.nodes[node].get('gt', '')
                nodes_changed = str(graph.nodes[node].get('nodesChanged', ''))

                results.append({
                    'Network': network_name,
                    'Method': method_name,
                    'Supernode_Label': label,
                    'NodesInSuperNode': nodes_in_supernode,
                    'GT': gt,
                    'NodesChanged': nodes_changed
                })

    return results

if __name__ == "__main__":
    print("Starting...")


    # print("Starting")
    # input_path = "nets/BigNets/"
    # output_path = "result/"
    # testBigNetsAndSaveToGML(input_path, output_path)
    # testCoarseningMethods(input_path, output_path, our_method=True)
    # print("Finished")

    # assign_karate_gt("karate_originalPrueba.gml", "karate_with_gt.gml")

    # print("Done!")

    # input_file="dolphins(OldLabels).gml",
    # output_gml_file="dolphins_updated.gml",
    # output_mapping_file="label_mapping.json"
    # update_dolphins_network(input_file="dolphins(OldLabels).gml",output_gml_file="dolphins_updated.gml",output_mapping_file="label_mapping.json")   

    # print("Done!")

    # update_idNode_property('/home/darian/Graph-Reduction-Project/netsToGephiWithComm/polbooks.gml')
    # update_gt_property('/home/darian/Graph-Reduction-Project/netsToGephiWithComm/football.gml')

    
    testAcademicsNetsAndSaveToGML()

    # evlauateGT()
    
    # graph_names = ["karate", "dolphins", "polbooks", "football"]
    # methods = [
    #     "coarseNet",
    #     "variation_neighborhoods",
    #     "variation_edges",
    #     "variation_cliques",
    #     "heavy_edge",
    #     "algebraic_JC",
    #     "affinity_GS",
    #     "kron",
    # ]
    # for graph_name in graph_names:
    #     for method in methods:
    #         saveToGMLwithCommunityLabels(graph_name, method)
    


    # # Directorio donde se encuentran los archivos .gml
    # directory = '/home/darian/Graph-Reduction-Project/networkToGML/'

    # # Procesar los archivos .gml
    # results = process_gml_files(directory)

    # # Crear tablas en formato LaTeX
    # create_latex_tables(results)

    print("Done!")