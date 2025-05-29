import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_data_grouped_by_city(file_path):
    """Load and group robbery descriptions by city from Excel file."""
    df = pd.read_excel(file_path)
    grouped = df.groupby('municipio')['descricao'].apply(lambda x: ' '.join(x)).reset_index()
    cidades = grouped['municipio'].tolist()
    textos_agrupados = grouped['descricao'].tolist()
    return textos_agrupados, cidades

def generate_embeddings(descriptions):
    """Generate embeddings for robbery descriptions using sentence-transformers."""
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    embeddings = model.encode(descriptions)
    return embeddings

def build_semantic_graph(cities, embeddings, initial_threshold=0.7):
    """Build a connected graph based on semantic similarity."""
    # Calculate similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes (cities)
    for city in cities:
        G.add_node(city)
    
    # Add edges based on similarity threshold
    threshold = initial_threshold
    while not nx.is_connected(G):
        # Clear existing edges
        G.clear_edges()
        
        # Add edges above threshold
        for i in range(len(cities)):
            for j in range(i+1, len(cities)):
                if sim_matrix[i][j] > threshold:
                    G.add_edge(cities[i], cities[j], weight=sim_matrix[i][j])
        
        # If graph is not connected, decrease threshold
        if not nx.is_connected(G):
            threshold -= 0.05
            print(f"Graph not connected. Decreasing threshold to {threshold:.2f}")
            
            # If threshold gets too low, break to avoid infinite loop
            if threshold < 0.1:
                print("Warning: Could not create connected graph with reasonable threshold")
                break
    
    return G, sim_matrix

def ensure_results_dir():
    os.makedirs('results-graph', exist_ok=True)

def visualize_graph(G):
    """Visualize the graph using matplotlib."""
    ensure_results_dir()
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=1500, font_size=8, font_weight='bold')
    plt.title("Semantic Similarity Graph of Brazilian Cities")
    plt.savefig('results-graph/semantic_graph.png')
    plt.close()

def plot_adjacency_heatmap(A, cidades, filename):
    ensure_results_dir()
    plt.figure(figsize=(12, 10))
    plt.imshow(A, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Peso da aresta')
    plt.title('Matriz de Adjacência (heatmap)')
    plt.xlabel('Cidade')
    plt.ylabel('Cidade')
    step = max(1, len(cidades)//20)
    plt.xticks(range(0, len(cidades), step), [cidades[i] for i in range(0, len(cidades), step)], rotation=90, fontsize=8)
    plt.yticks(range(0, len(cidades), step), [cidades[i] for i in range(0, len(cidades), step)], fontsize=8)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def process_sampled_graph(n_sample=25):
    ensure_results_dir()
    df = pd.read_excel('assaltos.xlsx')
    cidades_unicas = df['municipio'].drop_duplicates().sample(n=n_sample, random_state=42)
    df_sample = df[df['municipio'].isin(cidades_unicas)]
    grouped = df_sample.groupby('municipio')['descricao'].apply(lambda x: ' '.join(x)).reset_index()
    cidades = grouped['municipio'].tolist()
    textos_agrupados = grouped['descricao'].tolist()
    print(f"\nProcessando amostra de {n_sample} cidades...")
    embeddings = generate_embeddings(textos_agrupados)
    G, sim_matrix = build_semantic_graph(cidades, embeddings, initial_threshold=0.7)
    A = nx.to_numpy_array(G, weight='weight')
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', 
            node_size=1500, font_size=10, font_weight='bold')
    plt.title(f"Semantic Similarity Graph (Amostra de {n_sample} cidades)")
    plt.savefig('results-graph/semantic_graph_25.png')
    plt.close()
    plot_adjacency_heatmap(A, cidades, 'results-graph/adjacency_heatmap_25.png')

def main():
    ensure_results_dir()
    threshold = 0.7  # Ajuste aqui o threshold desejado
    textos_agrupados, cidades = load_data_grouped_by_city('assaltos.xlsx')
    print("Gerando embeddings...")
    embeddings = generate_embeddings(textos_agrupados)
    print("Construindo grafo de similaridade semântica...")
    G, sim_matrix = build_semantic_graph(cidades, embeddings, initial_threshold=threshold)
    A = nx.to_numpy_array(G, weight='weight')
    print("Visualizando grafo...")
    visualize_graph(G)
    print("Salvando matriz de adjacência como heatmap...")
    plot_adjacency_heatmap(A, cidades, 'results-graph/adjacency_heatmap.png')
    np.save('results-graph/adjacency_matrix.npy', A)
    np.save('results-graph/similarity_matrix.npy', sim_matrix)
    print("\nEstatísticas do grafo:")
    print(f"Número de cidades (nós): {G.number_of_nodes()}")
    print(f"Número de arestas: {G.number_of_edges()}")
    print(f"Grau médio: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"Grafo é conexo: {nx.is_connected(G)}")
    # Também gera para amostra de 25 cidades
    process_sampled_graph(n_sample=25)
    return G, A, sim_matrix, cidades

if __name__ == "__main__":
    G, A, sim_matrix, cidades = main() 