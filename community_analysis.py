import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain
import os
from collections import defaultdict
import re
import matplotlib

def load_existing_data():
    """Load the existing graph and data."""
    # Load adjacency matrix
    A = np.load('results-graph/adjacency_matrix.npy')
    
    # Load original data
    df = pd.read_excel('assaltos.xlsx')
    
    # Create graph from adjacency matrix
    G = nx.from_numpy_array(A)
    
    # Get city names
    cidades = df['municipio'].drop_duplicates().tolist()
    
    # Map node indices to city names
    node_to_city = {i: city for i, city in enumerate(cidades)}
    nx.set_node_attributes(G, node_to_city, 'city')
    
    return G, df

def categorize_crime(description):
    """Categorize crime based on keywords in description."""
    description = description.lower()
    
    categories = {
        'Crimes relacionados a drogas': ['drogas', 'tráfico', 'entrega', 'droga', 'narcotráfico'],
        'Crimes com violência armada': ['arma', 'revólver', 'pistola', 'fuzil', 'tiro', 'disparo'],
        'Furtos em estabelecimentos': ['loja', 'comércio', 'estabelecimento', 'mercado', 'supermercado'],
        'Crimes noturnos': ['noite', 'madrugada', '23h', '22h', '21h', '20h', '19h'],
        'Crimes em residências': ['casa', 'residência', 'apartamento', 'moradia'],
        'Crimes em vias públicas': ['rua', 'avenida', 'praça', 'via pública', 'calçada'],
        'Crimes com veículos': ['carro', 'moto', 'veículo', 'automóvel', 'bicicleta']
    }
    
    for category, keywords in categories.items():
        if any(keyword in description for keyword in keywords):
            return category
    
    return 'Outros crimes'

def analyze_communities(G, df):
    """Analyze communities and their characteristics."""
    # Apply community detection
    communities = community_louvain.best_partition(G)
    
    # Add community labels to graph
    nx.set_node_attributes(G, communities, 'community')
    
    # Create mapping of node indices to cities
    node_to_city = nx.get_node_attributes(G, 'city')
    
    # Analyze each community
    community_analysis = defaultdict(list)
    community_crimes = defaultdict(list)
    
    for node, community_id in communities.items():
        city = node_to_city[node]
        city_crimes = df[df['municipio'] == city]['descricao'].tolist()
        
        # Categorize crimes in this city
        for crime in city_crimes:
            category = categorize_crime(crime)
            community_crimes[community_id].append(category)
    
    # Get most common crime types for each community
    for community_id, crimes in community_crimes.items():
        crime_counts = pd.Series(crimes).value_counts()
        community_analysis[community_id] = {
            'most_common_crimes': crime_counts.head(3).to_dict(),
            'total_crimes': len(crimes)
        }
    
    return communities, community_analysis

def visualize_communities(G, communities):
    """Create visualization of the graph with communities."""
    plt.figure(figsize=(15, 10))
    
    # Get community colors
    community_colors = [communities[node] for node in G.nodes()]
    
    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, 
            node_color=community_colors,
            with_labels=True,
            node_size=1000,
            font_size=8,
            font_weight='bold',
            cmap=plt.cm.rainbow)
    
    plt.title("Comunidades Criminais Identificadas")
    plt.savefig('results-comunity/community_graph.png', dpi=300, bbox_inches='tight')
    plt.close()

def get_community_colors(communities):
    # Gera um dicionário comunidade -> cor (nome)
    unique_communities = sorted(set(communities.values()))
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(unique_communities)-1)
    color_names = []
    for i in range(len(unique_communities)):
        rgba = cmap(norm(i))
        # Converte RGBA para nome aproximado
        color_names.append(matplotlib.colors.to_hex(rgba))
    # Mapeia comunidade para cor hex
    community_to_color = {comm: color_names[i] for i, comm in enumerate(unique_communities)}
    return community_to_color

def save_community_analysis(community_analysis, communities):
    community_to_color = get_community_colors(communities)
    with open('results-comunity/community_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("Análise de Comunidades Criminais\n")
        f.write("==============================\n\n")
        
        for community_id, analysis in community_analysis.items():
            color_hex = community_to_color.get(community_id, "#000000")
            color_name = color_hex  # pode converter para nome se quiser
            f.write(f"Comunidade {community_id} - cor {color_name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total de crimes analisados: {analysis['total_crimes']}\n")
            f.write("Principais tipos de crimes:\n")
            
            for crime_type, count in analysis['most_common_crimes'].items():
                f.write(f"- {crime_type}: {count} ocorrências\n")
            
            f.write("\n")

def main():
    # Create results directory
    os.makedirs('results-comunity', exist_ok=True)
    
    # Load data
    print("Carregando dados existentes...")
    G, df = load_existing_data()
    
    # Analyze communities
    print("Analisando comunidades...")
    communities, community_analysis = analyze_communities(G, df)
    
    # Visualize results
    print("Criando visualizações...")
    visualize_communities(G, communities)
    
    # Save analysis
    print("Salvando resultados...")
    save_community_analysis(community_analysis, communities)
    
    print("Análise concluída! Resultados salvos em 'results-comunity/'")

if __name__ == "__main__":
    main() 