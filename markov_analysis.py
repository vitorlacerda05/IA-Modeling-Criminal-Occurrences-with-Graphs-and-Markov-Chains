import numpy as np
import matplotlib.pyplot as plt
import random
import os

# 1. Carregar matriz de adjacência e lista de cidades
A = np.load('results-graph/adjacency_matrix.npy')
with open('assaltos.xlsx', 'rb') as f:
    import pandas as pd
    df = pd.read_excel(f)
    cidades = df['municipio'].drop_duplicates().tolist()

# 2. Construção da matriz de transição estocástica
row_sums = A.sum(axis=1, keepdims=True)
P = A / row_sums
assert np.allclose(P.sum(axis=1), 1), 'Alguma linha de P não soma 1!'

# 3. Cálculo da distribuição estacionária (power method)
pi = np.ones(len(P)) / len(P)
for _ in range(1000):
    pi_next = pi @ P
    if np.allclose(pi_next, pi, atol=1e-8):
        break
    pi = pi_next

# 4. Ranking das cidades
ranking_idx = np.argsort(-pi)
ranking_cidades = [(cidades[i], float(pi[i])) for i in ranking_idx]

# 5. Simulação de caminho markoviano
estado_atual = random.choice(range(len(P)))
caminho = [cidades[estado_atual]]
for _ in range(20):
    estado_atual = np.random.choice(range(len(P)), p=P[estado_atual])
    caminho.append(cidades[estado_atual])

def ensure_results_dir():
    os.makedirs('results-markov', exist_ok=True)

# 6. Exportar resultados
ensure_results_dir()
# Tabela
print('Cidade         | Probabilidade (%)')
print('-------------------------------')
for cidade, prob in ranking_cidades[:20]:
    print(f'{cidade:<15} | {prob*100:.2f}')

# Gráfico de barras
plt.figure(figsize=(12,6))
plt.bar([cidades[i] for i in ranking_idx[:20]], [pi[i]*100 for i in ranking_idx[:20]])
plt.xticks(rotation=45, ha='right')
plt.ylabel('Probabilidade Estacionária (%)')
plt.title('Top 20 cidades - Probabilidade de Assaltos (Modelo de Markov)')
plt.tight_layout()
plt.savefig('results-markov/markov_probabilidades.png')
plt.show()

# Caminho simulado
print('\nCaminho simulado (20 passos):')
print(' -> '.join(caminho))

# Salvar outputs principais
np.save('results-markov/markov_stationary.npy', pi)
np.save('results-markov/markov_transition.npy', P)
with open('results-markov/markov_ranking.txt', 'w', encoding='utf-8') as f:
    for cidade, prob in ranking_cidades:
        f.write(f'{cidade}: {prob*100:.2f}\n') 