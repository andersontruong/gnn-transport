import grape
import cupy
import numpy as np
import networkx as nx
import random
import os
from gensim.models import Word2Vec
from graphcuda import generate_random_adjacency_matrix, floyd_warshall_gpu, faq_align

def damage_graph(A: np.array, max_deleted_edges=10):
    assert A.shape[0] == A.shape[1]

    result = cupy.copy(A)

    # Upper triangle # entries: N(N−1)/2
    edges = cupy.nonzero(cupy.triu(A, 1))
    edges = list(map(cupy.asnumpy, edges))
    random_indices = np.random.choice(edges[0].shape[0], random.randint(0, max_deleted_edges), replace=False)

    for i in random_indices:
        x = edges[0][i]
        y = edges[1][i]
        
        temp = cupy.copy(result)
        temp[y, x] = 0
        temp[x, y] = 0
        G = nx.from_numpy_array(temp)
        if nx.is_connected(G):
            result[y, x] = 1000000000
            result[x, y] = 1000000000
    
    return result

def get_embedding(G: np.array, walk_len=100, num_walks=10, dimension_size=128, p=1, q=2):
    folder_path = os.path.join(os.getcwd(), 'tmp')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    temp_file_path = os.path.join(folder_path, 'tmp.tsv')
    with open(temp_file_path, 'w') as f:
        # Write edgelist
        for i, row in enumerate(G[:-1]):
            for j, col in enumerate(row[i+1:]):
                if col > 0:
                    f.write(f'{i}\t{j + i + 1}\t{col}\n')
    
    # GRAPE Model
    grape_model = grape.Graph.from_csv(
        # Edges related parameters

        ## The path to the edges list tsv
        edge_path=temp_file_path,
        ## Set the tab as the separator between values
        edge_list_separator="\t",
        ## The first rows should NOT be used as the columns names
        edge_list_header=False,
        ## The source nodes are in the first nodes
        sources_column_number=0,
        ## The destination nodes are in the second column
        destinations_column_number=1,
        ## Both source and destinations columns use numeric node_ids instead of node names
        edge_list_numeric_node_ids=True,
        ## The weights are in the third column
        weights_column_number=2,

        # Graph related parameters
        ## The graph is undirected
        directed=False,
        ## The name of the graph is HomoSapiens
        name="Temp Grape Graph",
        ## Display a progress bar, (this might be in the terminal and not in the notebook)
        verbose=True,
    )
    walks = grape_model.complete_walks(
        walk_length=walk_len,
        iterations=num_walks,
        return_weight=p, # p
        explore_weight=q # q
    )

    grape_word2vec = Word2Vec(
        walks.tolist(),
        vector_size=dimension_size,
        window=5,
        min_count=0,
        sg=1,
        workers=16,
        epochs=10,
        seed=123
    )
    
    return grape_word2vec.wv.vectors

def distance_sum(G: np.ndarray):
    fw = floyd_warshall_gpu(G)
    return cupy.sum(fw)
    
class SGNNSample:
    def __init__(self, n_nodes: int, prob_edge: float, max_deleted_edges: int = 10, walk_len: int = 100, num_walks: int = 10, dimension_size: int = 128, p: int = 1, q: int = 2, seed: int = 42, max_iter_align: int = 30):

        self.seed = seed
        self.max_deleted_edges = max_deleted_edges

        self.G_base = generate_random_adjacency_matrix(n_nodes, prob_edge)
        self.G_modified = damage_graph(self.G_base, max_deleted_edges=max_deleted_edges)
        self.G_modified = faq_align(self.G_base, self.G_modified, seed=seed, max_iter=max_iter_align)

        self.embedding_base = get_embedding(self.G_base, walk_len=walk_len, num_walks=num_walks, dimension_size=dimension_size, p=p, q=q)
        self.embedding_modified = get_embedding(self.G_modified, walk_len=walk_len, num_walks=num_walks, dimension_size=dimension_size, p=p, q=q)

        self.centrality_base = distance_sum(self.G_base)
        self.centrality_modified = distance_sum(self.G_modified)