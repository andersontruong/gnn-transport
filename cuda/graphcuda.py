import cupy
import numpy as np
import random
import networkx as nx
import math
from numba import cuda
from tqdm import tqdm

def generate_random_adjacency_matrix(n_nodes, prob_edge=1.2/(100-1)):
    mask = cupy.random.choice([0.0, 1.0], size=(n_nodes, n_nodes), p=[1-prob_edge, prob_edge])
    weights = cupy.random.rand(n_nodes, n_nodes, dtype=cupy.float32)
    masked_matrix = cupy.multiply(mask, weights)
    
    for src in range(n_nodes - 1):
        dst = random.randint(src + 1, n_nodes - 1)
        masked_matrix[src, dst] = random.random() * 100

    adjacency_matrix = cupy.asnumpy(cupy.triu(masked_matrix, 1) + cupy.tril(masked_matrix.T, -1))

    return adjacency_matrix

def generate_random_graph(n_nodes, prob_edge=1.2/(100-1)):
    return nx.from_numpy_array(generate_random_adjacency_matrix(n_nodes, prob_edge))

@cuda.jit
def _replace_inf(adjacency_matrix):
    x = cuda.grid(1)
    y = x // adjacency_matrix.shape[0]
    x %= adjacency_matrix.shape[0]

    if x >= adjacency_matrix.shape[1] or y >= adjacency_matrix.shape[0]:
        return
    
    if adjacency_matrix[y, x] == 0 and x != y:
        adjacency_matrix[y, x] = np.inf

def _preprocess_fw(adjacency_matrix):
    threads_per_block = 256 # 16x16
    size = adjacency_matrix.size

    dev_matrix = cuda.to_device(adjacency_matrix)

    _replace_inf[math.ceil(size / threads_per_block), threads_per_block](dev_matrix)

    return dev_matrix.copy_to_host()

@cuda.jit
def _floyd_warshall_row(matrix, b, k_block):
    x = cuda.grid(1)
    if x == k_block:
        return
    k_start = k_block*b
    x_start = x*b
    size = matrix.shape[0]

    if x_start >= size or k_start >= size:
        return

    for k in range(b):
        k_pos = k_start + k
        if k_pos >= size:
            return
        for i in range(b):
            ki = k_start + i
            if ki >= size:
                break
            for j in range(b):
                xj = x_start + j
                if xj >= size:
                    break
                matrix[ki, xj] = min(matrix[ki, xj], matrix[ki, k_pos] + matrix[k_pos, xj])

@cuda.jit
def _floyd_warshall_col(matrix, b, k_block):
    y = cuda.grid(1)
    if y == k_block:
        return
    y_start = y*b
    k_start = k_block*b
    size = matrix.shape[0]

    if y_start >= size or k_start >= size:
        return

    for k in range(b):
        k_pos = k_start + k
        if k_pos >= size:
            return
        for i in range(b):
            y_pos = y_start + i
            if y_pos >= size:
                break
            for j in range(b):
                kj = k_start + j
                if kj >= size:
                    break
                matrix[y_pos, kj] = min(matrix[y_pos, kj], matrix[y_pos, k_pos] + matrix[k_pos, kj])

@cuda.jit
def _floyd_warshall_ind(matrix, b, k_block):
    x, y = cuda.grid(2)
    if y == k_block:
        return
    y_start = y*b
    x_start = x*b
    k_start = k_block*b
    size = matrix.shape[0]

    if y_start >= size or x_start >= size or k_start >= size:
        return

    for k in range(b):
        k_pos = k_start + k
        if k_pos >= size:
            return
        for i in range(b):
            yi = y_start + i
            if yi >= size:
                break
            for j in range(b):
                xj = x_start + j
                if xj >= size:
                    break
                matrix[yi, xj] = min(matrix[yi, xj], matrix[yi, k_pos] + matrix[k_pos, xj])

@cuda.jit
def _floyd_warshall_kernel(matrix, b, pos):
    x, y = cuda.grid(2)
    x += pos[0]
    y += pos[1]

    y_start = y*b
    x_start = x*b

    size = matrix.shape[0]

    if y_start >= size or x_start >= size:
        return

    for k in range(b):
        yk = y_start + k
        xk = x_start + k

        if yk >= size or xk >= size:
            return

        for i in range(b):
            yj = y_start + i
            if yj >= size:
                break
            for j in range(b):
                xj = x_start + j
                if xj >= size:
                    break
                matrix[yj, xj] = min(matrix[yj, xj], matrix[yj, xk] + matrix[yk, xj])

def floyd_warshall_gpu(graph, matrix_block_size=4, threads_per_block=256):
    assert graph.shape[0] == graph.shape[1]

    graph = _preprocess_fw(graph)

    # Number of vertices (divisible by matrix_block_size)
    V = graph.shape[0]

    # N matrix blocks (NxN in total)
    matrix_blocks_dim = int(V / matrix_block_size)

    # Blocks per grid for row-wise and column-wise operations
    blocks_per_grid_dim = math.ceil(matrix_blocks_dim / threads_per_block)

    block_dim = math.floor(math.sqrt(threads_per_block))

    square_blocks_per_grid_dim = math.ceil(matrix_blocks_dim / block_dim)

    dev_graph = cuda.to_device(graph)
    for k in tqdm(range(matrix_blocks_dim), position=1, desc='GPU', leave=False, ncols=80):
        _floyd_warshall_kernel[1, 1](dev_graph, matrix_block_size, (k, k))

        # floyd_warshall_cross[blocks_per_grid_dim, (2, threads_per_block)](dev_graph, matrix_block_size, k)

        _floyd_warshall_row[blocks_per_grid_dim, threads_per_block](dev_graph, matrix_block_size, k)
        _floyd_warshall_col[blocks_per_grid_dim, threads_per_block](dev_graph, matrix_block_size, k)

        _floyd_warshall_ind[(square_blocks_per_grid_dim, square_blocks_per_grid_dim), (block_dim, block_dim)](dev_graph, matrix_block_size, k)

    return dev_graph.copy_to_host()