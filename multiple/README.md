# Siamese GNN for Generalizable Betweenness Centrality Estimation

## Dataset Generation
- `V`: number of vertices
- `N`: number of graph pairs

1. Generate `N` graph pairs, each with `V` vertices
    - Use CUDA-accelerated graph generation from `graphcuda`
2. [Graph Alignment with Weisfeiler-Leman (WL) graph isomorphism test](https://direct.mit.edu/netn/article/5/3/711/101836/Network-alignment-and-similarity-reveal-atlas)
    1. Select baseline graph (index 0, original)
    2. Using [WL-align](https://pypi.org/project/wlalign/), Align all other `2N-1` graphs
3. Generate Betweenness Centrality on all `N` graph pairs
    1. Run CUDA-accelerated Floyd-Warshall algorithm from `graphcuda`
    2. Perform distance sum on all paths
4. Generate Graph Feature Embeddings on all `N` graph pairs
    1. Use GRAPE for fast and consistent random walks
    2. Use Word2Vec to learn and generate embeddings

## GNN Training