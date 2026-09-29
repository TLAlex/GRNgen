# GRNgen: Relaxed Directed Configuration Model for GRN Generation
GRNgen is a network generator designed to produce synthetic Gene Regulatory Networks (GRNs) that faithfully reproduce the topological features of real-world biological datasets. 
Unlike traditional methods that often fail when faced with strict degree sequences, GRNgen utilizes a Relaxed Directed Configuration Model (RDCM) to ensure successful generation while maintaining structural fidelity.

## How it works:
**Stub Matching with Relaxation**:
For every source vertex, the algorithm satisfies the target out-degree. 
If insufficient eligible targets exist (to avoid self-loops or duplicates), a Relaxation Step increments the in-stub count of a random vertex. 
This guarantees that every out-degree is satisfied, resulting in a valid simple graph.

**Weak Connectivity Enforcement**:
Once all arcs are placed, any remaining isolated components are merged through random arc additions until the graph forms a single weakly connected component.

## Main Features
**Degree constraint:** Takes an input sequence of In-Out degrees ($d^i, d^o$).

**Multi-Property Optimization:** Uses a weighted error function across graph properties (including path-related metrics and motif distributions) to rank the generated ensemble.

**Ensemble Generation:** Generates $n$ random graphs, allowing users to select samples that best represent the target network topology.

## Installation

### Prerequisite

- Python 3.10+
- `pip`

### Installation steps

1. Clone repository :
    ```sh
    git clone https://github.com/TLAlex/GRNgene
    cd GRNgene
    ```

2. Install dependencies
    ```sh
    pip install -r requirements.txt
    ```

3. Install package
    ```sh
    pip install .
    ```

### Examples
#### Graph generation
Raw data of several Gene Regulatory networks are provided. Graphs are stored as .graphml and graph properties were already computed and can be found in data/graphs to facilitate the use of GRNgen.
```python
specie_list = [
    'GSD', 'HSC', 'mCAD', 'VSC', #(Pratapa et al., 2020)
    'yeast','hESC', 'mESC', 'mDC', #(McCalla et al., 2023)
    'ecoli_gnw', #(Schaffter et al., 2011)
    'human_trrust',  #(Han et al., 2018)
    "athaliana_wolf", "dmelanogaster_wolf",  "hsapiens_wolf", "scerevisiae_wolf", "ecoli_wolf" (Wolf et al., 2021)
    ]
```

The following is a snippet demonstrating how to generate random graph ensemble together with some selected topological properties stored as .parquet. 

```python
specie = "human_trrust"
input_dir = f"GRNgen/data/graphs/" # reference graph and associated data folder path
ground_truth_graph, _, _ = load_graphs(input_dir+specie)
node_degree_sequence = get_node_degrees(ground_truth_graph)

ngraphs = 1000 # number of graph to generate
n_jobs = os.cpu_count() - 1

output_dir = f"GRNgen/data/test/" # output folder path for graphs and aggregated data and plots

if not os.path.exists(output_dir): # should be moved toward each plotting function to prevent issues
    os.makedirs(output_dir)

_, _ = generate_random_graphs(
        ngraphs,
        node_degree_sequence
        f"{output_dir}graph_ensemble_test.parquet",
        specie,
        n_jobs=n_jobs,
        connect_type='random',
        method='grngen'
    )
```

### Sensitivity analysis 
An example of sensitivity analysis of GENIE3 is provided in `GRNgen/notebooks/sensi_analysis_example.ipynb`. The experiment settings are intentionnally kept small so that the results fits within the repository.

## References
Han, H., Cho, J. W., Lee, S., Yun, A., Kim, H., Bae, D., ... & Lee, I. (2018). TRRUST v2: an expanded reference database of human and mouse transcriptional regulatory interactions. Nucleic acids research, 46(D1), D380-D386.

McCalla, S. G., Fotuhi Siahpirani, A., Li, J., Pyne, S., Stone, M., Periyasamy, V., ... & Roy, S. (2023). Identifying strengths and weaknesses of methods for computational network inference from single-cell RNA-seq data. G3: Genes, Genomes, Genetics, 13(3), jkad004.

Pratapa, A., Jalihal, A. P., Law, J. N., Bharadwaj, A., & Murali, A. T. (2020). Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data. Nature methods, 17(2), 147-154.

Schaffter, T., Marbach, D., & Floreano, D. (2011). GeneNetWeaver: in silico benchmark generation and performance profiling of network inference methods. Bioinformatics, 27(16), 2263-2270.

Wolf, I.R., Simões, R.P., Valente, G.T.: Three topological features of regulatory networks control life-essential and specialized subsystems. Sci. Rep. 11(1), 24209 (2021)

## If you need to cite
Tan-Lhernould, A., Dorval, T., & Fages, F. (2026, July). GRNgen: A Generator of Gene Regulatory Networks Fitting Graph and Motif Properties. In International Conference on Computational Methods in Systems Biology (pp. 288-307). Cham: Springer Nature Switzerland.