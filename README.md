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

### Main Features
**Customizable Topology:** Takes an input sequence of In-Out degrees ($d^i, d^o$) to guide the generation.
**Multi-Property Optimization:** Uses a weighted error function across graph properties (including path-related metrics and motif distributions) to rank the generated ensemble.
**Ensemble Generation:** Generates $n$ random graphs, allowing users to select samples that best represent the target network topology.
**Analysis Tools:** Normalizes properties, generates comparison plots, and identifies the Top-N best-fitting graphs.

# Installation
*In progress*

# Examples
*In progress*