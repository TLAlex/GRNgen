# =========================================================
# Imports
# =========================================================
# -----------------------
# Standard library
# -----------------------
import json
import math
import random
from collections import Counter

# -----------------------
# Third-party libraries
# -----------------------
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt
from tqdm import tqdm 
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------
# Local modules
# -----------------------
from process_data import *

# =========================================================
# Connectivity utilities
# =========================================================
def connect_components(G, reference_graph=None, method="random"):

    if method == "random":
        return connect_components_random(G)

    if method == "min":
        return connect_components_min_diameter(G)

    if method == "undir":
        if reference_graph is None:
            raise ValueError("reference_graph required for 'undir'")
        return connect_components_adj_diameter(G, reference_graph)

    if method == "dir":
        if reference_graph is None:
            raise ValueError("reference_graph required for 'dir'")
        return connect_components_adj_dir_diameter(G, reference_graph)

    raise ValueError(f"Unknown connection method: {method}")

def connect_components_min_diameter(G):
    """
    Connect all weakly connected components to the largest one
    by adding edges. Does NOT preserve degrees.
    Guarantees that the diameter of the LCC does not increase.
    """

    # Make sure we work on directed graph
    if not isinstance(G, nx.DiGraph):
        raise ValueError("Graph must be a directed graph.")

    while True:
        comps = list(nx.weakly_connected_components(G))
        if len(comps) == 1:
            return G  # connected

        # Largest component
        LCC = max(comps, key=len)
        UG = G.subgraph(LCC).to_undirected()

        # Compute center of LCC
        ecc = nx.eccentricity(UG)
        min_e = min(ecc.values())
        centers = [n for n, e in ecc.items() if e == min_e]
        center = random.choice(centers)

        # Connect every other component one by one
        for C in comps:
            if C is LCC:
                continue

            # pick arbitrary node in C
            target = random.choice(list(C))

            # add an edge (center target)
            # direction doesn't matter for weak connectivity
            G.add_edge(center, target)

def connect_components_random(G):
    """
    Connect all weakly connected components to the largest one
    by randomly selecting a node from each component.
    """

    if not isinstance(G, nx.DiGraph):
        raise ValueError("Graph must be a directed graph.")

    comps = list(nx.weakly_connected_components(G))

    if len(comps) <= 1:
        return G

    LCC = max(comps, key=len)
    
    for C in comps:
        if C is not LCC:
            source = random.choice(list(LCC))
            target = random.choice(list(C))
            G.add_edge(source, target)

    return G

def connect_components_adj_diameter(G, reference_graph, max_trials=100):
    """
    Connect weakly connected components by adding edges, trying to match
    the reference graph's undirected diameter.
    Does NOT preserve degrees.
    """

    D_ref_undir = nx.diameter(reference_graph.to_undirected())

    #pbar = tqdm(desc="Connecting components")

    while True:
        comps = list(nx.weakly_connected_components(G))
        if len(comps) == 1:
            break

        #pbar.update(1)

        LCC = max(comps, key=len)
        G_LCC = G.subgraph(LCC)
        UG_LCC = G_LCC.to_undirected()
        ecc_LCC_undir = nx.eccentricity(UG_LCC)

        C = random.choice([c for c in comps if c != LCC])
        G_C = G.subgraph(C)
        UG_C = G_C.to_undirected()
        ecc_C_undir = nx.eccentricity(UG_C)

        rad_C = min(ecc_C_undir.values())
        # centers_C = [n for n, e in ecc_C_undir.items() if e == rad_C]
        
        min_LCC = min(ecc_LCC_undir.values())
        if min_LCC + 1 + rad_C > D_ref_undir:
            # impossible to connect C under the constraint
            continue

        for _ in range(max_trials):
            u = random.choice(list(LCC)) # pick random node in LCC
            target = random.choice(list(C))
            new_diam = max(
                max(ecc_LCC_undir.values()), # diameter is equal either to max eccentricity
                ecc_LCC_undir[u] + 1 + rad_C # or the eccentricity of the selected target + the added edge + the radius of the component
            )

            if new_diam <= D_ref_undir: # if the new diam is still smaller connect
                G.add_edge(u, target)
                break
            else: # otherwise retry another node
                continue

    #pbar.close()
    return G


def connect_components_adj_dir_diameter(G, reference_graph):
    """
    Connect weakly connected components by adding edges, trying to match
    the reference graph's directed diameter.
    Does NOT preserve degrees.
    Uses incremental directed-diameter estimation.
    """

    if not isinstance(G, nx.DiGraph):
        raise ValueError("Graph must be a directed graph.")
    # Compute diameter once. Fitting procedure will evolve with local changes
    _, diameter_dir_ref = directed_path_stats(reference_graph)
    _, D_current = directed_path_stats(G)
    # print(f'Ref diameter: {diameter_dir_ref}')
    # print(f'LCC diameter: {D_current}')

    while True:
        comps = list(nx.weakly_connected_components(G))
        # print(f'There are currently {len(comps)} remaining.')
        if len(comps) == 1:
            return G
        LCC = max(comps, key=len)
        G_LCC = G.subgraph(LCC)
        UG_LCC = G_LCC.to_undirected()
        ecc_LCC_undir = nx.eccentricity(UG_LCC) # compute eccentricity instead of diameter

        ### estimate diameter with out eccentricity. Double check the method ######################
        ecc_LCC_out = {
            u: max(nx.single_source_shortest_path_length(G_LCC, u).values())
            for u in G_LCC.nodes()
        }

        C = random.choice([c for c in comps if c is not LCC])
        # print(f'Connecting LCC of size {len(LCC)} and CC of size {len(C)}.')
        G_C = G.subgraph(C)
        UG_C = G_C.to_undirected()
        ecc_C_undir = nx.eccentricity(UG_C)

        ### estimate diameter with out eccentricity. Double check the method ######################
        ecc_C_out = {
            u: max(nx.single_source_shortest_path_length(G_C, u).values())
            for u in G_C.nodes()
        }

        rad_C = min(ecc_C_undir.values())
        centers_C = [n for n, e in ecc_C_undir.items() if e == rad_C]
        target = random.choice(centers_C)

        D_ref_undir = nx.diameter(reference_graph.to_undirected())
        desired_ecc = max(0, D_ref_undir - 1 - rad_C)
        center = min(
            ecc_LCC_undir,
            key=lambda n: abs(ecc_LCC_undir[n] - desired_ecc)
        )
        
        # estimate diameter change with upperbound change without graph mutation
        # Case 1: center -> target
        D_est_forward = max(
            D_current,
            ecc_LCC_out[center] + 1 + ecc_C_out[target]
        )

        # Case 2: target -> center
        D_est_reverse = max(
            D_current,
            ecc_C_out[target] + 1 + ecc_LCC_out[center]
        )

        # Select best direction
        if abs(D_est_forward - diameter_dir_ref) <= abs(D_est_reverse - diameter_dir_ref):
            G.add_edge(center, target)
            D_current = D_est_forward
        else:
            G.add_edge(target, center)
            D_current = D_est_reverse
        print(f'Forward changes: {abs(D_est_forward - diameter_dir_ref)}')
        print(f'Reverse changes: {abs(D_est_reverse - diameter_dir_ref)}')

# =========================================================
# Graph generation algorithms
# =========================================================

def validate_degree_sequences(in_degree, out_degree, reference_graph=None, connect_type='random'):
    if len(in_degree) != len(out_degree):
        raise ValueError("In-degree and out-degree sequences must have same length.")

    if sum(in_degree) != sum(out_degree):
        raise ValueError("Sum of in-degrees must equal sum of out-degrees.")


def random_config_graph(in_degree, out_degree, connect_type='random'):
    # Check degree coherence
    validate_degree_sequences(in_degree, out_degree)
    
    # Init
    N = len(in_degree)
    
    ## Init degree sequence that will be updated
    in_ = in_degree.copy()
    out_ = out_degree.copy()
    
    ## Init available node with >0 out degrees
    out_nodes = [i for i, d in enumerate(out_) if d > 0]

    ## Init the graph with nodes
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    
    while out_nodes:
        # Find max-degree among remaining nodes
        max_deg = max(out_[u] for u in out_nodes)
        candidates = [u for u in out_nodes if out_[u] == max_deg]
        u = random.choice(candidates)

        k = out_[u]  # number of outgoing edges to assign from u

        # Find targets for u
        feasible_targets = [
            v for v in range(N)
            if v != u                    # no self-loop
            and in_[v] > 0            # target must have remaining inbound demand
            and not G.has_edge(u, v)     # avoid parallel edges
        ]

        if len(feasible_targets) < k:
            raise ValueError("Degree sequence is not graphical (VL failed: insufficient targets).")

        # 3. Randomly choose k feasible targets
        chosen_targets = random.sample(feasible_targets, k)

        # 4. Add the edges and update degrees
        for v in chosen_targets:
            G.add_edge(u, v)
            in_[v] -= 1

        out_[u] = 0  # u is now satisfied

        # 5. Update the list of nodes with remaining out-degree
        out_nodes = [i for i, d in enumerate(out_) if d > 0]

    return connect_components(G, reference_graph, connect_type)


def random_config_graph_igraph(in_degree, out_degree, reference_graph=None, connect_type='random'):
    # Check degree coherence
    validate_degree_sequences(in_degree, out_degree)
    
    N = len(in_degree)
    
    # Create a directed graph using igraph's degree sequence function
    g = ig.Graph.Degree_Sequence(in_degree, out_degree, method="simple")
    G = nx.DiGraph(g.get_edgelist()) # convert to a NetworkX graph for compatibility

    return connect_components(G, reference_graph, connect_type)



def random_config_graph_degree_relax(in_degree, out_degree, reference_graph=None, connect_type='random'):
    # Check degree coherence
    validate_degree_sequences(in_degree, out_degree)
    
    N = len(in_degree)

    # Init degree sequence that will be updated
    in_ = in_degree.copy()
    out_ = out_degree.copy()
    out_nodes = [i for i, d in enumerate(out_) if d > 0] # filter nodes with out-edges

    # Init the graph with disconnected nodes
    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    i=0 #debugging iteration counter
    failcount = 0 #debugging fail counter

    while out_nodes:
        u = random.choice(out_nodes) # get the index of a random out node

        k = out_[u]  # number of outgoing edges to assign from u

        # Find targets for u
        feasible_targets = [
            v for v in range(N)
            if v != u                    # no self-loop
            and in_[v] > 0            # target must have remaining inbound demand
            and not G.has_edge(u, v)     # avoid parallel edges
        ]

        while len(feasible_targets) < k:
            failcount += 1
            # print(
            #     f"Greedy failure at step {i}: node {u} "
            #     f"needs {k} targets, only {len(feasible_targets)} available,"
            #     f"global in-degree relaxation."
            # )

            # Select ANY node except u
            candidates = [v for v in range(N) if v != u]

            if not candidates:
                raise RuntimeError("No nodes available for in-degree relaxation ? Check this part for correct implementation.")

            v = random.choice(candidates)
            in_[v] += 1  # inject inbound capacity

            # Recompute feasible targets
            feasible_targets = [
                w for w in range(N)
                if w != u
                and in_[w] > 0
                and not G.has_edge(u, w)
            ]

        chosen_targets = random.sample(feasible_targets, k) # randomly choose k feasible targets

        # add the edges and update degrees
        for v in chosen_targets:
            G.add_edge(u, v)
            in_[v] -= 1
        out_[u] = 0  # u is now satisfied

        # update the list of nodes with remaining out-degree
        out_nodes.remove(u)
        i+=1
    # print('Random graph generated. Now connecting isolated components...')
    return connect_components(G, reference_graph, connect_type), failcount



import inspect

# =========================================================
# Experiment helpers
# =========================================================

GENERATORS = {
    "c_cm": random_config_graph_degree_relax,
}


def call_generator(generator, args_dict):
    """Call generator while passing only supported kwargs."""
    sig = inspect.signature(generator)
    valid_args = {
        k: v for k, v in args_dict.items()
        if k in sig.parameters
    }
    return generator(**valid_args)

# Pipeline with failure tracking
def generate_one_graph(
    i,
    ground_truth_in_degree,
    ground_truth_out_degree,
    out_in_pairs,
    out_out_pairs,
    ground_truth_graph,
    output_dir,
    specie,
    connect_type="random",
    method="c_cm",
    N=None,
    check_failures=True, # Default to True to capture the data
    save_graphs=True
):
    generator = GENERATORS[method]

    args = {
        "in_degree": ground_truth_in_degree,
        "out_degree": ground_truth_out_degree,
        "pairs": out_in_pairs,           
        "out_in_pairs": out_in_pairs,    
        "out_out_pairs": out_out_pairs,  
        "reference_graph": ground_truth_graph,
        "connect_type": connect_type,
        "N": N,
    }

    failcounts = 0
    try:
        if check_failures:
            random_graph, failcounts = call_generator(generator, args)
        else:
            random_graph, _ = call_generator(generator, args)

        if save_graphs:
            graph_path = f"{output_dir}/{specie}_random{i}.graphml"
            nx.write_graphml(random_graph, graph_path)

        random_stats = compute_network_properties(random_graph)
        random_motifs = count_motifs(random_graph)
        
        # Return failcounts as the "error/metadata" field
        return i, random_stats, random_motifs, failcounts

    except Exception as e:
        print(f"Error in graph {i}: {e}")
        return i, None, None, str(e)

def generate_random_graphs(
    ngraphs,
    ground_truth_in_degree,
    ground_truth_out_degree,
    out_in_pairs,
    out_out_pairs,
    ground_truth_graph,
    output_dir,
    specie,
    parallel=True,
    n_jobs=None,
    connect_type='random',
    method='c_cm',
    N=None,
    save_graphs=True
):
    stat_results = []
    motif_results = []
    failure_results = [] # New list for tracking failures

    if parallel:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    generate_one_graph,
                    i,
                    ground_truth_in_degree,
                    ground_truth_out_degree,
                    out_in_pairs,
                    out_out_pairs,
                    ground_truth_graph,
                    output_dir,
                    specie,
                    connect_type,
                    method,
                    N,
                    True, # check_failures,
                    save_graphs
                )
                for i in range(ngraphs)
            ]

            for future in tqdm(as_completed(futures),
                               total=ngraphs,
                               desc="Generating random graphs"):
                i, stats, motifs, fail_data = future.result()

                # If fail_data is a string, it's an error message
                if isinstance(fail_data, str):
                    print(f"[{i}] Graph could not be generated: {fail_data}")
                    continue
                stats['graph_id'] = i
                motifs['graph_id'] = i
                stat_results.append(stats)
                motif_results.append(motifs)
                failure_results.append({"graph_id": i, "failure_count": fail_data})

    else:
        for i in tqdm(range(ngraphs), desc="Generating random graphs"):
            i, stats, motifs, fail_data = generate_one_graph(
                i,
                ground_truth_in_degree,
                ground_truth_out_degree,
                out_in_pairs,
                out_out_pairs,
                ground_truth_graph,
                output_dir,
                specie,
                connect_type,
                method,
                N,
                True
            )

            if isinstance(fail_data, str):
                print(f"[{i}] Graph could not be generated: {fail_data}")
                continue
            stats['graph_id'] = i
            motifs['graph_id'] = i
            stat_results.append(stats)
            motif_results.append(motifs)
            failure_results.append({"graph_id": i, "failure_count": fail_data})

    # Saving properties and motifs
    df_stats = pd.DataFrame(stat_results)
    df_motifs = pd.DataFrame(motif_results).astype("int32")
    df_stats.to_csv(f"{output_dir}/random_properties.csv", index=False)
    df_motifs.to_csv(f"{output_dir}/random_motifs.csv", index=False)

    df_failures = pd.DataFrame(failure_results)
    df_failures.to_csv(f"{output_dir}/random_failures.csv", index=False)

    print(f"Pipeline completed successfully. Average failures per graph: {df_failures['failure_count'].mean():.2f}")

    return df_stats, df_motifs
