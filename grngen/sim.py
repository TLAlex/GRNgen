import sys
import os
import json

import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from grngen import *

import numpy as np
import random
from scipy import sparse
import numpy as np
from scipy import sparse
import networkx as nx
from joblib import Parallel, delayed
import h5py
from pathlib import Path
import random
from tqdm import tqdm

def simulate_boolean_async(A, time=50, x0=None, seed=None):
    """
    Asynchronous Boolean simulator.
    Works with both dense and sparse matrices.
    
    At each time step, a random node is selected and updated.

    Rule:
      - If any inhibitor (A[i,j] = -1) is ON -> node j = 0
      - Else if any activator (A[i,j] = +1) is ON -> node j = 1
      - Else -> node j unchanged
    """
    n = A.shape[0]
    is_sparse = sparse.issparse(A)
    
    if is_sparse:
        A = A.tocsc()

    if x0 is None:
        print("x0 has not been provided. It will be automatically generated.")
        rng = np.random.default_rng(seed)
        x0 = rng.integers(0, 2, size=n)

    if len(x0) != n:
        raise ValueError(f"x0 length {len(x0)} does not match network size {n}")

    x = np.array(x0, dtype=int)
    traj = np.zeros((time + 1, n), dtype=int)
    traj[0] = x

    for step in range(1, time + 1):
        target = random.randrange(n)
        
        # Extract column (works for both sparse and dense)
        if is_sparse:
            col = A[:, [target]].toarray().flatten()
        else:
            col = A[:, target]
        
        activators = np.where(col == 1)[0]
        inhibitors = np.where(col == -1)[0]

        # Update rule
        if np.any(x[inhibitors] == 1):
            x[target] = 0
        elif np.any(x[activators] == 1):
            x[target] = 1

        traj[step] = x.copy()

    return traj, x0

def simulate_boolean_async_notraj(A, time=50, x0=None):
    """
    Asynchronous Boolean simulator.
    At each time step, a random node is selected and updated.
    The trajectory is not saved to prevent RAM overload.

    Rule:
      - If any inhibitor (A[i,j] = -1) is ON -> node j = 0
      - Else if any activator (A[i,j] = +1) is ON -> node j = 1
      - Else -> node j unchanged
    """
    n = A.shape[0]
    if x0 is None:
        print("x0 has not beend provided. It will be automatically generated.")
        rng = np.random.default_rng(seed)
        x0 = rng.integers(0, 2, size=n)
    if len(x0) != n:
        raise ValueError(f"x0 length {len(x0)} does not match network size {n}")

    x = np.array(x0, dtype=np.int8)

    for _ in range(1, time + 1):
        target = random.randrange(n)

        activators = np.where(A[:, target] == 1)[0]
        inhibitors = np.where(A[:, target] == -1)[0]

        # Apply logic rule
        if np.any(x[inhibitors] == 1):
            x[target] = 0
        elif np.any(x[activators] == 1):
            x[target] = 1

    return x, x0

def simulate_multi_bool(adj_mx, seed, n_sims, n_step):
    """
    Runs multiple asynchronous Boolean simulations and collects final states.

    Parameters
    ----------
    adj_mx : np.ndarray
        Signed adjacency matrix (n x n) with values in {-1, 0, +1}.
    seed : int
        Seed for the random number generator.
    n_sims : int
        Number of independent simulations.
    n_step : int
        Number of asynchronous update steps per run.

    Returns
    -------
    final_states : np.ndarray, shape (n_sims, n)
        Each row is the final Boolean state of one simulation.
    """
    A = adj_mx.copy()
    n = A.shape[0]
    rng = np.random.default_rng(seed)

    final_states = np.zeros((n_sims, n), dtype=np.int8)

    for run in range(n_sims):
        x0 = rng.integers(0, 2, size=n)
        final_state, _ = simulate_boolean_async_notraj(A, time=n_step, x0=x0)
        final_states[run] = final_state

    print("Final states shape:", final_states.shape)
    return final_states


def plot_traj_heatmap(traj, node_label=False, step_label=False, n_nodes_to_plot=None, node_indices=None):
    """
    Plot trajectory heatmap for Boolean simulation.
    
    Parameters:
    -----------
    traj : np.ndarray
        Trajectory matrix (time_steps x n_nodes)
    node_label : bool
        Show node index labels on y-axis
    step_label : bool
        Show time step labels on x-axis
    n_nodes_to_plot : int, optional
        Number of first nodes to plot (e.g., 10 plots nodes 0-9)
    node_indices : list/array, optional
        Specific node indices to plot (overrides n_nodes_to_plot)
    """
    
    n_total_nodes = traj.shape[1]
    n_time = traj.shape[0]
    
    # Determine which nodes to plot
    if node_indices is not None:
        # Use specific indices provided
        selected_indices = np.array(node_indices)
        traj_subset = traj[:, selected_indices]
    elif n_nodes_to_plot is not None:
        # Use first n nodes
        n_nodes_to_plot = min(n_nodes_to_plot, n_total_nodes)  # Safety check
        selected_indices = np.arange(n_nodes_to_plot)
        traj_subset = traj[:, :n_nodes_to_plot]
    else:
        # Plot all nodes (default behavior)
        selected_indices = np.arange(n_total_nodes)
        traj_subset = traj
    
    n_nodes = traj_subset.shape[1]
    
    # Adjust figure height based on number of nodes
    fig_height = max(3, min(n_nodes * 0.3, 12))
    plt.figure(figsize=(8, fig_height))
    
    plt.imshow(
        traj_subset.T,
        aspect='auto',
        cmap=plt.cm.get_cmap('Greys', 2),
        interpolation='nearest',
        origin='lower'
    )

    plt.xlabel("Time step")
    plt.ylabel("Node index")
    plt.title(f"Asynchronous Boolean Simulation ({n_nodes}/{n_total_nodes} nodes)")

    plt.xticks(range(0, n_time))
    plt.yticks(range(0, n_nodes), labels=selected_indices)  # Show actual node indices

    # Hide tick labels if requested
    if not node_label:
        plt.gca().set_yticklabels([])
    if not step_label:
        plt.gca().set_xticklabels([])

    plt.ylim(-0.5, n_nodes - 0.5)
    plt.xlim(-0.5, n_time - 0.5)

    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(['OFF (0)', 'ON (1)'])

    plt.tight_layout()
    plt.show()
    
import numpy as np
from scipy import sparse

def randomize_edge_signs(A, prob_inhibitor=0.5, seed=None):
    """
    Convert a binary adjacency matrix (0/1) to a signed matrix (-1/0/1).
    
    Parameters:
    -----------
    A : np.ndarray or scipy.sparse matrix
        Binary adjacency matrix with 0s and 1s
    prob_inhibitor : float
        Probability that an edge becomes inhibitory (-1)
        Default 0.5 means 50% activators, 50% inhibitors
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    Signed adjacency matrix with -1 (inhibitor), 0 (no edge), 1 (activator)
    """
    rng = np.random.default_rng(seed)
    is_sparse = sparse.issparse(A)
    
    if is_sparse:
        # Work with sparse matrix
        A_signed = A.tocsr().copy()
        
        # Get number of non-zero elements
        nnz = A_signed.nnz
        
        # Generate random signs: -1 with prob_inhibitor, +1 otherwise
        random_signs = rng.choice(
            [-1, 1], 
            size=nnz, 
            p=[prob_inhibitor, 1 - prob_inhibitor]
        )
        
        # Apply signs to non-zero data
        A_signed.data = A_signed.data * random_signs
        
        return A_signed
    
    else:
        # Work with dense matrix
        A_signed = A.copy().astype(int)
        
        # Find edges (non-zero entries)
        edges = np.where(A == 1)
        n_edges = len(edges[0])
        
        # Generate random signs
        random_signs = rng.choice(
            [-1, 1], 
            size=n_edges, 
            p=[prob_inhibitor, 1 - prob_inhibitor]
        )
        
        # Apply signs
        A_signed[edges] = random_signs
        
        return A_signed
    
import numpy as np

def count_flipping_ratio(traj):
    """
    Count the ratio of nodes that flip vs those that don't.
    
    Parameters:
    -----------
    traj : np.ndarray
        Trajectory matrix (time_steps x n_nodes)
    
    Returns:
    --------
    dict with flipping analysis
    """
    n_time, n_nodes = traj.shape
    
    # Count changes per node (any non-zero diff means a flip)
    changes_per_node = np.sum(np.diff(traj, axis=0) != 0, axis=0)
    
    # Nodes that flipped at least once
    flipping_mask = changes_per_node > 0
    n_flipping = np.sum(flipping_mask)
    n_static = n_nodes - n_flipping
    
    return {
        'n_flipping': n_flipping,
        'n_static': n_static,
        'ratio_flipping': n_flipping / n_nodes,
        'ratio_static': n_static / n_nodes,
        'flipping_indices': np.where(flipping_mask)[0],
        'static_indices': np.where(~flipping_mask)[0]
    }

def get_flipping_distribution(traj):
    """
    Get the distribution of how many times each node flips.
    
    Parameters:
    -----------
    traj : np.ndarray
        Trajectory matrix (time_steps x n_nodes)
    
    Returns:
    --------
    dict with flipping distribution analysis
    """
    n_time, n_nodes = traj.shape
    
    # Count changes per node (number of flips for each node)
    flips_per_node = np.sum(np.diff(traj, axis=0) != 0, axis=0)
    
    # Get unique flip counts and their frequencies
    unique_flips, counts = np.unique(flips_per_node, return_counts=True)
    
    # Create a distribution dictionary: {number_of_flips: number_of_nodes}
    flip_distribution = dict(zip(unique_flips, counts))
    
    return {
        'flips_per_node': flips_per_node,           # Array: flips for each node
        'flip_distribution': flip_distribution,      # Dict: {x flips: n nodes}
        'unique_flip_counts': unique_flips,          # Unique flip values
        'node_counts': counts,                       # How many nodes for each flip count
        'max_flips': np.max(flips_per_node),
        'mean_flips': np.mean(flips_per_node),
        'median_flips': np.median(flips_per_node),
        'std_flips': np.std(flips_per_node)
    }

def get_steady_state_time(traj):
    """
    Find the time step at which the system reaches steady state
    (i.e., no more changes in any node).
    
    Parameters:
    -----------
    traj : np.ndarray
        Trajectory matrix (time_steps x n_nodes)
    
    Returns:
    --------
    dict with steady state analysis
    """
    n_time, n_nodes = traj.shape
    
    # Compute differences between consecutive time steps
    diffs = np.diff(traj, axis=0)  # Shape: (n_time-1, n_nodes)
    
    # Check if ANY node changed at each time step
    any_change = np.any(diffs != 0, axis=1)  # Shape: (n_time-1,)
    
    # Find the last time step where a change occurred
    change_indices = np.where(any_change)[0]
    
    if len(change_indices) == 0:
        # No changes at all - steady state from the beginning
        steady_state_time = 0
        reached_steady_state = True
    else:
        last_change_time = change_indices[-1]
        # Steady state is reached at the step AFTER the last change
        steady_state_time = last_change_time + 1
        # Check if we actually reached steady state (last change wasn't at the end)
        reached_steady_state = last_change_time < (n_time - 2)
    
    return {
        'steady_state_time': steady_state_time,
        'reached_steady_state': reached_steady_state,
        'total_time': n_time,
        'fraction_to_steady': steady_state_time / n_time,
        'n_changes_total': np.sum(any_change),
        'last_change_time': change_indices[-1] if len(change_indices) > 0 else None
    }

import numpy as np
from pathlib import Path

def save_traces(filepath, traces_list, x0_list, adjacency_matrix, metadata=None):
    """
    Save multiple Boolean traces to a compressed NPZ file.
    
    Parameters
    ----------
    filepath : str or Path
        Output file path (will add .npz extension if missing)
    traces_list : list of np.ndarray
        List of trajectory arrays, each shape (time+1, n)
    x0_list : list of np.ndarray
        List of initial conditions corresponding to each trace
    adjacency_matrix : np.ndarray or sparse matrix
        The adjacency matrix used for simulation
    metadata : dict, optional
        Additional metadata (time, seed, etc.)
    """
    from scipy import sparse
    
    # Stack traces into 3D array: (num_traces, time+1, n)
    traces_array = np.stack(traces_list, axis=0).astype(np.int8)  # int8 saves space
    x0_array = np.stack(x0_list, axis=0).astype(np.int8)
    
    # Handle sparse adjacency matrix
    if sparse.issparse(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.toarray()
    
    # Prepare save dict
    save_dict = {
        'traces': traces_array,
        'x0': x0_array,
        'adjacency_matrix': adjacency_matrix.astype(np.int8),
    }
    
    # Store metadata as a serialized string (NPZ limitation)
    if metadata:
        import json
        save_dict['metadata'] = np.array([json.dumps(metadata)])
    
    np.savez_compressed(filepath, **save_dict)
    print(f"Saved {len(traces_list)} traces to {filepath}")


def load_traces(filepath):
    """
    Load traces from NPZ file.
    
    Returns
    -------
    dict with keys: 'traces', 'x0', 'adjacency_matrix', 'metadata'
    """
    import json
    
    data = np.load(filepath, allow_pickle=False)
    
    result = {
        'traces': data['traces'],
        'x0': data['x0'],
        'adjacency_matrix': data['adjacency_matrix'],
    }
    
    if 'metadata' in data:
        result['metadata'] = json.loads(str(data['metadata'][0]))
    else:
        result['metadata'] = {}
    
    return result

def simulate_single_run(adj_mx, n_steps, seed):
    """
    Run a single Boolean simulation.
    
    Parameters
    ----------
    adj_mx : np.ndarray
        Dense signed adjacency matrix.
    n_steps : int
        Number of asynchronous update steps.
    seed : int
        Seed for this specific run (used for x0 generation).
    
    Returns
    -------
    final_state : np.ndarray
        Final Boolean state.
    x0 : np.ndarray
        Initial state.
    seed : int
        The seed used (for tracking).
    """
    rng = np.random.default_rng(seed)
    n = adj_mx.shape[0]
    x0 = rng.integers(0, 2, size=n)
    
    final_state, _ = simulate_boolean_async_notraj(adj_mx, time=n_steps, x0=x0)
    
    return final_state, x0, seed


def simulate_multi_bool_parallel(adj_mx, base_seed, n_sims, n_steps, n_jobs=-1):
    """
    Parallel version of simulate_multi_bool.
    
    Parameters
    ----------
    adj_mx : np.ndarray
        Signed adjacency matrix (dense).
    base_seed : int
        Base seed - each run gets base_seed + run_index.
    n_sims : int
        Number of independent simulations.
    n_steps : int
        Number of asynchronous update steps per run.
    n_jobs : int
        Number of parallel workers (-1 = all cores).
    
    Returns
    -------
    final_states : np.ndarray, shape (n_sims, n)
    x0_states : np.ndarray, shape (n_sims, n)
    """
    # Run in parallel
    results = Parallel(n_jobs=n_jobs, verbose=False)(
        delayed(simulate_single_run)(adj_mx, n_steps, seed=base_seed + i)
        for i in range(n_sims)
    )
    
    # Unpack and stack results
    final_states = np.stack([r[0] for r in results])
    x0_states = np.stack([r[1] for r in results])
    
    #print(f"Final states shape: {final_states.shape}")
    return final_states, x0_states