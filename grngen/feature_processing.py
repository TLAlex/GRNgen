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

import pandas as pd
import numpy as np
from scipy.stats import entropy, median_abs_deviation

def norm_min_max(df):
    df_min_max = df.copy()
    for col in df.columns:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > col_min:
            df_min_max[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df_min_max[col] = 0
    return df_min_max

def norm_mad(df):
    df_mad = df.copy()
    for col in df.columns:
        df_mad[col] = median_abs_deviation(col)
    return df_mad

def select_optimal_graphs(df, n_select, target_feature=None, feature_cols=None, alpha=1.0, beta=1.0, gamma=1.0, stat='var'):
    """
    Select n graphs optimizing variance constraints.
    
    Parameters:
    -----------
    df : DataFrame with graph properties (includes 'graph_id')
    n_select : number of graphs to select
    target_feature : column to maximize variance (auto-detected if None)
    alpha : weight for maximizing target column variance
    beta : weight for minimizing other columns variance
    gamma : weight for uniform distribution
    
    Returns:
    --------
    Selected graph IDs and the filtered DataFrame
    """
    ############### Categorize features #######################
    # Remove graph_id from feature columns
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != 'graph_id']
    
    # Auto-detect highest variance column if not provided
    if target_feature is None:
        if stat == 'var':
            target_feature = df[feature_cols].var().idxmax()
            print(f"Auto-detected highest variance column: {target_feature}")
        elif stat == 'std':
            target_feature = df[feature_cols].std().idxmax()
            print(f"Auto-detected highest standard deviation column: {target_feature}")
    
    # Get features to minimize spread
    other_cols = [col for col in feature_cols if col != target_feature]
    
    ############# Preprocessing ##############################    
    # Min-max normalization
    df_normalized = df.copy()
    for col in feature_cols:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > col_min:
            df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df_normalized[col] = 0
    
    # Create bins for uniform distribution check
    n_bins = min(n_select, 10)
    df_normalized['bin'] = pd.cut(
        df_normalized[target_feature], 
        bins=n_bins, 
        labels=False, 
        include_lowest=True
    )
    
    ############ Graph selection ###########################
    selected_indices = []
    remaining_indices = list(df.index)
    
    for i in tqdm(range(n_select), desc="Selecting graphs"):
        best_score = -np.inf
        best_idx = None
        
        for idx in remaining_indices:
            # Temporarily add this graph
            temp_selected = selected_indices + [idx]
            temp_df = df_normalized.loc[temp_selected]
            
            # Score 1: Variance of target column (MAXIMIZE)
            if stat == 'var':
                target_var = temp_df[target_feature].var() if len(temp_selected) > 1 else 0
            elif stat == 'std':
                target_var = temp_df[target_feature].std() if len(temp_selected) > 1 else 0
            
            # Score 2: Mean variance of other columns (MINIMIZE)
            if stat == 'var':
                other_var = temp_df[other_cols].var().mean() if len(temp_selected) > 1 else 0
            elif stat == 'std':
                other_var = temp_df[other_cols].std().mean() if len(temp_selected) > 1 else 0
            
            # Score 3: Uniformity of distribution (MAXIMIZE entropy)
            bin_counts = temp_df['bin'].value_counts()
            bin_probs = bin_counts / bin_counts.sum()
            uniformity = entropy(bin_probs) / np.log(n_bins)  # Normalized entropy [0,1]
            
            # Combined score
            score = (alpha * target_var) - (beta * other_var) + (gamma * uniformity)
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
    selected_df = df.loc[selected_indices]
    
    return selected_df['graph_id'].tolist(), selected_df, df_normalized

def select_optimal_graphs_fast(df, n_select, target_feature=None, feature_cols=None, alpha=1.0, beta=1.0, gamma=1.0, stat='var'):
    """
    Select n graphs optimizing variance constraints.
    
    Parameters:
    -----------
    df : DataFrame with graph properties (includes 'graph_id')
    n_select : number of graphs to select
    target_feature : column to maximize variance (auto-detected if None)
    alpha : weight for maximizing target column variance
    beta : weight for minimizing other columns variance
    gamma : weight for uniform distribution
    stat : 'var' for variance, 'std' for standard deviation
    
    Returns:
    --------
    Selected graph IDs and the filtered DataFrame
    """
    ############### Categorize features #######################
    # Remove graph_id from feature columns
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != 'graph_id']
    
    # Auto-detect highest variance column if not provided
    if target_feature is None:
        if stat == 'var':
            target_feature = df[feature_cols].var().idxmax()
            print(f"Auto-detected highest variance column: {target_feature}")
        elif stat == 'std':
            target_feature = df[feature_cols].std().idxmax()
            print(f"Auto-detected highest standard deviation column: {target_feature}")
    
    # Get features to minimize spread
    other_cols = [col for col in feature_cols if col != target_feature]
    
    ############# Preprocessing ##############################    
    # Min-max normalization
    df_normalized = df.copy()
    for col in feature_cols:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > col_min:
            df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df_normalized[col] = 0
    
    # Create bins for uniform distribution check
    n_bins = min(n_select, 10)
    df_normalized['bin'] = pd.cut(
        df_normalized[target_feature], 
        bins=n_bins, 
        labels=False, 
        include_lowest=True
    )
    
    # Pre-extract values as numpy arrays for faster access
    target_values = df_normalized[target_feature].values
    other_values = df_normalized[other_cols].values
    bin_values = df_normalized['bin'].values
    index_to_pos = {idx: i for i, idx in enumerate(df_normalized.index)}
    
    ############ Graph selection ###########################
    selected_indices = []
    remaining_indices = set(df.index)
    
    # Running statistics (Welford's algorithm for numerical stability)
    n_selected = 0
    target_mean = 0.0
    target_m2 = 0.0  # Sum of squared differences from mean
    other_means = np.zeros(len(other_cols))
    other_m2 = np.zeros(len(other_cols))
    bin_counts = np.zeros(n_bins, dtype=int)
    
    # Helper function to convert variance to std if needed
    def apply_stat(variance):
        if stat == 'std':
            return np.sqrt(variance) if np.isscalar(variance) else np.sqrt(variance)
        return variance
    
    for i in tqdm(range(n_select), desc="Selecting graphs"):
        best_score = -np.inf
        best_idx = None
        
        for idx in remaining_indices:
            pos = index_to_pos[idx]
            
            # Simulate adding this point using Welford's online algorithm
            new_n = n_selected + 1
            
            # Target column variance (incremental)
            new_target_val = target_values[pos]
            delta_target = new_target_val - target_mean
            new_target_mean = target_mean + delta_target / new_n
            new_target_m2 = target_m2 + delta_target * (new_target_val - new_target_mean)
            target_var = new_target_m2 / new_n if new_n > 1 else 0
            target_stat = apply_stat(target_var)
            
            # Other columns variance (incremental)
            new_other_vals = other_values[pos]
            delta_other = new_other_vals - other_means
            new_other_means = other_means + delta_other / new_n
            new_other_m2 = other_m2 + delta_other * (new_other_vals - new_other_means)
            other_vars = new_other_m2 / new_n if new_n > 1 else np.zeros(len(other_cols))
            other_stat = apply_stat(other_vars).mean()
            
            # Uniformity (incremental bin counts)
            new_bin = bin_values[pos]
            temp_bin_counts = bin_counts.copy()
            if not np.isnan(new_bin):
                temp_bin_counts[int(new_bin)] += 1
            non_zero_counts = temp_bin_counts[temp_bin_counts > 0]
            if len(non_zero_counts) > 0:
                bin_probs = non_zero_counts / non_zero_counts.sum()
                uniformity = entropy(bin_probs) / np.log(n_bins)
            else:
                uniformity = 0
            
            # Combined score
            score = (alpha * target_stat) - (beta * other_stat) + (gamma * uniformity)
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        # Actually add the best candidate - update running statistics
        pos = index_to_pos[best_idx]
        n_selected += 1
        
        # Update target stats
        new_target_val = target_values[pos]
        delta_target = new_target_val - target_mean
        target_mean += delta_target / n_selected
        target_m2 += delta_target * (new_target_val - target_mean)
        
        # Update other stats
        new_other_vals = other_values[pos]
        delta_other = new_other_vals - other_means
        other_means += delta_other / n_selected
        other_m2 += delta_other * (new_other_vals - other_means)
        
        # Update bin counts
        new_bin = bin_values[pos]
        if not np.isnan(new_bin):
            bin_counts[int(new_bin)] += 1
        
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
    selected_df = df.loc[selected_indices]
    
    return selected_df['graph_id'].tolist(), selected_df, df_normalized

def evaluate_selection(df, selected_ids, target_feature):
    """Evaluate the quality of selection."""
    feature_cols = [col for col in df.columns if col != 'graph_id']
    other_cols = [col for col in feature_cols if col != target_feature]
    
    selected_df = df[df['graph_id'].isin(selected_ids)]
    df_normalized = df.copy()
    for col in feature_cols:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > col_min:
            df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df_normalized[col] = 0
    selected_df_normalized = df_normalized[df_normalized['graph_id'].isin(selected_ids)]
    
    print(f"\n Target Column '{target_feature}':")
    print(f"   Variance:")
    print(f"   Original variance: {df[target_feature].var():.4f}")
    print(f"   Selected variance: {selected_df[target_feature].var():.4f}")
    print(f"   Ratio: {selected_df[target_feature].var() / df[target_feature].var():.2%}\n")
    
    print(f"   Original variance normalized: {df_normalized[target_feature].var():.4f}")
    print(f"   Selected variance normalized: {selected_df_normalized[target_feature].var():.4f}")
    print(f"   Ratio normalized: {selected_df_normalized[target_feature].var() / df_normalized[target_feature].var():.2%}\n")
    
    print(f"   Standard Deviation:")
    print(f"   Original standard deviation: {df[target_feature].std():.4f}")
    print(f"   Selected standard deviation: {selected_df[target_feature].std():.4f}")
    print(f"   Ratio: {selected_df[target_feature].std() / df[target_feature].std():.2%}\n")
    
    print(f"   Original standard deviation normalized: {df_normalized[target_feature].std():.4f}")
    print(f"   Selected standard deviation normalized: {selected_df_normalized[target_feature].std():.4f}")
    print(f"   Ratio normalized: {selected_df_normalized[target_feature].std() / df_normalized[target_feature].std():.2%}")
    
    print(f"\n Other Columns:")
    print(f"   Variance:")
    orig_mean_var = df[other_cols].var().mean()
    sel_mean_var = selected_df[other_cols].var().mean()
    print(f"   Original: {orig_mean_var:.4f}")
    print(f"   Selected: {sel_mean_var:.4f}")
    print(f"   Reduction: {(1 - sel_mean_var/orig_mean_var):.2%}\n")
    
    orig_norm_mean_var = df_normalized[other_cols].var().mean()
    sel_norm_mean_var = selected_df_normalized[other_cols].var().mean()
    print(f"   Original normalized: {orig_norm_mean_var:.4f}")
    print(f"   Selected normalized: {sel_norm_mean_var:.4f}")
    print(f"   Reduction normalized: {(1 - sel_norm_mean_var/orig_norm_mean_var):.2%}\n")
    
    orig_mean_std = df[other_cols].std().mean()
    sel_mean_std = selected_df[other_cols].std().mean()
    print(f"   Standard Deviation:")
    print(f"   Original: {orig_mean_std:.4f}")
    print(f"   Selected: {sel_mean_std:.4f}")
    print(f"   Reduction: {(1 - sel_mean_std/orig_mean_std):.2%}\n")

    orig_norm_mean_std = df_normalized[other_cols].std().mean()
    sel_norm_mean_std = selected_df_normalized[other_cols].std().mean()
    print(f"   Original normalized: {orig_norm_mean_std:.4f}")
    print(f"   Selected normalized: {sel_norm_mean_std:.4f}")
    print(f"   Reduction normalized: {(1 - sel_norm_mean_std/orig_norm_mean_std):.2%}")
    
    return selected_df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_variance_comparison(df_full, selected_ids, target_feature, n_cols=3, figsize_per_plot=(4, 3), save_path=None):
    """
    Plot normalized histograms of all properties before/after selection.
    
    Parameters:
    -----------
    df_full : DataFrame with all graphs
    selected_ids : list of selected graph IDs
    target_feature : column with highest spread (highlighted)
    n_cols : number of columns in subplot grid
    figsize_per_plot : (width, height) per subplot
    """
    
    feature_cols = [col for col in df_full.columns if col != 'graph_id']
    
    # Get selected subset
    df_selected = df_full[df_full['graph_id'].isin(selected_ids)]
    
    # Normalize using FULL dataset min/max
    df_full_norm = df_full[feature_cols].apply(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    df_selected_norm = df_full[feature_cols].apply(
        lambda x: (x - x.min()) / (x.max() - x.min())
    ).loc[df_selected.index]
    
    # Calculate grid dimensions
    n_features = len(feature_cols)
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    )
    axes = axes.flatten()
    
    bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
    
    for idx, col in enumerate(feature_cols):
        ax = axes[idx]
        
        # Get data
        full_data = df_full_norm[col]
        selected_data = df_selected_norm[col]
        
        # Calculate variances
        var_full = full_data.var()
        var_selected = selected_data.var()
        var_change = ((var_selected - var_full) / var_full) * 100
        
        # Plot histograms
        ax.hist(
            full_data, 
            bins=bins, 
            alpha=0.5, 
            label=f'Full (n={len(df_full)})', 
            color='steelblue',
            density=True,
            edgecolor='white'
        )
        ax.hist(
            selected_data, 
            bins=bins, 
            alpha=0.7, 
            label=f'Selected (n={len(df_selected)})', 
            color='coral',
            density=True,
            edgecolor='white'
        )
        
        # Highlight the target column
        if col == target_feature:
            ax.set_facecolor('#e8f4e8')  # Light green background
            title_prefix = "Target feature: "
        else:
            title_prefix = ""
        
        # Title with variance info
        ax.set_title(
            f"{title_prefix}{col}\n"
            f"Var: {var_full:.3f} → {var_selected:.3f} ({var_change:+.1f}%)",
            fontsize=10,
            fontweight='bold' if col == target_feature else 'normal'
        )
        
        ax.set_xlabel('Normalized Value', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=8)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(
        'Property Distributions: Full Dataset vs Selected Subset\n',
        fontsize=12,
        fontweight='bold',
        y=1.02
    )
    
    plt.tight_layout()
    plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    
    

def plot_hist_comparison(df_full, selected_ids, target_feature, n_cols=3, figsize_per_plot=(4, 3), save_path=None):
    """
    Plot normalized histograms of all properties before/after selection.
    
    Parameters:
    -----------
    df_full : DataFrame with all graphs
    selected_ids : list of selected graph IDs
    target_feature : column with highest spread (highlighted)
    n_cols : number of columns in subplot grid
    figsize_per_plot : (width, height) per subplot
    """
    
    feature_cols = [col for col in df_full.columns if col != 'graph_id']
    
    # Get selected subset
    df_selected = df_full[df_full['graph_id'].isin(selected_ids)]
    
    # Normalize using FULL dataset min/max
    df_full_norm = df_full[feature_cols].apply(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    df_selected_norm = df_full[feature_cols].apply(
        lambda x: (x - x.min()) / (x.max() - x.min())
    ).loc[df_selected.index]
    
    # Calculate grid dimensions
    n_features = len(feature_cols)
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    )
    axes = axes.flatten()
    
    bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
    
    for idx, col in enumerate(feature_cols):
        ax = axes[idx]
        
        # Get data
        full_data = df_full_norm[col]
        selected_data = df_selected_norm[col]
        
        # Calculate stats
        var_full = full_data.var()
        var_selected = selected_data.var()
        var_change = ((var_selected - var_full) / var_full) * 100
        
        std_full = full_data.std()
        std_selected = selected_data.std()
        std_change = ((std_selected - std_full) / std_full) * 100
        
        # Plot histograms
        ax.hist(
            full_data, 
            bins=bins, 
            alpha=0.5, 
            label=f'Full (n={len(df_full)})', 
            color='steelblue',
            density=True,
            edgecolor='white'
        )
        ax.hist(
            selected_data, 
            bins=bins, 
            alpha=0.7, 
            label=f'Selected (n={len(df_selected)})', 
            color='coral',
            density=True,
            edgecolor='white'
        )
        
        # Highlight the target column
        if col == target_feature:
            ax.set_facecolor('#e8f4e8')  # Light green background
            title_prefix = "Target feature: "
        else:
            title_prefix = ""
        
        # Title with stat info
        ax.set_title(
            f"{title_prefix}{col}\n"
            f"Std: {std_full:.3f} → {std_selected:.3f} ({std_change:+.1f}%)\n"
            f"Var: {var_full:.3f} → {var_selected:.3f} ({var_change:+.1f}%)",
            fontsize=10,
            fontweight='bold' if col == target_feature else 'normal'
        )
        
        ax.set_xlabel('Normalized Value', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=8)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(
        'Property Distributions: Full Dataset vs Selected Subset\n',
        fontsize=12,
        fontweight='bold',
        y=1.02
    )
    
    plt.tight_layout()
    plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    
    


def plot_var_summary(df_full, selected_ids, target_feature, save_path=None):
    """
    Plot a summary bar chart of variance changes for all properties.
    """
    
    feature_cols = [col for col in df_full.columns if col != 'graph_id']
    
    df_selected = df_full[df_full['graph_id'].isin(selected_ids)]
    
    # Normalize
    df_full_norm = df_full[feature_cols].apply(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    df_selected_norm = df_full[feature_cols].apply(
        lambda x: (x - x.min()) / (x.max() - x.min())
    ).loc[df_selected.index]
    
    # Calculate variance changes
    var_full = df_full_norm.var()
    var_selected = df_selected_norm.var()
    var_change_pct = ((var_selected - var_full) / var_full) * 100
    
    # Sort by change
    other_cols = [c for c in feature_cols if c != target_feature]
    sorted_cols = sorted(other_cols, key=lambda x: var_change_pct[x]) + [target_feature]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['coral' if c != target_feature else 'forestgreen' for c in sorted_cols]
    bars = ax.barh(sorted_cols, var_change_pct[sorted_cols], color=colors, edgecolor='white')
    
    # Add value labels
    for bar, col in zip(bars, sorted_cols):
        width = bar.get_width()
        label_x = width + 2 if width >= 0 else width - 8
        ax.text(
            label_x, bar.get_y() + bar.get_height()/2,
            f'{width:+.1f}%',
            va='center', ha='left' if width >= 0 else 'right',
            fontsize=9, fontweight='bold'
        )
    
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Variance Change (%)', fontsize=11)
    ax.set_title(
        'Variance Change After Selection',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlim(min(var_change_pct.min() - 50, -50), max(var_change_pct.max() + 50, 50))
    
    plt.tight_layout()
    plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    

def plot_std_summary(df_full, selected_ids, target_feature, save_path=None):
    """
    Plot a summary bar chart of std changes for all properties.
    """
    
    feature_cols = [col for col in df_full.columns if col != 'graph_id']
    
    df_selected = df_full[df_full['graph_id'].isin(selected_ids)]
    
    # Normalize
    df_full_norm = df_full[feature_cols].apply(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    df_selected_norm = df_full[feature_cols].apply(
        lambda x: (x - x.min()) / (x.max() - x.min())
    ).loc[df_selected.index]
    
    # Calculate std changes
    std_full = df_full_norm.std()
    std_selected = df_selected_norm.std()
    std_change_pct = ((std_selected - std_full) / std_full) * 100
    
    # Sort by change
    other_cols = [c for c in feature_cols if c != target_feature]
    sorted_cols = sorted(other_cols, key=lambda x: std_change_pct[x]) + [target_feature]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['coral' if c != target_feature else 'forestgreen' for c in sorted_cols]
    bars = ax.barh(sorted_cols, std_change_pct[sorted_cols], color=colors, edgecolor='white')
    
    # Add value labels
    for bar, col in zip(bars, sorted_cols):
        width = bar.get_width()
        label_x = width + 2 if width >= 0 else width - 8
        ax.text(
            label_x, bar.get_y() + bar.get_height()/2,
            f'{width:+.1f}%',
            va='center', ha='left' if width >= 0 else 'right',
            fontsize=9, fontweight='bold'
        )
    
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Std Change (%)', fontsize=11)
    ax.set_title(
        'Std Change After Selection',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlim(min(std_change_pct.min() - 50, -50), max(std_change_pct.max() + 50, 50))
    
    plt.tight_layout()
    plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def compute_cv_robust(df, feature_cols=None):
    """
    Robust CV computation handling edge cases.
    Should be updated such that we always take the abs-mean since the range-based is never used and the abs-mean is equivalent to the standard for our usage.
    """
    
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != 'graph_id']
    
    results = []
    
    for col in feature_cols:
        values = df[col].dropna()
        
        mean_val = values.mean()
        median_val = values.median()
        std_val = values.std()
        
        # Handle zero or near-zero means
        if abs(mean_val) < 1e-10:
            # Use range-based metric instead
            range_val = values.max() - values.min()
            cv = range_val / std_val if std_val > 0 else 0  # Alternative metric
            cv_type = 'range-based'
        elif mean_val < 0:
            # CV not well-defined for negative means
            # Use absolute mean
            cv = std_val / abs(mean_val)
            cv_type = 'abs-mean'
        else:
            cv = std_val / mean_val
            cv_type = 'standard'
        
        results.append({
            'feature': col,
            'mean': mean_val,
            'std': std_val,
            'CV': cv,
            'CV_type': cv_type,
            'CV_pct': cv * 100,  # As percentage
        })
    
    return pd.DataFrame(results).sort_values('CV', ascending=False)