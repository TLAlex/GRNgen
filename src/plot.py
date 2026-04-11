import numpy as np
import math
import pandas as pd
import networkx as nx
from collections import Counter
import os

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import glob
import random
from multiprocessing import Pool, cpu_count

def plot_in_out_total_degree_distribution(
    G,
    label="Network",
    save_path=None,
    log=True
):
    """
    Plot the log-log in-degree, out-degree, and total-degree distributions
    of a directed graph.

    Parameters
    ----------
    G : nx.DiGraph or nx.Graph
        The network to analyze.
    label : str, optional
        Label used in the plot title.
    save_path : str, optional
        If provided, saves the figure to this path.
    """

    # Compute degrees depending on graph type
    if isinstance(G, nx.DiGraph):
        indegree_seq = [d for _, d in G.in_degree()]
        outdegree_seq = [d for _, d in G.out_degree()]
    else:
        raise Exception('The graph is not directed')

    totaldegree_seq = [d for _, d in G.degree()]

    # Convert to distributions
    def degree_distribution(seq, N):
        degrees, counts = np.unique(seq, return_counts=True)
        probs = counts / N
        return degrees, probs

    N = G.number_of_nodes()
    deg_in, p_in = degree_distribution(indegree_seq, N)
    deg_out, p_out = degree_distribution(outdegree_seq, N)
    deg_total, p_total = degree_distribution(totaldegree_seq, N)

    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the three distributions
    ax.scatter(deg_total, p_total, color='tab:blue', alpha=0.7, label="Total degree", marker='o')
    ax.scatter(deg_in, p_in, color='tab:orange', alpha=0.7, label="In-degree", marker='<')
    ax.scatter(deg_out, p_out, color='tab:green', alpha=0.7, label="Out-degree", marker='>')

    if log:
        # Log scale on both axes
        ax.set_xscale('log')
        ax.set_yscale('log')

    # Axis labels and title
    ax.set_xlabel("Degree k")
    ax.set_ylabel("Frequency P(k)")
    if log:
        ax.set_title(f"Log-Log Degree Distributions ({label})")
        # Show real tick values instead of scientific notation
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(axis='both', style='plain')
    else:
        ax.set_title(f"Degree Distributions ({label})")
    ax.legend()
    
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')

    plt.show()


MOTIF_ORDER = [
    'Fan-In', 'Fan-Out', 'Cascade', 'Mutual-Out', 'Mutual-In',
    'FFL', 'FBL', 'Bi-Mutual', 'Regulated-Mutual', 'Regulating-Mutual',
    'Mutual-Cascade', 'Semi-Clique', 'Clique'
]

def plot_motifs_count(motifs_count, label="network", log=True, save_filename=None):
    """
    Plot motif counts in alphabetical order.

    Parameters
    ----------
    motifs_count : dict
        Mapping motif name -> count
    label : str
        Title label for the plot
    log : bool
        Whether to use a logarithmic y-axis
    save_filename : str or None
        If provided, saves plot to this filename
    """

    # Sort alphabetically
    motifs_sorted = sorted(motifs_count.keys())
    values = [motifs_count[m] for m in motifs_sorted]

    x = np.arange(len(motifs_sorted))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, values, width=0.6)

    if log:
        ax.set_yscale("log", base=10)

    ax.set_ylabel("Motif Count")
    ax.set_xlabel("Motif")
    ax.set_xticks(x)
    ax.set_xticklabels(motifs_sorted, rotation=45)
    ax.set_title(f"Motif distribution: {label}")
    ax.grid(axis='y', which='both', linestyle=':', alpha=0.4)

    plt.tight_layout()
    if save_filename:
        fig.savefig(save_filename)
    plt.show()

def compare_motifs_count(motifs_count1, motifs_count2, label1="Network 1", label2="Network 2", log=True, save_filename=None):
    """
    Plot motif counts for two networks side by side.

    Parameters
    ----------
    motifs_count1 : dict
        Mapping motif name -> count for the first network.
    motifs_count2 : dict
        Mapping motif name -> count for the second network.
    label1 : str
        Title label for the first plot.
    label2 : str
        Title label for the second plot.
    log : bool
        Whether to use a logarithmic y-axis.
    save_filename : str or None
        If provided, saves plot to this filename.
    """

    # Sort alphabetically
    motifs_sorted = sorted(set(motifs_count1.keys()).union(set(motifs_count2.keys())))
    
    # Get values for both motifs counts, defaulting to 0 if a motif is not present in one of the counts
    values1 = [motifs_count1.get(m, 0) for m in motifs_sorted]
    values2 = [motifs_count2.get(m, 0) for m in motifs_sorted]

    x = np.arange(len(motifs_sorted))

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define bar width and positions
    width = 0.4
    ax.bar(x - width/2, values1, width=width, color='red', label=label1)
    ax.bar(x + width/2, values2, width=width, color='blue', label=label2)

    if log:
        ax.set_yscale("log", base=10)

    ax.set_ylabel("Motif Count")
    ax.set_xlabel("Motif")
    ax.set_xticks(x)
    ax.set_xticklabels(motifs_sorted, rotation=45)
    ax.set_title("Motif Distribution")
    ax.grid(axis='y', which='both', linestyle=':', alpha=0.4)
    ax.legend()  # Add legend to differentiate between the two sets

    plt.tight_layout()
    if save_filename:
        fig.savefig(save_filename)
    plt.show()

def compare_motifs_count_three(
        motifs_count1, motifs_count2, motifs_count3,
        label1="Network 1", label2="Network 2", label3="Network 3",
        log=True, save_filename=None):
    """
    Plot motif counts for three networks side by side.

    Parameters
    ----------
    motifs_count1, motifs_count2, motifs_count3 : dict
        Mapping motif name -> count for each network.
    label1, label2, label3 : str
        Labels for each network.
    log : bool
        Whether to use a logarithmic y-axis.
    save_filename : str or None
        If provided, saves plot to this filename.
    """

    # ---- Combine motif sets and sort ----
    motifs_sorted = sorted(
        set(motifs_count1.keys()) |
        set(motifs_count2.keys()) |
        set(motifs_count3.keys())
    )

    # ---- Extract counts (default 0 if missing) ----
    values1 = [motifs_count1.get(m, 0) for m in motifs_sorted]
    values2 = [motifs_count2.get(m, 0) for m in motifs_sorted]
    values3 = [motifs_count3.get(m, 0) for m in motifs_sorted]

    x = np.arange(len(motifs_sorted))

    fig, ax = plt.subplots(figsize=(14, 6))

    # ---- Bar width and positions ----
    width = 0.25  # three bars → smaller width
    ax.bar(x - width,     values1, width=width, label=label1, color='red')
    ax.bar(x,             values2, width=width, label=label2, color='blue')
    ax.bar(x + width,     values3, width=width, label=label3, color='green')

    # ---- Scaling ----
    if log:
        ax.set_yscale("log")

    # ---- Labels & formatting ----
    ax.set_ylabel("Motif Count")
    ax.set_xlabel("Motif")
    ax.set_xticks(x)
    ax.set_xticklabels(motifs_sorted, rotation=45)
    ax.set_title("Motif Distribution Across Three Networks")
    ax.grid(axis='y', which='both', linestyle=':', alpha=0.4)
    ax.legend()

    plt.tight_layout()

    if save_filename:
        fig.savefig(save_filename)

    plt.show()

def plot_deg_ref_vs_sim_3panels(ref_g, sim_g, xmin=1,
                                colors=("tab:blue", "tab:orange", "tab:green"),
                                figsize=(12, 4)):
    def degs(G):
        if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
            return ([d for _, d in G.degree()],
                    [d for _, d in G.in_degree()],
                    [d for _, d in G.out_degree()])
        d = [d for _, d in G.degree()]
        z = [0]*len(d)  # placeholders for undirected
        return d, z, z

    def to_points(arr):
        v, c = np.unique(arr, return_counts=True)
        m = v >= xmin
        x = np.log10(v[m] + 1); y = np.log10(c[m] + 1)
        return x, y

    rt, ri, ro = degs(ref_g)
    st, si, so = degs(sim_g)

    labels = ["Total degree", "In-degree", "Out-degree"]
    ref_arrs = [rt, ri, ro]
    sim_arrs = [st, si, so]

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for ax, lab, col, r, s in zip(axes, labels, colors, ref_arrs, sim_arrs):
        xr, yr = to_points(r)
        xs, ys = to_points(s)
        # Skip empty panels (e.g., in/out for undirected)
        had_any = False
        if xr.size:
            ax.scatter(xr, yr, s=28, alpha=0.9, color=col, marker="o", label="True")
            had_any = True
        if xs.size:
            ax.scatter(xs, ys, s=28, alpha=0.9, color=col, marker="x", label="Synthetic")
            had_any = True
        ax.set_title(lab)
        ax.set_xlabel("Degree (log10)")
        if ax is axes[0]:
            ax.set_ylabel("Count (log10)")
        ax.grid(True, ls=":", alpha=0.5)
        if had_any:
            ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    plt.show()

def normalize_motif_counts(G, motif_dict):
    n = G.number_of_nodes()
    norm_term = n*(n-1)*(n-2)
    motifs_normalized = {key: motif_dict[key]/norm_term for key in motif_dict.keys()}
    return motifs_normalized

def plot_single_radar(
    data_dict,
    parameters,
    save_filepath=None,
    title=None,
    log=False,
    eps=1e-6,
    color='tab:blue'
):
    """
    Plot a radar chart for a single dataset of normalized network properties.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing normalized network properties.
    parameters : list[str]
        List of keys (features) to display on the radar chart.
    save_filepath : str
        Path to save the generated radar chart image.
    title : str, optional
        Title of the radar chart.
    log : bool, optional
        Whether to use a logarithmic radial scale.
    eps : float, optional
        Small value added to zero or negative values to avoid log(0).
    """

    # --- Validate inputs ---
    for p in parameters:
        if p not in data_dict:
            raise KeyError(f"Parameter '{p}' not found in data_dict")

    # --- Prepare data (log-safe) ---
    values = []
    for p in parameters:
        v = data_dict[p]
        if log and v <= 0:
            v = eps
        values.append(v)

    values += values[:1]  # close the polygon

    n_params = len(parameters)
    angles = np.linspace(0, 2 * np.pi, n_params, endpoint=False).tolist()
    angles += angles[:1]

    # --- Create radar chart ---
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid', color=color)
    ax.fill(angles, values, alpha=0.25, color=color)

    # --- Configure axes ---
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(parameters)

    if log:
        ax.set_yscale("log")
        ax.set_ylim(eps, max(values) * 1.1)

    ax.set_rlabel_position(0)

    if title:
        plt.title(title, size=14, weight="bold", pad=20)

    # --- Save and show ---
    plt.tight_layout()
    if save_filepath:
        plt.savefig(save_filepath, dpi=300, bbox_inches="tight")
    print(f"Radar chart saved to: {save_filepath}")
    plt.show()

def plot_double_radar(
    data_dict_1,
    data_dict_2,
    parameters,
    labels=("Graph 1", "Graph 2"),
    colors=("tab:blue", "tab:orange"),
    save_filepath=None,
    title=None,
    log=False,
    eps=1e-6
):
    """
    Plot a radar chart comparing two datasets of normalized network properties.

    Parameters
    ----------
    data_dict_1 : dict
        Dictionary containing normalized network properties for graph 1.
    data_dict_2 : dict
        Dictionary containing normalized network properties for graph 2.
    parameters : list[str]
        List of keys (features) to display on the radar chart.
    labels : tuple[str, str], optional
        Labels for the two graphs (used in legend).
    colors : tuple[str, str], optional
        Line/fill colors for the two graphs.
    save_filepath : str, optional
        Path to save the generated radar chart image.
    title : str, optional
        Title of the radar chart.
    log : bool, optional
        Whether to use a logarithmic radial scale.
    eps : float, optional
        Small value added to zero or negative values to avoid log(0).
    """

    # --- Validate inputs ---
    for p in parameters:
        if p not in data_dict_1:
            raise KeyError(f"Parameter '{p}' not found in data_dict_1")
        if p not in data_dict_2:
            raise KeyError(f"Parameter '{p}' not found in data_dict_2")

    # --- Prepare data (log-safe) ---
    def prepare_values(data_dict):
        vals = []
        for p in parameters:
            v = data_dict[p]
            if log and v <= 0:
                v = eps
            vals.append(v)
        return vals + vals[:1]  # close polygon

    values_1 = prepare_values(data_dict_1)
    values_2 = prepare_values(data_dict_2)

    n_params = len(parameters)
    angles = np.linspace(0, 2 * np.pi, n_params, endpoint=False).tolist()
    angles += angles[:1]

    # --- Create radar chart ---
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, values_1, linewidth=2, color=colors[0], label=labels[0])
    ax.fill(angles, values_1, color=colors[0], alpha=0.25)

    ax.plot(angles, values_2, linewidth=2, color=colors[1], label=labels[1])
    ax.fill(angles, values_2, color=colors[1], alpha=0.25)

    # --- Configure axes ---
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(parameters)

    if log:
        ax.set_yscale("log")
        ymax = max(max(values_1), max(values_2))
        ax.set_ylim(eps, ymax * 1.1)

    ax.set_rlabel_position(0)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    if title:
        plt.title(title, size=14, weight="bold", pad=20)

    # --- Save and show ---
    plt.tight_layout()
    if save_filepath:
        plt.savefig(save_filepath, dpi=300, bbox_inches="tight")
        print(f"Radar chart saved to: {save_filepath}")

    plt.show()


def plot_correlation_heatmap(motifs_count_df, save_filename=None):
    """
    Create a correlation heatmap for motif counts with Pearson r values displayed.

    Parameters
    ----------
    motifs_count_df : pd.DataFrame
        DataFrame containing motif counts (each column is a motif, each row is a graph).
    save_filename : str or None
        If provided, saves plot to this filename.
    """
    
    # Calculate the correlation matrix
    correlation_matrix = motifs_count_df.corr(method='pearson')

    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))

    # Create a heatmap with annotations
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',mask=mask, 
                square=True, cbar_kws={"shrink": .8}, linewidths=0.5, linecolor='black')

    # Set titles and labels
    plt.title('Correlation Heatmap of Motif Counts', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a filename is provided
    if save_filename:
        plt.savefig(save_filename, dpi=300)
    
    plt.show()


# def plot_violin_with_targets(
#     df: pd.DataFrame,
#     target_dict: dict,
#     bounds_dict: dict,
#     plots_per_row=4,
#     figsize_per_plot=(3, 4),
#     violin_color="skyblue",
#     target_color="red",
#     n_yticks=5,
#     show_xtick=True
# ):
#     """
#     Plot violin plots for each dataframe column and overlay target values,
#     using provided bounds to control y-limits and y-ticks.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame containing distributions for each property.
#     target_dict : dict
#         Dictionary of target values (keys must match df columns).
#     bounds_dict : dict
#         Dictionary {property: (ymin, ymax)} specifying bounds.
#     plots_per_row : int
#         Number of plots per row.
#     figsize_per_plot : tuple
#         Size of each subplot (width, height).
#     violin_color : str
#         Color of violin plots.
#     target_color : str
#         Color of target value line.
#     n_yticks : int
#         Number of y-ticks to display.
#     show_xtick : bool
#         Whether to show a dummy x-tick.
#     """

#     properties = list(target_dict.keys())
#     n_props = len(properties)
#     n_rows = math.ceil(n_props / plots_per_row)

#     fig, axes = plt.subplots(
#         n_rows,
#         plots_per_row,
#         figsize=(figsize_per_plot[0] * plots_per_row,
#                  figsize_per_plot[1] * n_rows)
#     )

#     axes = axes.flatten()

#     for i, prop in enumerate(properties):
#         ax = axes[i]

#         # --- Violin ---
#         sns.violinplot(
#             y=df[prop],
#             ax=ax,
#             color=violin_color,
#             inner="quartile"
#         )

#         # --- Target line ---
#         ax.axhline(
#             target_dict[prop],
#             color=target_color,
#             linestyle="--",
#             linewidth=2
#         )

#         # --- Bounds handling ---
#         if prop in bounds_dict:
#             ymin, ymax = bounds_dict[prop]
#             ax.set_ylim(ymin, ymax)
#             ax.set_yticks(np.linspace(ymin, ymax, n_yticks))

#         # --- Titles and ticks ---
#         ax.set_title(prop)

#         if show_xtick:
#             ax.set_xticks([0])
#             ax.set_xticklabels([""])
#         else:
#             ax.set_xticks([])

#     # Remove unused subplots
#     for j in range(i + 1, len(axes)):
#         fig.delaxes(axes[j])

#     plt.tight_layout()
#     plt.show()

def plot_violin_with_targets(
    df: pd.DataFrame,
    target_dict: dict,
    bounds_dict: dict,
    plots_per_row=4,
    figsize_per_plot=(3, 4),
    violin_color="skyblue",
    target_color="red",
    n_yticks=5,
    show_xtick=True,
    other_line=None,              
    other_line_color="black",
    other_line_style=":",
    target="Target",
    save_filename=None
):
    """
    Plot violin plots for each dataframe column and overlay target values,
    using provided bounds to control y-limits and y-ticks.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing distributions for each property.
    target_dict : dict
        Dictionary of target values (keys must match df columns).
    bounds_dict : dict
        Dictionary {property: (ymin, ymax)} specifying bounds.
    other_line : tuple (row_index, label), optional
        Row index in df and legend label for an additional reference line.
    """

    properties = list(target_dict.keys())
    n_props = len(properties)
    n_rows = math.ceil(n_props / plots_per_row)

    fig, axes = plt.subplots(
        n_rows,
        plots_per_row,
        figsize=(figsize_per_plot[0] * plots_per_row,
                 figsize_per_plot[1] * n_rows)
    )

    axes = axes.flatten()

    # --- Legend handles (created once) ---
    target_handle = plt.Line2D(
        [], [], color=target_color, linestyle="--", linewidth=2, label=target
    )

    other_handle = None
    if other_line is not None:
        row_index, other_label = other_line
        other_handle = plt.Line2D(
            [], [], color=other_line_color,
            linestyle=other_line_style, linewidth=2,
            label=other_label
        )

    for i, prop in enumerate(properties):
        ax = axes[i]

        # --- Violin ---
        sns.violinplot(
            data=df,
            y=prop,
            ax=ax,
            color=violin_color,
            inner="quartile",
            cut=0
        )
        # sns.stripplot(
        #     data=df, y=prop, ax=ax, color="#1f77b4",
        #     size=3, alpha=0.5, jitter=True
        # )

        # --- Target line ---
        ax.axhline(
            target_dict[prop],
            color=target_color,
            linestyle="--",
            linewidth=2
        )

        # --- Optional other line ---
        if other_line is not None:
            row_index, _ = other_line
            if prop in df.columns:
                ax.axhline(
                    df.iloc[row_index][prop],
                    color=other_line_color,
                    linestyle=other_line_style,
                    linewidth=2
                )

        # --- Bounds handling ---
        if prop in bounds_dict:
            ymin, ymax = bounds_dict[prop]
            ax.set_ylim(ymin, ymax)
            ax.set_yticks(np.linspace(ymin, ymax, n_yticks))
        # --- Titles and ticks ---
        ax.set_title(prop)

        if show_xtick:
            ax.set_xticks([0])
            ax.set_xticklabels([""])
        else:
            ax.set_xticks([])

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # --- Single shared legend ---
    handles = [target_handle]
    if other_handle is not None:
        handles.append(other_handle)

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_filename:
        plt.savefig(save_filename, dpi=300)
    plt.show()

def plot_violin_two_conditions(
    df,
    target_dict,
    bounds_dict,
    hue_col="condition",
    plots_per_row=4,
    figsize=None,
    target_color="red",
    target_label="Target",
    other_line=None,
    other_line_style=":",
    save_filename=None
):
    metrics = [col for col in df.columns if col != hue_col]

    n_plots = len(metrics)
    n_rows = math.ceil(n_plots / plots_per_row)

    if figsize is None:
        figsize = (5 * plots_per_row, 4 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        plots_per_row,
        figsize=figsize
    )

    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    hue_levels = df[hue_col].unique()
    palette = sns.color_palette()

    # -----------------------------
    # Legend handles
    # -----------------------------

    target_handle = plt.Line2D(
        [], [],
        color=target_color,
        linestyle='--',
        linewidth=2,
        label=target_label
    )

    other_handles = []
    if other_line is not None:
        for idx, (_, other_label) in enumerate(other_line):
            other_handles.append(
                plt.Line2D(
                    [], [],
                    color=palette[idx],
                    linestyle=other_line_style,
                    linewidth=2,
                    label=other_label
                )
            )

    # -----------------------------
    # Plotting
    # -----------------------------

    for i, metric in enumerate(metrics):

        ax = axes[i]

        sns.violinplot(
            data=df,
            x=hue_col,
            y=metric,
            hue=hue_col,
            legend=False,
            ax=ax,
            cut=0
        )

        # Target line
        if metric in target_dict:
            ax.axhline(
                target_dict[metric],
                color=target_color,
                linestyle='--',
                linewidth=1.5
            )

        # Other reference lines
        if other_line is not None:
            for idx, (row_index, _) in enumerate(other_line):

                ax.axhline(
                    df.iloc[row_index][metric],
                    color=palette[idx],
                    linestyle=other_line_style,
                    linewidth=2
                )

        # Bounds
        if metric in bounds_dict:
            ax.set_ylim(bounds_dict[metric])

        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # -----------------------------
    # Legend
    # -----------------------------

    condition_handles = [
        plt.Line2D(
            [], [],
            color=palette[idx],
            marker='s',
            linestyle='',
            markersize=10,
            label=str(level)
        )
        for idx, level in enumerate(hue_levels)
    ]

    handles = condition_handles + [target_handle] + other_handles

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_filename:
        plt.savefig(save_filename, dpi=300)

    plt.show()


def plot_violin_multi(
    df,
    target_dict,
    bounds_dict,
    hue_col="condition",
    plots_per_row=4,
    figsize=None,
    target_color="red",
    target_label="Target",
    other_line=None,
    other_line_style=":",
    save_filename=None
):
    # Filter numeric columns only
    metrics = [col for col in df.select_dtypes(include='number').columns]

    n_plots = len(metrics)
    n_rows = math.ceil(n_plots / plots_per_row)

    if figsize is None:
        figsize = (5 * plots_per_row, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, plots_per_row, figsize=figsize)

    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    hue_levels = list(df[hue_col].unique())
    # FIX: Palette is sized only to the number of unique conditions
    palette = sns.color_palette(n_colors=len(hue_levels))
    # Map each label to a specific color
    color_map = {level: palette[i] for i, level in enumerate(hue_levels)}

    # -----------------------------
    # Legend handles
    # -----------------------------
    target_handle = plt.Line2D([], [], color=target_color, linestyle='--', linewidth=2, label=target_label)

    other_handles = []
    if other_line is not None:
        for _, other_label in other_line:
            other_handles.append(
                plt.Line2D(
                    [], [],
                    color=color_map[other_label], # FIX: Use the color mapped to this label
                    linestyle=other_line_style,
                    linewidth=2,
                    label=f"Best {other_label}"
                )
            )

    # -----------------------------
    # Plotting
    # -----------------------------
    for i, metric in enumerate(metrics):
        ax = axes[i]

        sns.violinplot(
            data=df,
            x=hue_col,
            y=metric,
            hue=hue_col,
            legend=False,
            ax=ax,
            cut=0,
            palette=palette # Consistent with hue_levels
        )

        # Target line (usually 0 for errors)
        if metric in target_dict:
            ax.axhline(target_dict[metric], color=target_color, linestyle='--', linewidth=1.5)

        # Other reference lines (Best samples)
        if other_line is not None:
            for row_index, other_label in other_line:
                ax.axhline(
                    df.iloc[row_index][metric],
                    color=color_map[other_label], # FIX: Use the mapped color
                    linestyle=other_line_style,
                    linewidth=2
                )
        if bounds_dict == None:
            ax.set_ylim((-2,2))
            
        elif metric in bounds_dict:
            ax.set_ylim(bounds_dict[metric])


        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # -----------------------------
    # Legend
    # -----------------------------
    condition_handles = [
        plt.Line2D([], [], color=color_map[level], marker='s', linestyle='', markersize=10, label=str(level))
        for level in hue_levels
    ]

    handles = condition_handles + [target_handle] + other_handles
    
    # We move the anchor slightly further right (1.05 instead of 1.0)
    fig.legend(
        handles=handles, 
        loc="center left", # Change to center left so it grows to the right
        ncol=1, 
        frameon=False, 
        bbox_to_anchor=(0.85, 0.5) # Position it at 85% of fig width
    )

    # If only 1 plot, we need more aggressive padding on the right
    right_margin = 0.75 if n_plots == 1 else 0.82
    
    plt.tight_layout(rect=[0, 0, right_margin, 1]) 

    # If tight_layout still fails for single plots, use subplots_adjust as a fallback
    if n_plots == 1:
        fig.subplots_adjust(right=0.7) # Manually force right margin

    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.show()

# PAIR PLOT

def plot_degree_3d_histogram(pairs, title="3D Degree Distribution (Discrete)"):
    # 1. Count frequencies of each (source, target) pair
    counts = Counter(pairs)
    
    x_coords = []
    y_coords = []
    z_heights = []
    
    # 2. Prepare the pillars
    # To create a "bar" effect in 3D without smoothing, 
    # we can use Mesh3d or simply create a scatter with "markers" 
    # shaped like cubes, but for Plotly the cleanest 'bar' look 
    # is often achieved by plotting the points as vertical lines.
    
    for (src, tgt), count in counts.items():
        x_coords.append(src)
        y_coords.append(tgt)
        z_heights.append(count)

    # 3. Use 3D Bar representation
    # We use a trick: Plotly doesn't have a simple 'Bar3d' like Matplotlib, 
    # but we can simulate it perfectly using a Scatter3d with 'lines+markers' 
    # or by using a custom function. 
    # However, for a clean 'histogram' look, we'll use a 3D Scatter 
    # with a large 'square' symbol or a custom Mesh.
    
    fig = go.Figure()

    # Add each bar as a separate vertical line to ensure no interpolation (smoothing)
    for x, y, z in zip(x_coords, y_coords, z_heights):
        fig.add_trace(go.Scatter3d(
            x=[x, x], y=[y, y], z=[0, z],
            mode='lines',
            line=dict(color='royalblue', width=10), # 'width' creates the bar thickness
            hoverinfo='none',
            showlegend=False
        ))

    # Add the "tops" of the bars for hover information
    fig.add_trace(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_heights,
        mode='markers',
        marker=dict(
            size=5,
            color='royalblue',
            symbol='square',
            opacity=1
        ),
        hovertemplate='Src Degree: %{x}<br>Tgt Degree: %{y}<br>Count: %{z}<extra></extra>',
        showlegend=False
    ))

    # 4. Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Source Degree',
            yaxis_title='Target Degree',
            zaxis_title='Count',
            xaxis=dict(showgrid=True, zeroline=False),
            yaxis=dict(showgrid=True, zeroline=False),
            zaxis=dict(showgrid=True, zeroline=False),
        ),
        width=800,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

def plot_degree_3d_surface(pairs, title="Interactive Degree Distribution"):
    # 1. Prepare data
    x_data = [p[0] for p in pairs]
    y_data = [p[1] for p in pairs]
    
    # 2. Create a 2D histogram (grid of counts)
    # We determine the range based on the max degrees found
    max_x = max(x_data) + 1
    max_y = max(y_data) + 1
    
    # Create a 2D array to store counts for every degree combination
    z_matrix = np.zeros((max_y, max_x))
    
    for x, y in pairs:
        z_matrix[y, x] += 1

    # 3. Create the 3D Surface Plot
    fig = go.Figure(data=[go.Surface(
        z=z_matrix,
        colorscale='Viridis',
        colorbar_title='Counts'
    )])

    # 4. Update layout for interactivity and labels
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Source Degree (x)',
            yaxis_title='Target Degree (y)',
            zaxis_title='Counts (z)'
        ),
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    fig.show()

def plot_degree_3d_scatter_no_legend(pairs, title="3D Degree Distribution"):
    # 1. Count frequencies
    counts = Counter(pairs)
    
    # 2. Extract coordinates and frequencies
    x_coords = []
    y_coords = []
    z_counts = []
    
    for (src, tgt), count in counts.items():
        x_coords.append(src)
        y_coords.append(tgt)
        z_counts.append(count)

    # 3. Create the simplified 3D Scatter Plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_counts,
        mode='markers',
        marker=dict(
            size=3,              # Constant small size for all points
            color='royalblue',    # Single uniform color
            opacity=0.7,
            showscale=False      # Removes the color bar/legend
        ),
        hovertemplate='Src: %{x}<br>Tgt: %{y}<br>Count: %{z}<extra></extra>'
    )])

    # 4. Layout configuration
    fig.update_layout(
        title=title,
        showlegend=False,        # Ensures the legend is hidden
        scene=dict(
            xaxis_title='Source Degree',
            yaxis_title='Target Degree',
            zaxis_title='Count',
            aspectmode='cube',
            xaxis=dict(gridcolor='rgb(230, 230, 230)'),
            yaxis=dict(gridcolor='rgb(230, 230, 230)'),
            zaxis=dict(gridcolor='rgb(230, 230, 230)'),
            bgcolor='white'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        width=800,
        height=700
    )

    fig.show()

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import math

# def plot_grouped_barplot(
#     df,
#     target_dict,
#     hue_col="condition",
#     figsize=(14, 7),
#     palette=["blue", "red"],
#     log_scale=True,
#     save_filename=None
# ):
#     """
#     Generates a grouped bar plot with error bars for distributions (df)
#     and compares them against ground truth values (target_dict).
#     """

#     # 1. Reshape the dataframe from wide to long format for Seaborn
#     # This turns columns [Metric1, Metric2, condition] into [condition, metric_name, value]
#     metrics = [col for col in df.columns if col != hue_col]
#     df_long = df.melt(id_vars=hue_col, value_vars=metrics, var_name="Metric", value_name="Count")

#     # 2. Prepare Ground Truth (Target) Data
#     # We create a temporary dataframe for the target bars to plot them alongside
#     target_data = pd.DataFrame([
#         {hue_col: "Ground Truth", "Metric": k, "Count": v} 
#         for k, v in target_dict.items()
#     ])

#     # Combine the simulation data and the ground truth data
#     combined_df = pd.concat([df_long, target_data], ignore_index=True)

#     # 3. Plotting
#     plt.figure(figsize=figsize)

#     # sns.barplot calculates the mean and 95% CI (or sd) by default
#     # errorbar="sd" shows the standard deviation
#     ax = sns.barplot(
#         data=combined_df,
#         x="Metric",
#         y="Count",
#         hue=hue_col,
#         palette=palette,
#         errorbar="sd",
#         capsize=.1
#     )

#     # 4. Styling
#     if log_scale:
#         ax.set_yscale("log")

#     plt.xticks(rotation=45)
#     plt.grid(axis='y', linestyle=':', which='both', alpha=0.7)

#     ax.set_title("Comparison of Motif Counts")
#     ax.set_xlabel("Motifs")
#     ax.set_ylabel("Count")

#     # Adjust legend
#     plt.legend(frameon=True, loc='upper right')

#     plt.tight_layout()

#     if save_filename:
#         plt.savefig(save_filename, dpi=300)

#     plt.show()

def plot_grouped_barplot_with_ground_truth(
    df,
    target_dict,
    hue_col="condition",
    figsize=(14, 7),
    target_label="Ground Truth",
    target_color="red",
    log_scale=True,
    save_filename=None
):
    # 1. Prepare simulation data (Wide to Long)
    metrics = [col for col in df.columns if col != hue_col]
    df_long = df.melt(id_vars=hue_col, value_vars=metrics, var_name="Metric", value_name="Count")

    # 2. Prepare Ground Truth data
    target_df = pd.DataFrame([
        {hue_col: target_label, "Metric": k, "Count": v} 
        for k, v in target_dict.items()
    ])
    
    # Combine both
    combined_df = pd.concat([df_long, target_df], ignore_index=True)

    # 3. Handle Colors
    # Get the original hue levels from the dataframe (e.g., Condition A, Condition B)
    original_hues = list(df[hue_col].unique())
    # Create a palette: original levels get default Seaborn colors, Ground Truth gets Red
    base_palette = sns.color_palette(n_colors=len(original_hues))
    palette_dict = {level: color for level, color in zip(original_hues, base_palette)}
    palette_dict[target_label] = target_color

    # 4. Plotting
    plt.figure(figsize=figsize)
    
    # We define hue_order to ensure Ground Truth is always the last bar in the group
    hue_order = original_hues + [target_label]

    ax = sns.barplot(
        data=combined_df,
        x="Metric",
        y="Count",
        hue=hue_col,
        hue_order=hue_order,
        palette=palette_dict,
        errorbar="sd", # Matches your need for distribution error bars
        capsize=.1,
        edgecolor="white",
        linewidth=0.5
    )

    # 5. Styling to match your image
    if log_scale:
        ax.set_yscale("log")
    
    plt.xticks(rotation=45, ha='right')
    # Adding the minor grid lines often seen in log plots
    plt.grid(axis='y', linestyle=':', which='both', alpha=0.5)
    
    ax.set_title("Motif Count Comparison", fontsize=14, pad=20)
    ax.set_xlabel("") # Cleaner look
    ax.set_ylabel("Motif count")
    
    # Move legend to match the layout
    plt.legend(frameon=True, loc='upper right', title=None)
    
    plt.tight_layout()

    if save_filename:
        plt.savefig(save_filename, dpi=300)

    plt.show()




def plot_feature_profiles(
    df: pd.DataFrame,
    target_dict: dict,
    bounds_dict: dict = None,
    plots_per_row: int = 4,
    figsize_per_plot: tuple = (300, 350),
    sample_colors: list = None,
    target_color: str = "black",
    target_dash: str = "dash",
    marker_size: int = 12,
    sample_opacity: float = 0.7,
    title: str = None,
    center_on_target: bool = True,
    default_margin_pct: float = 0.2,  # 20% margin around data if no bounds provided
    save_filename: str = None
):
    """
    Plot scatter subplots for each feature with target centered on y-axis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing sample values for each property.
    target_dict : dict
        Dictionary of target values (keys must match df columns).
    bounds_dict : dict, optional
        Dictionary {property: (ymin, ymax)} specifying y-axis bounds.
        If None or missing key, bounds are auto-calculated centered on target.
    plots_per_row : int
        Number of subplots per row.
    figsize_per_plot : tuple
        (width, height) in pixels for each subplot.
    sample_colors : list, optional
        List of colors for samples. Defaults to Plotly colors.
    target_color : str
        Color for target line.
    target_dash : str
        Dash style for target line ('dash', 'dot', 'dashdot', 'solid').
    marker_size : int
        Size of scatter markers.
    sample_opacity : float
        Opacity of sample markers.
    title : str
        Plot title.
    center_on_target : bool
        If True, center y-axis on target value.
    default_margin_pct : float
        Percentage margin around data range when auto-calculating bounds.
    save_filename : str, optional
        If provided, save the figure to this file.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    
    if bounds_dict is None:
        bounds_dict = {}
    
    if sample_colors is None:
        sample_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
                         '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    
    params = [p for p in target_dict.keys() if p in df.columns]
    n_params = len(params)
    n_rows = int(np.ceil(n_params / plots_per_row))
    n_cols = min(plots_per_row, n_params)
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=params,
        vertical_spacing=0.15 / n_rows if n_rows > 1 else 0.15,
        horizontal_spacing=0.08
    )
    
    n_samples = len(df)
    sample_labels = [f"Sample {i+1}" for i in range(n_samples)]
    
    for j, param in enumerate(params):
        row = j // plots_per_row + 1
        col = j % plots_per_row + 1
        
        target_val = target_dict[param]
        sample_vals = df[param].values
        
        # --- Calculate y-axis bounds ---
        if param in bounds_dict:
            ymin, ymax = bounds_dict[param]
        else:
            # Auto-calculate bounds centered on target
            all_vals = np.append(sample_vals, target_val)
            data_min, data_max = np.min(all_vals), np.max(all_vals)
            data_range = data_max - data_min
            
            if data_range == 0:
                data_range = abs(target_val) * 0.1 if target_val != 0 else 1.0
            
            margin = data_range * default_margin_pct
            
            if center_on_target:
                # Center on target: find max distance from target
                max_dist = max(
                    abs(data_max - target_val),
                    abs(data_min - target_val)
                ) + margin
                ymin = target_val - max_dist
                ymax = target_val + max_dist
            else:
                ymin = data_min - margin
                ymax = data_max + margin
        
        # --- Plot samples as scatter points ---
        for i, (idx, sample_row) in enumerate(df.iterrows()):
            fig.add_trace(
                go.Scatter(
                    x=[sample_labels[i]],
                    y=[sample_row[param]],
                    mode='markers',
                    marker=dict(
                        size=marker_size,
                        color=sample_colors[i % len(sample_colors)],
                        opacity=sample_opacity,
                        line=dict(width=1, color='white')
                    ),
                    name=f"Sample {i+1}",
                    legendgroup=f"sample_{i+1}",
                    showlegend=(j == 0),
                    hovertemplate=(
                        f"<b>Sample {i+1}</b><br>"
                        f"{param}: %{{y:.4g}}<br>"
                        f"Index: {idx}<extra></extra>"
                    )
                ),
                row=row, col=col
            )
        
        # --- Plot target as horizontal line ---
        fig.add_trace(
            go.Scatter(
                x=sample_labels,
                y=[target_val] * n_samples,
                mode='lines',
                line=dict(color=target_color, width=2, dash=target_dash),
                name='Target',
                legendgroup='target',
                showlegend=(j == 0),
                hovertemplate=(
                    f"<b>Target</b><br>"
                    f"{param}: %{{y:.4g}}<extra></extra>"
                )
            ),
            row=row, col=col
        )
        
        # --- Update y-axis for this subplot ---
        yaxis_key = f"yaxis{j+1}" if j > 0 else "yaxis"
        fig.update_layout(**{yaxis_key: dict(range=[ymin, ymax])})
    
    # --- Final layout ---
    if title:
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            height=figsize_per_plot[1] * n_rows,
            width=figsize_per_plot[0] * n_cols,
            hovermode="closest",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
    else:
        fig.update_layout(
            height=figsize_per_plot[1] * n_rows,
            width=figsize_per_plot[0] * n_cols,
            hovermode="closest",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
    )
    
    # Hide x-axis labels for cleaner look (optional)
    # fig.update_xaxes(showticklabels=True, tickangle=45)
    fig.update_xaxes(showticklabels=False)
    
    if save_filename:
        fig.write_html(save_filename) if save_filename.endswith('.html') else fig.write_image(save_filename)
    
    return fig

# def plot_histogram_distribution(
#     df: pd.DataFrame,
#     label_mapping: Optional[dict] = None,
#     y_range: tuple = (-1.0, 1.0),
#     n_bins: int = 30,
#     bar_width: float = 0.8,
#     fill_color: str = "rgba(31, 119, 180, 0.5)",
#     line_color: str = "#1f77b4",
#     stat_color: str = "black",
#     y_title: str = "Relative Difference",
#     width: int = 1000,
#     height: int = 500,
#     show_quartiles: bool = False,
#     show_mean: bool = True,
#     save_path: Optional[str] = None,
#     x_label: bool = False
# ) -> go.Figure:
#     """
#     Plot horizontal histograms with statistical overlays for each column.
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame where each column is a feature to plot.
#     label_mapping : dict, optional
#         Mapping from column names to display labels.
#     y_range : tuple
#         (min, max) range for the y-axis and histogram bins.
#     n_bins : int
#         Number of histogram bins.
#     bar_width : float
#         Width allocated for each histogram (0-1 scale).
#     fill_color : str
#         Fill color for histogram bars.
#     line_color : str
#         Outline color for histogram bars.
#     stat_color : str
#         Color for mean and quartile lines.
#     y_title : str
#         Label for y-axis.
#     width : int
#         Figure width in pixels.
#     height : int
#         Figure height in pixels.
#     show_quartiles : bool
#         Whether to display Q1/Q3 dashed lines.
#     show_mean : bool
#         Whether to display mean solid line.
#     save_path : str, optional
#         If provided, save figure to this path.
        
#     Returns
#     -------
#     go.Figure
#         The plotly figure object.
#     """
    
#     fig = go.Figure()
    
#     # Handle label mapping
#     if label_mapping is None:
#         label_mapping = {}
#     custom_labels = [label_mapping.get(col, col) for col in df.columns]
    
        
    
#     for i, col in enumerate(df.columns):
#         data = df[col].dropna()
        
#         # Calculate histogram
#         counts, bin_edges = np.histogram(data, bins=n_bins, range=y_range)
#         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
#         # Normalize counts
#         max_count = counts.max() if counts.max() > 0 else 1
#         normalized_counts = (counts / max_count) * (bar_width / 2)
        
#         # Create horizontal histogram
#         for j, (y_center, w) in enumerate(zip(bin_centers, normalized_counts)):
#             if w > 0:
#                 fig.add_shape(
#                     type="rect",
#                     x0=i - w,
#                     x1=i + w,
#                     y0=bin_edges[j],
#                     y1=bin_edges[j + 1],
#                     fillcolor=fill_color,
#                     line=dict(color=line_color, width=0.5),
#                 )
        
#         # Calculate statistics
#         mean = np.mean(data) # update variable name for consistency
#         q1 = np.percentile(data, 25) # remove 
#         q3 = np.percentile(data, 75)
        
#         # Add mean line (solid)
#         if show_mean:
#             fig.add_shape(
#                 type="line",
#                 x0=i - bar_width/2, x1=i + bar_width/2,
#                 y0=mean, y1=mean,
#                 line=dict(color=stat_color, width=2),
#             )
        
#         # Add quartile lines (dashed)
#         if show_quartiles:
#             for q in [q1, q3]:
#                 fig.add_shape(
#                     type="line",
#                     x0=i - bar_width/2, x1=i + bar_width/2,
#                     y0=q, y1=q,
#                     line=dict(color=stat_color, width=1.5, dash="dash"),
#                 )
    
#     # Calculate y-axis display range with padding
#     y_padding = (y_range[1] - y_range[0]) * 0.05
#     y_display_range = [y_range[0] - y_padding, y_range[1] + y_padding]
    
#     # Update layout
#     fig.update_layout(
#         yaxis_title=y_title,
#         yaxis_range=y_display_range,
#         xaxis=dict(
#             tickmode='array',
#             tickvals=list(range(len(df.columns))),
#             ticktext=custom_labels if x_label else [], # Only provide text if x_label is True
#             showticklabels=x_label,
#             range=[-0.5, len(df.columns) - 0.5],
#             tickangle=45,
#         ),
#         width=width,
#         height=height,
#         showlegend=False,
#         margin=dict(t=20, b=100 if x_label else 20),
#         font=dict(
#             family="Times New Roman, serif",
#             size=20,
#             color="black"
#         )
#     )
    
#     if save_path:
#         fig.write_image(save_path)
    
#     return fig


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional

def plot_histogram_distribution(
    df: pd.DataFrame,
    label_mapping: Optional[dict] = None,
    y_range: tuple = (-1.0, 1.0),
    n_bins: int = 30,
    bar_width: float = 0.8,
    fill_color: str = "rgba(31, 119, 180, 0.5)",
    line_color: str = "#1f77b4",
    stat_color: str = "black",
    y_title: str = "Relative Difference",
    width: int = 1200,
    height: int = 400,
    show_quartiles: bool = False,
    show_mean: bool = True,
    save_path: Optional[str] = None,
    x_label: bool = False
) -> go.Figure:
    
    fig = go.Figure()
    
    # Handle label mapping
    if label_mapping is None:
        label_mapping = {}
    custom_labels = [label_mapping.get(col, col) for col in df.columns]
    
    for i, col in enumerate(df.columns):
        data = df[col].dropna()
        
        # Calculate histogram
        counts, bin_edges = np.histogram(data, bins=n_bins, range=y_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Normalize counts
        max_count = counts.max() if counts.max() > 0 else 1
        normalized_counts = (counts / max_count) * (bar_width / 2)
        
        # Create horizontal histogram using shapes
        for j, (y_center, w) in enumerate(zip(bin_centers, normalized_counts)):
            if w > 0:
                fig.add_shape(
                    type="rect",
                    x0=i - w, x1=i + w,
                    y0=bin_edges[j], y1=bin_edges[j + 1],
                    fillcolor=fill_color,
                    line=dict(color=line_color, width=0.5),
                )
        
        # Calculate statistics
        mean = np.mean(data)
        
        # Add mean line (solid)
        if show_mean:
            fig.add_shape(
                type="line",
                x0=i - bar_width/2, x1=i + bar_width/2,
                y0=mean, y1=mean,
                line=dict(color=stat_color, width=2),
            )
        
        # Add quartile lines (dashed)
        if show_quartiles:
            q1, q3 = np.percentile(data, [25, 75])
            for q in [q1, q3]:
                fig.add_shape(
                    type="line",
                    x0=i - bar_width/2, x1=i + bar_width/2,
                    y0=q, y1=q,
                    line=dict(color=stat_color, width=1.5, dash="dash"),
                )

    # --- ADD DUMMY TRACES FOR LEGEND ---
    # These won't draw data but will create the legend entries
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(symbol='square', size=15, color=fill_color),
        name='Property<br>Distribution'
    ))

    if show_mean:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color=stat_color, width=2),
            name='Mean'
        ))

    # Calculate y-axis display range with padding
    y_padding = (y_range[1] - y_range[0]) * 0.05
    y_display_range = [y_range[0] - y_padding, y_range[1] + y_padding]
    
    # Update layout
    fig.update_layout(
        yaxis_title=y_title,
        yaxis_range=y_display_range,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(df.columns))),
            ticktext=custom_labels,
            tickfont_color='black' if x_label else 'white',
            range=[-0.5, len(df.columns) - 0.5],
            tickangle=45,
        ),
        showlegend=True,
        legend=dict(
            title=dict(text="<b>Over 1,000 samples</b>", font=dict(size=18)),
            bordercolor="Black",
            borderwidth=1,
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02 # Places legend outside to the right
        ),
        width=width,
        height=height,
        margin=dict(
            t=10, 
            b=100 if x_label else 40,
            l=80,
            r=250
        ),
        font=dict(
            family="Times New Roman, serif",
            size=20,
            color="black"
        )
    )
    
    if save_path:
        fig.write_image(save_path)
    
    return fig

def _extract_degrees_from_file(filepath):
    try:
        G = nx.read_graphml(filepath)
        if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
            in_deg = [int(round(d)) for _, d in G.in_degree()]
            out_deg = [int(round(d)) for _, d in G.out_degree()]
            return (in_deg, out_deg)
        return [], []
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return [], []

def load_degrees_parallel(sim_folder, pattern="*.graphml", max_samples=100):
    files = sorted(glob.glob(os.path.join(sim_folder, pattern)))
    if not files: return [], []
    if len(files) > max_samples:
        files = random.sample(files, max_samples)
    with Pool(cpu_count()) as pool:
        results = pool.map(_extract_degrees_from_file, files)
    return [r[0] for r in results if r[0]], [r[1] for r in results if r[1]]

def to_points(arr, xmin):
    arr = np.array(arr)
    filtered = arr[arr >= xmin]
    if len(filtered) == 0: return np.array([]), np.array([])
    vals, counts = np.unique(filtered, return_counts=True)
    return vals, counts / len(filtered)

def plot_deg_ref_vs_multi_sim_2panels(ref_g, sim_folder, xmin=1, 
                                       ref_label="Reference", save_path=None):
    ref_in = [int(round(d)) for _, d in ref_g.in_degree()]
    ref_out = [int(round(d)) for _, d in ref_g.out_degree()]
    sim_in_list, sim_out_list = load_degrees_parallel(sim_folder)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False) 
    
    labels = ["In-degree Distribution", "Out-degree Distribution"]
    ref_arrs = [ref_in, ref_out]
    sim_lists = [sim_in_list, sim_out_list]
    colors = ["tab:orange", "tab:green"]

    for ax, lab, col, r_arr, s_list in zip(axes, labels, colors, ref_arrs, sim_lists):
        # 1. Plot Individual Simulations (Dashed dimgray lines)
        for s_arr in s_list:
            xs, ys = to_points(s_arr, xmin)
            if xs.size:
                ax.plot(xs, ys, color="dimgray", alpha=0.15, lw=1.0, ls="--")

        # 2. Plot Reference (Scatter/Points)
        xr, yr = to_points(r_arr, xmin)
        if xr.size:
            # We use scatter here so simulation lines remain visible behind/around points
            ax.scatter(xr, yr, color=col, s=50, label=f'{ref_label}', zorder=10)

        # Formatting
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(lab, pad=20)
        ax.set_xlabel("Degree (d)")
        ax.set_ylabel("P(d)")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='x')
        
        # 3. PROXY ARTIST LEGEND 
        # Update legend to show a line for Simulations and a dot for Reference
        legend_elements = [
            Line2D([0], [0], color='dimgray', lw=2, ls='--', label='Random Graphs'),
            Line2D([0], [0], marker='o', color='w', label=f'{ref_label}',
                   markerfacecolor=col, markersize=10, linestyle='None')
        ]
        ax.legend(handles=legend_elements, frameon=False, loc='best')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()