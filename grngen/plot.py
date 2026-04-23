import math
from collections import Counter
import os
import json
from typing import Optional
import random
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import glob

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
            x=1.02
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

def to_points(arr, xmin):
    arr = np.array(arr)
    filtered = arr[arr >= xmin]
    if len(filtered) == 0: return np.array([]), np.array([])
    vals, counts = np.unique(filtered, return_counts=True)
    return vals, counts / len(filtered)

def plot_deg_ref_vs_multi_sim(ref_g, parquet_path, xmin=1, 
                                        ref_label="Reference", save_path=None):
    
    ref_in = [int(round(d)) for _, d in ref_g.in_degree()]
    ref_out = [int(round(d)) for _, d in ref_g.out_degree()]

    print(f"Data loaded from {parquet_path}...")
    df = pd.read_parquet(parquet_path, columns=['graph_structure'])
    
    sim_in_list = []
    sim_out_list = []

    for json_str in df['graph_structure']:
        if json_str:
            adj = json.loads(json_str)
            
            out_degrees = [len(neighbors) for neighbors in adj.values()]
            in_counts = {str(node): 0 for node in adj.keys()}
            
            for neighbors in adj.values():
                for n in neighbors:
                    n_str = str(n)
                    if n_str in in_counts:
                        in_counts[n_str] += 1
            
            sim_in_list.append(list(in_counts.values()))
            sim_out_list.append(out_degrees)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False) 
    
    labels = ["In-degree Distribution", "Out-degree Distribution"]
    ref_arrs = [ref_in, ref_out]
    sim_lists = [sim_in_list, sim_out_list]
    colors = ["tab:orange", "tab:green"]

    for ax, lab, col, r_arr, s_list in zip(axes, labels, colors, ref_arrs, sim_lists):
        for s_arr in s_list:
            xs, ys = to_points(s_arr, xmin)
            if xs.size:
                ax.plot(xs, ys, color="dimgray", alpha=0.15, lw=1.0, ls="--")

        xr, yr = to_points(r_arr, xmin)
        if xr.size:
            ax.scatter(xr, yr, color=col, s=50, label=f'{ref_label}', zorder=10)

        # Formatting
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(lab, pad=20)
        ax.set_xlabel("Degree (d)")
        ax.set_ylabel("P(d)")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='x')
        
        # 3. Legend
        legend_elements = [
            Line2D([0], [0], color='dimgray', lw=2, ls='--', label='Random Graphs'),
            Line2D([0], [0], marker='o', color='w', label=f'{ref_label}',
                   markerfacecolor=col, markersize=10, linestyle='None')
        ]
        ax.legend(handles=legend_elements, frameon=False, loc='best')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
def plot_best_profiles(df_errors, idces, specie, label_mapping, output_dir=None):
    """
    Plots the relative difference profile for the top N best-performing graphs.
    """
    # Prepare the data for the specific indices
    df_best = df_errors.loc[idces].drop(['total_error', 'graph_id'], axis=1)
    df_best = df_best.rename(columns=label_mapping)
    params = list(df_best.columns)

    fig = go.Figure()

    legend_labels = ['1st', '2nd', '3rd', '4th', '5th']
    
    for i, (idx, row) in enumerate(df_best.iterrows()):
        name = legend_labels[i] if i < len(legend_labels) else f"{i+1}th"
        fig.add_trace(go.Scatter(
            x=params,
            y=row[params].tolist(),
            mode='lines+markers',
            name=name,
            opacity=0.5,
            hovertemplate="Sample %{text}<br>%{x}: %{y:.4f}",
            text=[idx] * len(params)
        ))

    fig.update_layout(
        font=dict(family="Times New Roman, serif", size=20, color="black"),
        legend=dict(
            title=dict(text="<b>Top 5 GRN</b>", font=dict(size=18)),
            bordercolor="Black",
            borderwidth=1,
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        yaxis_title="Relative Difference",
        hovermode="x unified",
        margin=dict(t=10, b=100, l=80, r=250),
        width=1200,
        height=400,
        yaxis_range=[-1, 1],
        xaxis=dict(tickangle=25)
    )

    # Export images
    if output_dir:
        base_name = f"{output_dir}/5_profile_relative_{specie}"
        for ext in ['pdf', 'svg', 'png']:
            fig.write_image(f"{base_name}.{ext}")
    
    return fig