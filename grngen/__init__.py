"""GRNgen: Relaxed Directed Configuration Model for GRN generation."""

__version__ = "0.1.0"

from .process_data import get_node_degrees, load_graphs, compute_relative_error, get_best_indices, compute_network_properties, count_motifs, get_largest_cc, load_graphs_from_parquet
from .random_graph import generate_random_graphs, generate_one_graph_profiled, connect_components
from .plot import plot_histogram_distribution, plot_deg_ref_vs_multi_sim, plot_best_profiles, scatter_with_correlation
from .feature_processing import select_optimal_graphs, evaluate_selection, plot_hist_comparison, plot_var_summary, plot_std_summary, select_optimal_graphs_fast, compute_cv_robust, norm_min_max, norm_mad
from .inference import evaluate_inference
from .sim import randomize_edge_signs, simulate_multi_bool_parallel