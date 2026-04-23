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

def main():
    specie_list = [
        'yeast','hESC', 'mESC', 
        'ecoli_gnw', 
        'human_trrust', 
        "ecoli_wolf",
        'mDC'
        ]

    properties_to_keep = [
        'graph_id',
        'avg_path_dir', 'diameter',
        'nb_edges', 
        'Cascade',
        'FFL', 'Fan-In', 'Fan-Out',
        'Mutual-In', 'Mutual-Out',
        'Regulating-Mutual'
    ]

    idces_list = {}

    for specie in specie_list:
        input_dir = f"../../data/graphs/" # reference graph and associated data folder path
        ground_truth_graph, ground_truth_stat, ground_truth_motifs = load_graphs(input_dir+specie)
        node_degree_sequence = get_node_degrees(ground_truth_graph)

        ngraphs = 1000 # number of graph to generate
        n_jobs = os.cpu_count() - 1

        output_dir = f"../../data/random_graphs/{specie}" 

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        generate_random_graphs(
                ngraphs,
                node_degree_sequence,
                f'{output_dir}/{specie}_results.parquet',
                specie,
                connect_type='random',
                method='grngen',
            )

        ground_truth_all_stats = ground_truth_stat | ground_truth_motifs

        # Load random data from Parquet
        parquet_path = f'{output_dir}/{specie}_results.parquet'
        df_random = pd.read_parquet(parquet_path, columns=properties_to_keep)

        # Computing errors
        gt_prop = {
            key: value 
            for key, value in ground_truth_all_stats.items()
            if key in properties_to_keep
        }

        # Computing errors
        df_errors = compute_relative_error(df_random, gt_prop)
        df_errors.to_csv(f"{output_dir}/relative_error.csv", index=False)

        # Select best graph
        idces_list[specie] = get_best_indices(df_errors, n_top=5)
        print(f'Best graph index for dataset {specie}: {df_errors["graph_id"][idces_list[specie][0]]}')

        ################## Plot relative histogram ########################
        # Prepare data
        df = df_errors.drop(['total_error', 'graph_id'], axis=1)

        # Define labels
        label_mapping = {
            'avg_path_dir': "Average Path Length", 
            'diameter': "Diameter",
            'nb_edges': "Arc", 
            'Cascade': "Cascade", 
            'FFL': "Feed Forward Loop", 
            'Fan-In': "Fan-In",
            'Fan-Out': "Fan-Out", 
            'Mutual-In': "Mutual-In",
            'Mutual-Out': "Mutual-Out",
            'Regulating-Mutual': "Regulating-Mutual",
        }


        fig1 = plot_histogram_distribution(
            df,
            label_mapping=label_mapping,
            n_bins=40,
            save_path=f"{output_dir}/properties_distribution_{specie}.pdf"
        )
        fig1.write_image(f"{output_dir}/properties_distribution_{specie}.svg")
        fig1.write_image(f"{output_dir}/properties_distribution_{specie}.png")

        ############# Plot best profiles ##############
        fig2 = plot_best_profiles(
            df_errors, 
            idces_list[specie], 
            specie, 
            label_mapping, 
            output_dir
        )

        plot_deg_ref_vs_multi_sim(
                ground_truth_graph, 
                parquet_path,
                ref_label=specie,
                save_path=f"{output_dir}/degree_distribution_{specie}.pdf", 
            )
        
if __name__ == "__main__":
    main()