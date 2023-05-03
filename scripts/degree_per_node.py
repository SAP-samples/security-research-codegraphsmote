import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch_geometric.utils import add_remaining_self_loops, coalesce
from tqdm import tqdm

from utils import get_dataset, get_ae_model
from params import CLASSIFICATION_DATASET


def degree_of(graph):
    edge_index, _ = add_remaining_self_loops(graph.edge_index)
    edge_index = coalesce(edge_index)
    return edge_index.shape[1] / graph.num_nodes


def plot_ae_nonae(params, args):
    params["overwrite_cache"] = False
    params["dataset"]["overwrite_cache"] = False
    params["sample_type"] = args.sample_type
    
    dataset = get_dataset(params["dataset"])

    model = None
    if args.uncached:
        params["model"]["encoder"]["features"] = dataset.get_input_size()
        model = get_ae_model(params["model"]["encoder"], params["model"]["decoder"])
        model.load(params["model"]["checkpoint"])

    if args.uncached and hasattr(model.decoder.adj_decoder, "predicted_variance"):
        print("Predicted variance", model.decoder.adj_decoder.predicted_variance)

    degrees_orig = []
    node_counts = []
    for index, graph in enumerate(tqdm(itertools.islice(dataset, 5000), total=min(len(dataset), 5000))):
        node_counts.append(graph.num_nodes)
        degrees_orig.append(degree_of(graph))
    
    plt.scatter(node_counts, degrees_orig, c="b", alpha=0.1)


if __name__ == "__main__":
    parser = ArgumentParser(description=
        "Create plot to visualize the distribution of average node degree over graph size"
    )

    parser.add_argument('--repetitions', "-r", type=int, default=10,
        help="Number of repetitions per graph to smooth over")
    parser.add_argument('--sample_type', '-s', default="random",
        choices=["random", "threshold"], help="How to sample the adjacency")
    parser.add_argument("-p", "--path", default="results/node_degree.png",
        help="Output path for plot")
    parser.add_argument('--uncached', action="store_true",
        help="Re-Generate decoded graph instead of using cache")
    args = parser.parse_args()
    
    print(f"Starting {CLASSIFICATION_DATASET['name']}")
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)
    plt.rc("axes", labelsize=23)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Average Degree")
    plot_ae_nonae(CLASSIFICATION_DATASET, args)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(args.path)