import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import math
import itertools
import random

from collections import defaultdict

import seaborn as sns
from tqdm import tqdm
from scipy.spatial.distance import minkowski

from experiments.datasets.smote import pad
from params import PATCHDB_AUTOENCODED, PATCHDB_PARAMS
from utils import get_dataset


NUM_DISTANCE_SAMPLES = 1000


def distance(d, graph_id1, graph_id2):
    encoded1 = d._get_encoded_by_graph_id(graph_id1)
    encoded2 = d._get_encoded_by_graph_id(graph_id2)

    u = encoded1
    v = encoded2

    num_nodes = max(u.shape[0], v.shape[0])
    u_pad = pad(u, num_nodes).view(-1)
    v_pad = pad(v, num_nodes).view(-1)
    return float(minkowski(u_pad, v_pad) / (num_nodes**0.5))


def inter_cluster_distance(d, cluster1, cluster2, samples=NUM_DISTANCE_SAMPLES):
    size1 = len(cluster1)
    size2 = len(cluster2)
    combinations = list(itertools.product(range(size1), range(size2)))
    
    max_samples_possible = len(combinations)
    if max_samples_possible < samples:
        samples = max_samples_possible
    
    summed_distance = 0
    for (index1, index2) in random.sample(combinations, samples):
        graph_id1 = cluster1[index1]
        graph_id2 = cluster2[index2]
        summed_distance += distance(d, graph_id1, graph_id2)
    
    return summed_distance / samples


def intra_cluster_distance(d, cluster, samples=NUM_DISTANCE_SAMPLES):
    size = len(cluster)
    combinations = list(itertools.combinations(range(size), 2))
    
    max_samples_possible = len(combinations)
    if max_samples_possible < samples:
        samples = max_samples_possible
    
    summed_distance = 0
    for (index1, index2) in random.sample(combinations, samples):
        graph_id1 = cluster[index1]
        graph_id2 = cluster[index2]
        summed_distance += distance(d, graph_id1, graph_id2)
    
    return summed_distance / samples


def main():
    PATCHDB_AUTOENCODED["overwrite_cache"] = False
    d = get_dataset(PATCHDB_AUTOENCODED)

    with open(os.path.join(PATCHDB_PARAMS["dataset_dir"], "cve2cwe.json"), "r") as f:
        cve2cwe = json.load(f)
    
    with open(os.path.join(PATCHDB_PARAMS["dataset_dir"], "patchdb-cves.csv"), "r") as f:
        cves = f.readlines()[1:]
    
    cves = [ \
        {"cve": cve[0], "commit": cve[1], "project": os.path.split(cve[2])[1].strip()[:-len(".git")], "cwes": cve2cwe.get(cve[0])} \
        for cve in (row.split(",") for row in cves) \
        ]

    graph_ids = [ \
        {"project": id_parts[0], "commit": id_parts[1], "graph_id": graph_id} \
        for (id_parts, graph_id) in ((os.path.split(graph_id)[1].split("_"), graph_id) for graph_id in d.dataset.get_graph_identifiers()) \
        if int(id_parts[-1].split(".")[0]) == 1
        ]

    project_commit_to_cwes = {(cve["project"], cve["commit"]): cve["cwes"] for cve in cves}
    cwes = defaultdict(list)

    for graph_id in graph_ids:
        this_cwes = project_commit_to_cwes.get((graph_id["project"], graph_id["commit"]))
        if this_cwes is None or len(this_cwes) == 0:
            continue
        for cwe in this_cwes:
            cwes[cwe].append(graph_id["graph_id"])
    
    cwes = {cwe: cwe_list for (cwe, cwe_list) in cwes.items() if len(cwe_list) > 10}
    
    distances = {}
    print("Generating intra-cluster distances")
    for (cwe, cluster) in tqdm(cwes.items()):
        distances[(cwe, cwe)] = intra_cluster_distance(d, cluster)
    print("Generating inter-cluster distances")
    for (cwe1, cwe2) in tqdm(list(itertools.combinations(cwes.keys(), 2))):
        distances[(cwe1, cwe2)] = inter_cluster_distance(d, cwes[cwe1], cwes[cwe2])
        distances[(cwe2, cwe1)] = distances[(cwe1, cwe2)]
    
    with open("results/distances.json", "w") as f:
        json.dump({"_".join(k): v for k,v in distances.items()}, f)

    # normalize
    for cwe1 in cwes.keys():
        for cwe2 in cwes.keys():
            if cwe1 == cwe2:
                continue
            intra1 = distances[(cwe1, cwe1)]
            intra2 = distances[(cwe2, cwe2)]
            inter = distances[(cwe1, cwe2)]
            difference = intra1 - inter
            distances[(cwe1, cwe2)] =  math.copysign(difference, difference) / (intra1 * intra2)
    for cwe in cwes.keys():
        distances[(cwe, cwe)] = 0.0

    sorted_cwes = list(sorted(cwes.keys(), key=lambda k: int(k.split("-")[-1])))
    distances_arr = [[distances[(cwe1, cwe2)] for cwe2 in sorted_cwes] for cwe1 in sorted_cwes]
    ax = sns.heatmap(
        distances_arr,
        xticklabels=sorted_cwes,
        yticklabels=sorted_cwes,
        center=0.0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    fig = ax.get_figure()
    fig.savefig("results/distances.pdf")


if __name__ == "__main__":
    main()
