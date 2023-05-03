import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from experiments.datasets.vulnerability_dataset import get_cwe_subsets

from params import CLASSIFICATION_DATASET
from utils import get_dataset


def pool(X):
    X = torch.mean(X, axis=0)
    X = X.cpu().numpy()
    return X

def main():
    CLASSIFICATION_DATASET["overwrite_cache"] = False
    dataset = get_dataset(CLASSIFICATION_DATASET)

    graph_ids = set(dataset.get_graph_identifiers())

    cwes = dataset.dataset.dataset.get_cwes()
    cwes = dict((k, [os.path.split(g)[1] for g in v]) for (k,v) in cwes.items())
    cwes = dict((k,v) for (k,v) in cwes.items() if 10 < len(v))
    cwes = dict((k,[g for g in v if g in graph_ids][:35]) for (k,v) in cwes.items())

    X_cwes = {
        cwe: [pool(dataset._get_encoded_by_graph_id(graph_id)) for graph_id in subset]
        for cwe, subset in cwes.items()
    }
    X = []
    Y = []
    cwe_map = dict((cwe, i) for (i, cwe) in enumerate(X_cwes.keys()))
    for cwe, embeddings in X_cwes.items():
        X.extend(embeddings)
        Y.extend([cwe_map[cwe]] * len(embeddings))
    X = np.stack(X, axis=0)
    tsne = TSNE(
        n_components=2,
        n_jobs=4,
    )
    print("Starting t-SNE")
    X_new = tsne.fit_transform(X)
    X_new = StandardScaler().fit_transform(X_new)
    
    plt.axis("off")
    colors = ListedColormap(cm.rainbow(np.linspace(0, 1, len(X_cwes.keys()))))
    plt.scatter(X_new[:, 0], X_new[:, 1], s=5, c=Y, cmap=colors, alpha=0.6)

    plt.savefig(f"results/{CLASSIFICATION_DATASET['name']}_cwes.png", bbox_inches="tight", dpi=500)

if __name__ == "__main__":
    main()