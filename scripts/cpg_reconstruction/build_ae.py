import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import gzip
import pickle

import torch
import tokenizers
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse

from tqdm import tqdm

from params import PATCHDB_TRANSFORMER, ENCODER_PARAMS, COMPOSITE_GDN
from utils import get_dataset, get_ae_model
from experiments.autoencoding.helper_types import TupleGraph


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autoencode(model, graph, repetitions):
    num_nodes = graph.num_nodes
    device = graph.x.device
    dtype = graph.x.dtype
    hidden_channels = graph.x.shape[1]

    decoded_adj = torch.zeros((num_nodes, num_nodes), device=device, dtype=dtype)
    decoded_features = torch.zeros((num_nodes, hidden_channels), device=device, dtype=dtype)
    for i in range(repetitions):
        latent = model.encode(graph)
        latent = latent["mu"] + torch.randn_like(latent["logstd"]) * torch.exp(latent["logstd"])

        decoded = model.decode(TupleGraph(
            x=latent,
            num_nodes=num_nodes,
            batch=torch.zeros((num_nodes,), device=device, dtype=torch.long)
        ))
        decoded_adj += torch.sigmoid(decoded.adj)
        decoded_features += decoded.x
    
    decoded.adj = decoded_adj / repetitions
    decoded.x = decoded_features / repetitions
    decoded_adj = None
    decoded_features = None
    del decoded_adj
    del decoded_features
    
    rand_sample = torch.rand_like(decoded.adj)
    decoded.adj[decoded.adj >= rand_sample] = 1
    decoded.adj[decoded.adj < rand_sample] = 0
    decoded.edge_index, _ = dense_to_sparse(decoded.adj.detach())
    decoded.adj = None

    decoded.x[..., -128:] = F.normalize(decoded.x[..., -128:], p=2, dim=-1)

    return decoded

PATCHDB_TRANSFORMER["overwrite_cache"] = False
dataset = get_dataset(PATCHDB_TRANSFORMER)
ENCODER_PARAMS["features"] = dataset.get_input_size()
ae_model = get_ae_model(ENCODER_PARAMS, COMPOSITE_GDN)
ae_model.load("results/GDN_VAE_COMPOSITE_GDN_VAE_PATCHDB_TRANSFORMER_reveal/checkpoint")
ae_model.eval()
ae_model.to(device)

with torch.no_grad():
    pbar = tqdm(zip(dataset.get_graph_identifiers(), dataset), mininterval=1, total=len(dataset))
    for p, G in pbar:
        h = p.split("/")[-1].replace(".cpg","")

        G.to(device)
        G = autoencode(ae_model, G, 10)

        with gzip.open(f"cache/cpg_reconstruction/ae/{h}.gz", "wb") as f:
            pickle.dump(G.x[..., -128:].detach().cpu().numpy(), f)