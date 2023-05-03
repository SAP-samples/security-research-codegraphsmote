import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import tokenizers
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse

from tqdm import tqdm

from params import PATCHDB_TRANSFORMER, ENCODER_PARAMS, COMPOSITE_GDN
from utils import get_dataset, get_ae_model
from experiments.datasets.vulnerability_dataset_utils import read_dot
from experiments.autoencoding.helper_types import TupleGraph
from dataloader import CPGReconstructionDataset
from model import Transformer, MAX_LEN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = tokenizers.Tokenizer.from_pretrained("Salesforce/codegen-350M-multi")
tokenizer.add_special_tokens(["<|startoftext|>", "<|pad|>"])

END_TOK = tokenizer.token_to_id("<|endoftext|>")
PAD_TOK = tokenizer.token_to_id("<|pad|>")

def load_torch(p, dataset):
    with open(p, "r") as f:
        dot = read_dot(f, True)
        dot.graph["label"] = 0.5 # label unknown

        data = dataset.encode(dot, p, c_in_dot=True)

        xs = [data.astenc, data.codeenc]
        if dataset.bounds:
            xs.extend([data.lowerBound, data.upperBound])
        data.x = torch.cat(xs, dim=-1).float()

        return data

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

model = Transformer(num_tokens=tokenizer.get_vocab_size(),
        dim_model=128,
        num_heads=2,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout_p=0.1,
        padding_token=PAD_TOK).cuda()

model.load_state_dict(torch.load(os.path.join("cache", "cpg_reconstruction", "models", "model_9.chkpt")))
model.eval()

PATCHDB_TRANSFORMER["overwrite_cache"] = False
dataset = get_dataset(PATCHDB_TRANSFORMER)
ENCODER_PARAMS["features"] = dataset.get_input_size()
ae_model = get_ae_model(ENCODER_PARAMS, COMPOSITE_GDN)
ae_model.load("results/GDN_VAE_COMPOSITE_GDN_VAE_PATCHDB_TRANSFORMER_reveal/checkpoint")
ae_model.eval()
ae_model.to(device)


load_set = CPGReconstructionDataset(None, tokenizer)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOK)
with torch.no_grad():
    pbar = tqdm(zip(dataset.get_graph_identifiers(), dataset), mininterval=1, total=len(dataset))
    total_loss = 0
    total_loss_ae = 0
    count = 0
    for p, G in pbar:
        X, y = load_set.load(p)

        X = [[x.cuda() for x in X]]
        y = y.cuda().unsqueeze(0)

        y_input = y[:,:-1]
        y_expected = y[:,1:]

        # G = load_torch(p, dataset).to(device)
        G.to(device)
        G = autoencode(ae_model, G, 10)

        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        if sequence_length > MAX_LEN:
            print("skipped sequence with length {}".format(sequence_length))
            continue
        tgt_mask = model.get_tgt_mask(sequence_length)

        pred = model(X, y_input, tgt_mask)
        pred = pred.view(-1, tokenizer.get_vocab_size())
        y_expected = y_expected.reshape(-1)
        loss = loss_fn(pred, y_expected)

        count += 1
        loss = loss.detach().cpu().item()
        total_loss += loss

        pred_ae = model.pred_nodes(G.x[..., -128:].unsqueeze(0), y_input, tgt_mask)
        pred_ae = pred_ae.view(-1, tokenizer.get_vocab_size())
        loss_ae = loss_fn(pred_ae, y_expected)
        total_loss_ae += loss_ae.detach().cpu().item()

        pbar.set_description(f"loss {total_loss/count:.3f} ae {total_loss_ae/count:.3f}", refresh=False)
    print(f"Loss w/o VAE: {total_loss/count:.3f}")
    print(f"Loss w/  VAE: {total_loss_ae/count:.3f}")