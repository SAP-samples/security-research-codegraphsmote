import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import math
import subprocess

import torch
import tokenizers
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse

from experiments.datasets.vulnerability_dataset_utils import read_dot
from utils import get_dataset, get_ae_model
from params import PATCHDB_TRANSFORMER, ENCODER_PARAMS, COMPOSITE_GDN
from experiments.autoencoding.helper_types import TupleGraph

from dataloader import CPGReconstructionDataset
from model import Transformer, MAX_LEN


CPG_TO_DOT_PATH = "cpg-to-dot/build/install/cpg-to-dot/bin/cpg-to-dot"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = tokenizers.Tokenizer.from_pretrained("Salesforce/codegen-350M-multi")
tokenizer.add_special_tokens(["<|startoftext|>", "<|pad|>"])

PAD_TOK = tokenizer.token_to_id("<|pad|>")
END_TOK = tokenizer.token_to_id("<|endoftext|>")

model = Transformer(num_tokens=tokenizer.get_vocab_size(),
        dim_model=128,
        num_heads=2,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout_p=0.1,
        padding_token=PAD_TOK).cuda()

model.load_state_dict(torch.load(os.path.join("cache", "cpg_reconstruction", "models", "model_9__.chkpt")))
model.eval()

load_set = CPGReconstructionDataset(None, tokenizer)

PATCHDB_TRANSFORMER["overwrite_cache"] = False
encode_set = get_dataset(PATCHDB_TRANSFORMER)
ENCODER_PARAMS["features"] = encode_set.get_input_size()
ENCODER_PARAMS["edge_dim"] = encode_set.get_edge_size()
ae_model = get_ae_model(ENCODER_PARAMS, COMPOSITE_GDN)
ae_model.load("results/GDN_VAE_COMPOSITE_GDN_VAE_PATCHDB_TRANSFORMER_reveal/checkpoint")
ae_model.eval()
ae_model.cuda()

def decode_tokens(tokens):
    return tokenizer.decode(tokens, skip_special_tokens=False)


def c_to_cpg(c_text):
    with open("cache/cpg_reconstruction/demo.c", "w") as f:
        f.write(c_text)
    out = subprocess.run(
                [CPG_TO_DOT_PATH, "--file" ,"./cache/cpg_reconstruction/demo.c", "--output", f"./cache/cpg_reconstruction/"], 
                stdout=subprocess.PIPE,
                timeout=180).stdout.decode('utf-8')


@torch.no_grad()
def beam_decode(src, seq_len, first_token, output_queue, end_token=END_TOK, pad_token=PAD_TOK, beam_width=5):
    tgt = first_token.unsqueeze(0).unsqueeze(0)
    alpha = 1.0
    
    pred = model(src, tgt, model.get_tgt_mask(1))
    pred = F.log_softmax(pred.squeeze(), dim=-1)
    pred = torch.topk(pred, k=beam_width)
    tgts = pred.indices.reshape(beam_width, 1)
    probs = pred.values
    tgt = torch.cat([tgt.repeat(beam_width, 1), tgts], axis=1)

    padding = torch.empty_like(tgt[:, 0:1])
    padding.fill_(pad_token)
    
    src = src * beam_width
    for i in range(2, seq_len+1):
        pred = model(src, tgt, model.get_tgt_mask(i))
        
        probs_local = torch.repeat_interleave(probs, beam_width)
        tgt = torch.cat([tgt, padding], axis=1)
        tgts_local = torch.repeat_interleave(tgt, beam_width, 0)
        for j, seq in enumerate(pred):
            seq = seq[-1]
            seq = F.log_softmax(seq, dim=-1)
            preds = torch.topk(seq, k=beam_width)
            tgts_local[j*beam_width:((j+1)*beam_width), -1] = preds.indices
            probs_local[j*beam_width:((j+1)*beam_width)] += preds.values
            if tgt[j, -2] == end_token or tgt[j, -2] == pad_token:
                tgts_local[(j+1)*beam_width-1, -1] = pad_token
                probs_local[(j+1)*beam_width-1] -= preds.values[-1]

        # length penalty
        ends = torch.cat([(t == end_token).unsqueeze(0) for t in tgts_local[:, -1]], dim=0)
        pads = torch.cat([(t == pad_token).unsqueeze(0) for t in tgts_local[:, -1]], dim=0)
        probs_local[torch.logical_not(pads)] /= (5 + i)**alpha / (5 + 1)**alpha

        # after end filtering
        after_ends = torch.cat([(t == end_token).unsqueeze(0) for t in tgts_local[:, -2]], dim=0)
        after_pads = torch.cat([(t == pad_token).unsqueeze(0) for t in tgts_local[:, -2]], dim=0)
        after_mask = torch.logical_and(
            torch.logical_or(after_ends, after_pads), 
            torch.logical_not(pads)
        )
        probs_local.masked_fill_(after_mask, -1e20)

        # probs, indices = torch.topk(probs_local, k=beam_width)
        _, indices = torch.topk(probs_local, k=beam_width)

        # remove length penalty
        probs_local[torch.logical_not(torch.logical_or(pads, ends))] *= (5 + i)**alpha / (5 + 1)**alpha
        
        probs = probs_local[indices]
        tgt = tgts_local[indices, :]
        text = decode_tokens(tgt[0].detach().cpu().numpy())
        output_queue.put(((i/float(seq_len)), text))
        if tgt[0, -1] == end_token:
            break
    text = decode_tokens(tgt[0].detach().cpu().numpy())
    output_queue.put((1, text))
    return tgt


@torch.no_grad()
def beam_decode(src, seq_len, first_token, output_queue, end_token=END_TOK, pad_token=PAD_TOK, beam_width=5):
    tgt = first_token.unsqueeze(0).unsqueeze(0)
    alpha = 1.0
    
    pred = model.pred_nodes(src, tgt, model.get_tgt_mask(1)) # TODO
    pred = F.log_softmax(pred.squeeze(), dim=-1)
    pred = torch.topk(pred, k=beam_width)
    tgts = pred.indices.reshape(beam_width, 1)
    probs = pred.values
    tgt = torch.cat([tgt.repeat(beam_width, 1), tgts], axis=1)

    padding = torch.empty_like(tgt[:, 0:1])
    padding.fill_(pad_token)
    
    src = torch.repeat_interleave(src, beam_width, 0)
    for i in range(2, seq_len+1):
        pred = model.pred_nodes(src, tgt, model.get_tgt_mask(i)) # TODO
        
        probs_local = torch.repeat_interleave(probs, beam_width)
        tgt = torch.cat([tgt, padding], axis=1)
        tgts_local = torch.repeat_interleave(tgt, beam_width, 0)
        for j, seq in enumerate(pred):
            seq = seq[-1]
            seq = F.log_softmax(seq, dim=-1)
            preds = torch.topk(seq, k=beam_width)
            tgts_local[j*beam_width:((j+1)*beam_width), -1] = preds.indices
            probs_local[j*beam_width:((j+1)*beam_width)] += preds.values
            if tgt[j, -2] == end_token or tgt[j, -2] == pad_token:
                tgts_local[(j+1)*beam_width-1, -1] = pad_token
                probs_local[(j+1)*beam_width-1] -= preds.values[-1]

        # length penalty
        ends = torch.cat([(t == end_token).unsqueeze(0) for t in tgts_local[:, -1]], dim=0)
        pads = torch.cat([(t == pad_token).unsqueeze(0) for t in tgts_local[:, -1]], dim=0)
        probs_local[torch.logical_not(pads)] /= (5 + i)**alpha / (5 + 1)**alpha

        # after end filtering
        after_ends = torch.cat([(t == end_token).unsqueeze(0) for t in tgts_local[:, -2]], dim=0)
        after_pads = torch.cat([(t == pad_token).unsqueeze(0) for t in tgts_local[:, -2]], dim=0)
        after_mask = torch.logical_and(
            torch.logical_or(after_ends, after_pads), 
            torch.logical_not(pads)
        )
        probs_local.masked_fill_(after_mask, -1e20)

        # probs, indices = torch.topk(probs_local, k=beam_width)
        _, indices = torch.topk(probs_local, k=beam_width)

        # remove length penalty
        probs_local[torch.logical_not(torch.logical_or(pads, ends))] *= (5 + i)**alpha / (5 + 1)**alpha
        
        probs = probs_local[indices]
        tgt = tgts_local[indices, :]
        text = decode_tokens(tgt[0].detach().cpu().numpy())
        output_queue.put(((i/float(seq_len)), text))
        if tgt[0, -1] == end_token:
            break
    text = decode_tokens(tgt[0].detach().cpu().numpy())
    output_queue.put((1, text))
    return tgt


def process(text, queue):
    c_to_cpg(text)
    X, y, dot = load_set.load("cache/cpg_reconstruction/demo.cpg", use_cache=False, return_dot=True)
    queue.put((0, dot))
    
    X = [[x.cuda() for x in X]]
    y = y.cuda()
    sequence_length = y.size(0) - 1
    beam_width = 20
    beam_decode(X, sequence_length, y[0], beam_width=beam_width, output_queue=queue)


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


@torch.no_grad()
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


def pad(latent, num_nodes):
    padding_size = num_nodes - latent.shape[0]
    if padding_size > 0:
        latent = F.pad(latent, pad=(0, 0, 0, padding_size))
    elif padding_size < 0:
        latent = latent[:padding_size, :]
    return latent


@torch.no_grad()
def autoencode_interpolated(model, graph1, graph2, ratio, repetitions):
    num_nodes = int(ratio * graph1.num_nodes + (1 - ratio) * graph2.num_nodes)
    device = graph1.x.device
    dtype = graph1.x.dtype
    hidden_channels = graph1.x.shape[1]

    decoded_adj = torch.zeros((num_nodes, num_nodes), device=device, dtype=dtype)
    decoded_features = torch.zeros((num_nodes, hidden_channels), device=device, dtype=dtype)
    for i in range(repetitions):
        latent1 = model.encode(graph1)
        latent2 = model.encode(graph2)
        latent1 = latent1["mu"] + torch.randn_like(latent1["logstd"]) * torch.exp(latent1["logstd"])
        latent2 = latent2["mu"] + torch.randn_like(latent2["logstd"]) * torch.exp(latent2["logstd"])

        latent1 = pad(latent1, num_nodes)
        latent2 = pad(latent2, num_nodes)

        latent = ratio * latent1 + (1 - ratio) * latent2

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


@torch.no_grad()
def process_ae(text, queue):
    c_to_cpg(text)
    _X, y, dot = load_set.load("cache/cpg_reconstruction/demo.cpg", use_cache=False, return_dot=True)
    queue.put((0, dot))
    
    G = load_torch("cache/cpg_reconstruction/demo.cpg", encode_set).cuda()
    G = autoencode(ae_model, G, 10)

    # X = [[x.cuda() for x in X]]
    y = y.cuda()
    sequence_length = y.size(0) - 1
    beam_width = 20
    beam_decode(G.x[..., -128:].unsqueeze(0), sequence_length, y[0], beam_width=beam_width, output_queue=queue)


@torch.no_grad()
def process_ae(text1, text2, queue):
    c_to_cpg(text1)
    _X, y1, _dot = load_set.load("cache/cpg_reconstruction/demo.cpg", use_cache=False, return_dot=True)
    G1 = load_torch("cache/cpg_reconstruction/demo.cpg", encode_set).cuda()

    c_to_cpg(text2)
    _X, y2, _dot = load_set.load("cache/cpg_reconstruction/demo.cpg", use_cache=False, return_dot=True)
    G2 = load_torch("cache/cpg_reconstruction/demo.cpg", encode_set).cuda()
    
    G = autoencode_interpolated(ae_model, G1, G2, 0.5, 10)

    # X = [[x.cuda() for x in X]]
    y1 = y1.cuda()
    y2 = y2.cuda()
    sequence_length = max(y1.size(0), y2.size(0), 256) - 1
    beam_width = 20
    beam_decode(G.x[..., -128:].unsqueeze(0), sequence_length, y1[0], beam_width=beam_width, output_queue=queue)