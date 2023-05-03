import os
import re
import glob
import gzip
import pickle
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
import torch_geometric

from sklearn.preprocessing import OneHotEncoder
from torch.nn import functional as F


def filter_length(files, length, tokenizer):
    temp_set = CPGReconstructionDataset(files, tokenizer)

    cache_path = os.path.join("cache", "cpg_reconstruction", "lengths.pkl.gz")
    lengths = {}
    if os.path.isfile(cache_path):
        with gzip.open(cache_path, "r") as f:
            lengths = pickle.load(f)
    for i, file in tqdm(enumerate(files), total=len(files)):
        if file not in lengths:
            X, y = temp_set[i]
            lengths[file] = max(len(y), len(X))
    with gzip.open(cache_path, "w") as f:
        pickle.dump(lengths, f)
    return [key for (key, value) in lengths.items() if value <= length and key in files]

class CPGReconstructionDataset(torch.utils.data.Dataset):
    def __init__(self, files, tokenizer, ae=False) -> None:
        super().__init__()

        # has a memory leak when using dataloader with multiprocessing
        # see https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        self.files = files
        self.tokenizer = tokenizer

        self.START_TOK = torch.tensor([self.tokenizer.token_to_id("<|startoftext|>")]).long()
        self.END_TOK = torch.tensor([self.tokenizer.token_to_id("<|endoftext|>")]).long()
        self.PAD_TOK = torch.tensor(self.tokenizer.token_to_id("<|pad|>")).long()

        self.ae = ae

        if self.ae:
            print(f"Using ae, all files are {len(self.files)}")
            def ae_exists(p):
                h = p.split("/")[-1].replace(".cpg","")
                return os.path.exists(f"cache/cpg_reconstruction/ae/{h}.gz")
            self.files = list(filter(ae_exists, self.files))
            print(f"After filtering {len(self.files)}")
    
    def __len__(self):
        return len(self.files)
    
    def load(self, p, use_cache=True, return_dot=False):
        h = p.split("/")[-1].replace(".cpg","")
        if self.ae:
            if not os.path.exists(f"cache/cpg_reconstruction/ae/{h}.gz"):
                raise ValueError(f"Autoencoded graphs must be built by build_ae first")
            with gzip.open(f"cache/cpg_reconstruction/ae/{h}.gz", "rb") as f:
                G = pickle.load(f)
                f = open("cache/cpg_reconstruction/{}".format(h),'rb')
                X, y = pickle.load(f)
                f.close()
                return torch.as_tensor(G), y.squeeze()
        if os.path.exists("cache/cpg_reconstruction/{}".format(h)) and use_cache:
            f = open("cache/cpg_reconstruction/{}".format(h),'rb')
            X, y = pickle.load(f)
            f.close()
        else:
            sourcefile = p.replace(".cpg","").replace("/cpg/","/c_files/")
            if sourcefile[-2:] != ".c":
                sourcefile += ".c"
            sourcetext = self.read_cfile(sourcefile)

            g, dot = self.load_dot(p, sourcetext)

            X = self.sort_tokens(g)
            
            source = self.tokens_by_c(sourcetext)
            y = source.unsqueeze(0)
            
            if use_cache:
                f = open("cache/cpg_reconstruction/{}".format(h),"wb")
                pickle.dump((X,y), f)
                f.close()
        if return_dot:
            assert not use_cache
            return X, y.squeeze(), dot
        return X, y.squeeze()

    @torch.no_grad()
    def __getitem__(self, index):
        p = self.files[index]
        return self.load(p)
    
    def load_dot(self, p, sourcetext=None):
        with open(p, "r") as f:
            modern = bool(".c.cpg" != p[-len(".c.cpg"):])
            dot = read_dot(f, modern)
            dot.graph["label"] = 0.5  # label unknown

            data = self.encode(dot, sourcetext)

            data.x = data.codeenc
            
            return data, dot

    def encode(self, graph, sourcetext):
        for node_id in graph:
            node = graph.nodes[node_id]
            if "enclosing" in node:
                node["ast"] = node["label"]
                node["lines"] = node["location"]
                node["code"] = node["enclosing"]
            else:
                paramsplit = node["label"].split(",")
                ast = paramsplit[0]
                # lines = "".join(paramsplit[2:])
                lines = paramsplit[-1]
                node["ast"] = ast
                node["lines"] = lines
                node["code"] = " "

                if re.match(r"^\s*\d+:\d+\s*\d+:\d+\s*$", lines) is not None:
                    lines, cols = lines.strip().split(" ")
                    start_line, end_line = lines.split(":")
                    start_line, end_line = int(start_line), int(end_line)
                    start_line -= 1  # offset for starting with 1
                    start_col, end_col = cols.split(":")
                    start_col, end_col = int(start_col), int(end_col)
                    start_col -= 1  # offset for starting with 1
                    num_lines = end_line - start_line
                    for index, line in enumerate(sourcetext[start_line:end_line]):
                        if index == 0 and index == num_lines - 1:
                            node["code"] = sourcetext[start_line][start_col:end_col]
                        elif index == 0:
                            node["code"] = sourcetext[start_line][start_col:]
                        elif index == num_lines - 1:
                            node["code"] += "\n" + sourcetext[start_line + index][:end_col]
                        else:
                            node["code"] += "\n" + sourcetext[start_line + index]

        for node_id in graph:
            indices = self.tokenizer.encode(graph.nodes[node_id]["code"]).ids
            graph.nodes[node_id]["codeenc"] = indices
        torch_graph = from_networkx_multi(graph)
        torch_graph.y = graph.graph["label"]
        return torch_graph

    def sort_tokens(self, g):
        result = []
        for n in range(len(g.ast)):
            result.append(torch.as_tensor(g.codeenc[n], dtype=torch.long))
        return result

    def read_cfile(self, cfile):
        with open(cfile, "r") as f:
            return f.read()

    def tokens_by_c(self, c):
        tokens = self.tokenizer.encode(c).ids
        tokens = torch.as_tensor(tokens, dtype=torch.long)
        return torch.cat([self.START_TOK, tokens, self.END_TOK], dim=0)
    
    def pad(self, sequence, length):
        if len(sequence) == length:
            return sequence
        # assert len(sequence) < length
        return F.pad(sequence, (0, length - len(sequence)), value=self.PAD_TOK)

    def collate_fn(self, samples):
        sequences = []
        graphs = []
        max_length = max(len(y) for (X, y) in samples)
        for (X, y) in samples:
            graphs.append(X)
            sequences.append(self.pad(y, max_length).unsqueeze(0))
        return graphs, torch.cat(sequences, dim=0)


class ASTEncoder(object):
    def __init__(self, params, overwrite_cache=False):
        self.params = params
        self.overwrite_cache = overwrite_cache
        self.cache_dir = params["cache_dir"]
        self.ast_dict = None

        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

    def _get_cache_path(self):
        return os.path.join(self.cache_dir, "asttypes.pkl.gz")

    def _clean(self, s):
        if len(s) > 0 and s[0] == s[-1] and s[0] in ["'", '"']:
            return s[1:-1].strip()
        return s.strip()

    def build_astdict(self):
        asttypes = list(self.get_asttypes().keys())
        asttypes.append("UNKNOWN")
        self.astenc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.astenc.fit(np.array(asttypes).reshape(-1, 1))
        self.ast_dict = {asttype: np.squeeze(self.astenc.transform(np.array(asttype).reshape(1, -1)))
                         for asttype in asttypes}

    def get_asttypes(self):
        if os.path.isfile(self._get_cache_path()) and not self.overwrite_cache:
            with gzip.open(self._get_cache_path(), "r") as f:
                return pickle.load(f)
        asttypes = defaultdict(int)
        for directory, modern in self.params["training_files"]:
            cpg_files = list(glob.glob(os.path.join(directory, "**", "*.cpg")))
            assert len(cpg_files) > 0, "Must find cpg files for AST labels"

            print(f"Loading cpg files for {directory}")
            for path in tqdm(cpg_files[:10]):
                print(path)
                with open(path, "r", encoding='utf-8', errors='ignore') as f:
                    graph = read_dot(f, modern)

                    for node_id in graph:
                        node = graph.nodes[node_id]
                        if node.get("label") is None:
                            asttypes["UNKNOWN"] += 1
                            continue
                        if modern:
                            asttypes[self._clean(node["label"])] += 1
                        else:
                            information = ",".join(node["label"].split(",")[:-1])
                            paramsplit = information.split(",")
                            ast = paramsplit[0].strip()
                            asttypes[self._clean(ast)] += 1

        with gzip.open(self._get_cache_path(), "w") as f:
            pickle.dump(asttypes, f)

        return asttypes


def load(p, astencoder, ast_dict, w2v, return_dot=False):
    with open(p, "r") as f:
        dot = read_dot(f, True)
        dot.graph["label"] = 0.5  # label unknown

        data = encode(dot, astencoder, ast_dict, w2v)

        xs = [data.astenc, data.codeenc]
        data.x = torch.cat(xs, dim=-1).float()

        if return_dot:
            return data, dot
        return data


def encode(graph, astencoder, ast_dict, w2v):
    for node_id in graph:
        node = graph.nodes[node_id]
        node["ast"] = node["label"]
        node["lines"] = node["location"]
        node["code"] = node["enclosing"]

    for node in graph:
        asttype = astencoder._clean(graph.nodes[node]["ast"])
        if not asttype in ast_dict:
            asttype = "UNKNOWN"
        graph.nodes[node]["astenc"] = ast_dict[asttype]
        graph.nodes[node]["codeenc"] = w2v.get_embedding(graph.nodes[node]["code"])
    try:
        torch_graph = from_networkx_multi(graph)
        torch_graph.y = graph.graph["label"]
        return torch_graph
    except Exception as e:
        print(f"failed with {repr(e)}")


def read_dot(f, modern):
    if modern:
        graph = nx.Graph(nx.drawing.nx_pydot.read_dot(f))
        graph.remove_nodes_from(list(nx.isolates(graph)))
        edges = []
        for (u, v, label) in graph.edges(data=True):
            edges.append((u, v))
        return graph
    else:
        content = f.read()
        if content == "":
            print(f" is empty")
        for num, line in enumerate(content.split("\n")):
            if "digraph G {" in line:
                content = "\n".join(content.replace("yodigraph", "digraph").split("\n")[num:])
                break
        return _forgiving_dot_parser(content)


def _forgiving_dot_parser(content):
    graph = nx.Graph()
    lines = content.split("\n")
    for line in lines:
        if len(line) <= 0:
            continue
        if line[0] == "\"":
            name = line.split(" ")[0].replace("\"", "")
            label = "".join(line.split(" [label = \"(")[1])[:-3]
            graph.add_node(name, label=label)
        elif line[0] == " ":
            splitted = line.split(" -> ")
            lnode = splitted[0].replace("\"", "")
            rnode = splitted[1].replace("\"", "").split(" ")[0]
            graph.add_edge(lnode.strip(), rnode.strip())
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph


def from_networkx_multi(G):
    # original code: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/convert.html#from_networkx

    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()
    if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        edge_index = edge_index[0:2, :]

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            # print(key)
            # print(item)
            if type(item) is list and len(item) > 0 and type(item[0]) is np.ndarray:
                item = np.stack(item)
            data[key] = torch.as_tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data
