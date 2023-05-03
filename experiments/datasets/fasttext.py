import glob
import os
import re

import numpy as np
import networkx as nx
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.fasttext import FastText
from tqdm import tqdm


class FastTextBuilder(object):
    def __init__(self, params, overwrite_cache=False):
        self.params = params
        self.vector_size = params["vector_size"]
        self.window = params["window"]
        self.overwrite_cache = overwrite_cache
        self.cache_dir = os.path.join("cache", params["cache_dir"])
        self.model = None

        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
    
    def get_corpus(self):
        corpus = []

        tokenizer = ParseTokenizer()
        for directory, _modern in self.params["training_files"]:
            c_files = list(glob.glob(os.path.join(directory, "**", "*.c")))
            if len(c_files) > 0:
                print(f"Loading C files for {directory}")
                for path in tqdm(c_files):
                    with open(path, "r", encoding='utf-8', errors='ignore') as f:
                        tokenizer = ParseTokenizer()
                        corpus.append(tokenizer(f.read()))
            else:
                cpg_files = list(glob.glob(os.path.join(directory, "**", "*.cpg")))
                assert len(cpg_files) > 0, "Must find either cpg files or c files for code"

                print(f"Loading cpg files for {directory}")
                for path in tqdm(cpg_files):
                    codelines = list()
                    with open(path, "r", encoding='utf-8', errors='ignore') as f:
                        graph = nx.Graph(nx.drawing.nx_pydot.read_dot(f))

                        for node_id in graph:
                            node = graph.nodes[node_id]
                            if node.get("enclosing") is None:
                                print("No enclosing found for ", node, graph, path)
                                continue
                            code = node["enclosing"]
                            if code[0] == "\"" and code[-1] == "\"":
                                code = code[1:-1]
                            if node.get("label") == "TranslationUnitDeclaration":
                                codelines = code
                                break
                            codelines.append(code)

                    corpus.append(tokenizer("\n".join(codelines)))

        return corpus

    def build_model(self):
        model_path = os.path.join(self.cache_dir, "fasttext.model")
        if os.path.isfile(model_path) and not self.overwrite_cache:
            print("Loading fasttext model.")
            self.model = FastText.load(model_path)
        else:
            corpus = self.get_corpus()
            self.model = FastText(window=self.window, min_count=1, alpha=0.01, min_alpha=0.0000001, sample=1e-5,
                                workers=8, sg=1, hs=0, negative=5, vector_size=self.vector_size)
            self.model.build_vocab(corpus)
            self.model.train(corpus, total_examples=len(corpus), epochs=400,
                           compute_loss=True, callbacks=[CbPrintLoss()])
            print("Saving fasttext model.")
            self.model.save(model_path)

    def needs_corpus(self):
        return False

    def get_embedding(self, code):
        if len(code.strip()) == 0:
            return np.zeros((self.vector_size,))
        tokenizer = ParseTokenizer()
        tokenized = tokenizer(code)
        if len(tokenized) == 0:
            return np.zeros((self.vector_size,))
        return np.mean(np.array(
            [self.model.wv[x] if len(x) > 0 else np.zeros((self.vector_size,)) for x in
             tokenized]),
            axis=0
        )


operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--', '**',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '&',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':',
    '{', '}', '!', '~'
}


def to_regex(lst):
    return r'|'.join([f"({re.escape(el)})" for el in lst])


regex_split_operators = to_regex(operators3) + to_regex(operators2) + to_regex(operators1)


class ParseTokenizer(object):

    def __init__(self):
        pass

    def __call__(self, code):
        if len(code) == 0:
            return []
        
        if code[0] == code[-1] and (code[0] == "'" or code[0] == '"'):
            code = code[1:-1]

        tokenized = []

        for line in code.splitlines():
            line = line.strip()
            if line == '':
                continue

            # Mix split (characters and words)
            splitter = r' +|' + regex_split_operators + r'|(\/)|(\;)|(\-)|(\*)'
            line = re.split(splitter, line)

            # Remove None type
            line = list(filter(None, line))
            line = list(filter(str.strip, line))

            tokenized.extend(line)

        return tokenized


class CbPrintLoss(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0
        self.prev_loss = 0

    def on_epoch_end(self, model):
        curr_loss = model.get_latest_training_loss()
        # gensim stores only the total loss (for whichever reason),
        # so we have to track the previous loss value as well
        print('Loss after epoch {}: {}'.format(self.epoch, curr_loss - self.prev_loss))
        self.epoch += 1
        self.prev_loss = curr_loss
