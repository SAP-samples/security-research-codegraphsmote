import os
import math

import torch
import tokenizers

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


MAX_LEN = 4096
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerBuilder(object):
    def __init__(self, params, overwrite_cache=False):
        self.params = params
        self.tokenizer = tokenizers.Tokenizer.from_pretrained("Salesforce/codegen-350M-multi")
        self.tokenizer.add_special_tokens(["<|startoftext|>", "<|pad|>"])

        self.PAD_TOK = self.tokenizer.token_to_id("<|pad|>")
        self.vector_size = self.params["vector_size"]
        assert self.params["vector_size"] == 128
        self.model = Transformer(num_tokens=self.tokenizer.get_vocab_size(),
            dim_model=128,
            num_heads=2,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dropout_p=0.1,
            padding_token=self.PAD_TOK).cuda()
        self.model.load_state_dict(torch.load(self.params["model_file"]))
        self.model.eval()

    def build_model(self):
        pass

    def needs_corpus(self):
        return False
    
    def _tokenize(self, text):
        return self.tokenizer.encode(text).ids

    @torch.no_grad()
    def get_embedding(self, code):
        if len(code.strip()) == 0:
            return np.zeros((self.vector_size,))
        tokenized = self._tokenize(code)
        if len(tokenized) == 0:
            return np.zeros((self.vector_size,))
        tokenized = torch.as_tensor(tokenized, dtype=torch.long, device=device)
        # embed -> detach from computation graph -> remove batch wrapper -> move to cpu -> numpy
        return self.model.embed_nodes([tokenized]).detach()[0].cpu().numpy()


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding):
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(1), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return F.normalize(self.embedding(tokens.long()) * math.sqrt(self.emb_size), p=2, dim=-1)

class Transformer(nn.Module):
    """
    Model adapted from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    # Constructor
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        padding_token
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=MAX_LEN
        )
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            norm_first=True,
            batch_first=True
        )
        self.node_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model,
                nhead=num_heads,
                batch_first=True,
                norm_first=True,
                dim_feedforward=512,
                dropout=dropout_p
            ),
            num_layers=1
        )
        self.embedding = TokenEmbedding(num_tokens, dim_model)
        self.out = nn.Linear(dim_model, num_tokens)

        self.tgt_mask = self.get_tgt_mask(MAX_LEN, cached=False)
        self.PAD_TOK = torch.tensor(padding_token, requires_grad=False, device=device).long()

    def embed_nodes(self, sample):
        return torch.cat([
            F.normalize(torch.mean(self.node_encoder(self.positional_encoder(self.embedding(node).unsqueeze(0))), dim=1), dim=-1, p=2)
            for node in sample
            ], dim=0)
    
    def forward(self, src, tgt, tgt_mask=None):
        tgt_pad_mask = tgt == self.PAD_TOK
        src = [self.embed_nodes(sample) for sample in src]
        srcs = []
        masks = []
        max_src_length = max(len(sample) for sample in src)
        for sample in src:
            mask = torch.zeros(max_src_length, device=sample.device, dtype=torch.uint8)
            if len(sample) < max_src_length:
                mask[len(sample):] = 1
                sample = F.pad(sample, (0, 0, 0, max_src_length - len(sample)), value=self.PAD_TOK)
            masks.append(mask.unsqueeze(0))
            srcs.append(sample.unsqueeze(0))
        src = torch.cat(srcs, dim=0)
        src_pad_mask = torch.cat(masks, dim=0) == 1
        tgt = self.embedding(tgt)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        
        return out
      
    def get_tgt_mask(self, size, cached=True):
        if cached:
            return self.tgt_mask[:size, :size]
        # Generates a square matrix where the each row allows one more word to be seen
        mask = torch.tril(torch.ones(size, size, device=device) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        return mask