import math

import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_LEN = 4096
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        if len(tokens) == 0:
            return F.normalize(
                torch.ones(
                    (1,self.emb_size),
                    dtype=self.embedding.weight.dtype,
                    device=self.embedding.weight.device
                ),
                p=2, dim=-1
            )
        emb = self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        return F.normalize(emb, p=2, dim=-1)

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
            F.normalize(torch.mean(self.node_encoder(self.positional_encoder(self.embedding(node).unsqueeze(0))), dim=1), p=2, dim=-1)
            for node in sample
            ], dim=0)
    
    def forward(self, src, tgt, tgt_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
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
            # else:
            #     assert len(sample) == max_src_length
            masks.append(mask.unsqueeze(0))
            srcs.append(sample.unsqueeze(0))
        src = torch.cat(srcs, dim=0)
        src_pad_mask = torch.cat(masks, dim=0) == 1
        tgt = self.embedding(tgt)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
        # !! using batch_first=True !!
        # Transformer blocks - Out size = (batch_size, sequence length, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        
        return out
    
    def pred_nodes(self, src, tgt, tgt_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        tgt_pad_mask = tgt == self.PAD_TOK
        tgt = self.embedding(tgt)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
        # !! using batch_first=True !!
        # Transformer blocks - Out size = (batch_size, sequence length, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        
        return out
      
    def get_tgt_mask(self, size, cached=True):
        if cached:
            return self.tgt_mask[:size, :size]
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size, device=device) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask