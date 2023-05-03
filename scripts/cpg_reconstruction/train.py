import os
import sys
import math

import torch
import tokenizers
import numpy as np

from tqdm import tqdm

from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataloader import CPGReconstructionDataset, filter_length
from model import Transformer, MAX_LEN

from params import REVEAL_DATASET_PARAMS, PATCHDB_PARAMS
from utils import get_dataset


AUTOENCODED = True

NUM_BATCHES = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REVEAL_DATASET_PARAMS["overwrite_cache"] = False
reveal = get_dataset(REVEAL_DATASET_PARAMS)
PATCHDB_PARAMS["overwrite_cache"] = False
patchdb = get_dataset(PATCHDB_PARAMS)

tokenizer = tokenizers.Tokenizer.from_pretrained("Salesforce/codegen-350M-multi")
tokenizer.add_special_tokens(["<|startoftext|>", "<|pad|>"])

PAD_TOK = tokenizer.token_to_id("<|pad|>")

model = Transformer(num_tokens=tokenizer.get_vocab_size(),
        dim_model=128,
        num_heads=2,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout_p=0.1,
        padding_token=PAD_TOK).to(device)
model.load_state_dict(torch.load(os.path.join("cache", "cpg_reconstruction", "models", "model_9_.chkpt")))

# files = reveal.get_graph_identifiers() + patchdb.get_graph_identifiers()
files = patchdb.get_graph_identifiers()
del reveal
del patchdb
files = filter_length(files, MAX_LEN, tokenizer)
np.random.seed(0)
np.random.shuffle(files)
train_files = files[:len(files)-1000]
train_set = CPGReconstructionDataset(train_files, tokenizer, ae=AUTOENCODED)
train_loader = DataLoader(train_set, batch_size=1, collate_fn=train_set.collate_fn, pin_memory=True,
                            shuffle=True, num_workers=0)
test_files = files[-1000:]
test_set = CPGReconstructionDataset(test_files, tokenizer, ae=AUTOENCODED)
test_loader = DataLoader(test_set, batch_size=1, collate_fn=test_set.collate_fn, pin_memory=True,
                            shuffle=False, num_workers=0)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0005, total_steps=30*len(train_files)//NUM_BATCHES)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOK)

for epoch in range(1000):
    total_loss = 0
    count = 0
    testpbar = tqdm(test_loader, mininterval=1)
    model.eval()
    with torch.no_grad():
        for (X, y) in testpbar:
            if AUTOENCODED:
                X = X[0].to(device, non_blocking=True)
            else:
                X = [[node.to(device, non_blocking=True) for node in sample] for sample in X]
            y = y.to(device, non_blocking=True)
            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:,:-1]
            y_expected = y[:,1:]

            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            if sequence_length > MAX_LEN:
                print("skipped sequence with length {}".format(sequence_length))
                continue
            if max(len(sample) for sample in X) > MAX_LEN:
                print(f"skipping graph with {len(X)} nodes")
                continue
            tgt_mask = model.get_tgt_mask(sequence_length)

            # Standard training except we pass in y_input and tgt_mask
            if AUTOENCODED:
                pred = model.pred_nodes(X.unsqueeze(0), y_input, tgt_mask)
            else:
                pred = model(X, y_input, tgt_mask)
            pred = pred.view(-1, tokenizer.get_vocab_size())
            y_expected = y_expected.reshape(-1)
            loss = loss_fn(pred, y_expected)

            count += 1
            loss = loss.detach().cpu().item()
            if math.isnan(loss):
                print("x", X)
                print("x", X, "y", y)
                sys.exit(1)
            total_loss += loss
            testpbar.set_description(f"testing loss {total_loss/count:.3f}", refresh=False)
    print("total test loss {}".format(total_loss/count))
    model.train()  
    total_loss = 0
    count = 0
    pbar = tqdm(train_loader, mininterval=1)

    batch_index = 0
    optimizer.zero_grad(set_to_none=True)
    for (X, y) in pbar:
        if AUTOENCODED:
            X = X[0].to(device, non_blocking=True)
        else:
            X = [[node.to(device, non_blocking=True) for node in sample] for sample in X]
        y = y.to(device, non_blocking=True)
        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y[:,:-1]
        y_expected = y[:,1:]

        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        if sequence_length > MAX_LEN:
            print("skipped sequence with length {}".format(sequence_length))
            continue
        if max(len(sample) for sample in X) > MAX_LEN:
            print(f"skipping graph with {len(X)} nodes")
            continue
        tgt_mask = model.get_tgt_mask(sequence_length)

        # Standard training except we pass in y_input and tgt_mask
        if AUTOENCODED:
            pred = model.pred_nodes(X.unsqueeze(0), y_input, tgt_mask)
        else:
            pred = model(X, y_input, tgt_mask)
        pred = pred.view(-1, tokenizer.get_vocab_size())
        y_expected = y_expected.reshape(-1)
        loss = loss_fn(pred, y_expected) / NUM_BATCHES

        loss.backward()

        batch_index += 1
        if batch_index >= NUM_BATCHES - 1:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            batch_index = 0
            scheduler.step()
    
        count += 1
        loss = loss.detach().cpu().item()
        if math.isnan(loss):
                print("x", X)
                print("x", X, "y", y)
                sys.exit(1)
        total_loss += loss * NUM_BATCHES
        pbar.set_description(f"training loss {total_loss/count:.3f}", refresh=False)

    print("total epoch loss {}".format(total_loss/len(train_files)))
    torch.save(model.state_dict(), "./cache/cpg_reconstruction/models/model_{}.chkpt".format(epoch))