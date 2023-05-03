INNER_PRODUCT = {
    "type": "InnerProductDecoder",
    "name": "INNER"
}

VAE_INNER = {
    "type": "VAEDecoder",
    "decoder": INNER_PRODUCT,
    "name": "VAE_INNER"
}

DIRECTED_INNER = {
    "type": "DirectedInnerProductDecoder",
    "name": "DIRECTED_INNER",
    "n_layers": 1
}

ReGAE_DECODER = {
    "type": "ReGAE",
    "patch_dim": 8,
    "name": "ReGAE_DECODER"
}

VAE_DIRECTED_INNER = {
    "type": "VAEDecoder",
    "decoder": DIRECTED_INNER,
    "name": "VAE_DIRECTED_INNER"
}

COMPOSITE = {
    "type": "CompositeDecoder",
    "adj": DIRECTED_INNER,
    "feature": {
        "type": "GNNDecoder",
        "layer_type": "GAT",
        "num_layers": 2,
        "dropout": 0,
    }
}

COMPOSITE_VAE = {
    "type": "VAEDecoder",
    "decoder": COMPOSITE,
    "name": "COMPOSITE_VAE"
}

COMPOSITE_GDN = {
    "type": "CompositeDecoder",
    "adj": DIRECTED_INNER,
    "feature": {
        "type": "GNNDecoder",
        "layer_type": "GDN"
    }
}

COMPOSITE_GDN_VAE = {
    "type": "VAEDecoder",
    "decoder": COMPOSITE_GDN,
    "name": "COMPOSITE_GDN_VAE"
}

MLP_ADJ = {
    "type": "MLPAdj",
    "name": "MLPAdj",
    "n_layers": 2
}

VAE_MLP_ADJ = {
    "type": "VAEDecoder",
    "decoder": MLP_ADJ,
    "name": "VAE_MLP_ADJ"
}

COMPOSITE_GDN_MLP = {
    "type": "CompositeDecoder",
    "adj": MLP_ADJ,
    "feature": {
        "type": "GNNDecoder",
        "layer_type": "GDN"
    }
}

COMPOSITE_GDN_MLP_VAE = {
    "type": "VAEDecoder",
    "decoder": COMPOSITE_GDN_MLP,
    "name": "COMPOSITE_GDN_MLP_VAE"
}