from experiments.datasets.vulnerability_dataset import VulnerabilityDataset
from experiments.datasets.downsampled import DownsampledDataset
from experiments.datasets.combined import CombinedDataset
from experiments.datasets.smote import SMOTEDataset, AutoencodedDataset
from experiments.datasets.fillup import FillupDataset, NodeDropDataset, EdgeDropDataset

from experiments.autoencoding.encoders import GCNEncoder, GATEncoder, NodeVAEEncoder, GDNEncoder
from experiments.autoencoding.encoders import GGNNEncoder, GINEncoder
from experiments.autoencoding.regae import ReGAEEncoder, ReGAEDecoder
from experiments.autoencoding.encoder import CompositeGraphLevelEncoder
from experiments.autoencoding.decoder import CompositeDecoder
from experiments.autoencoding.decoders import InnerProductDecoder, NodeVAEDecoder, DirectedInnerProductDecoder
from experiments.autoencoding.decoders import GATDecoder, GDNDecoder, MLPAdjDecoder
from experiments.autoencoding.poolings import MeanPooling, MaxPooling, SumPooling
from experiments.autoencoding.classifier import GraphClassifier
from experiments.autoencoding.devign import DevignClassifier
from experiments.autoencoding.classifiers import MLPClassifier, IdentityClassifier
from experiments.autoencoding.autoencoder import AutoEncoder

def get_dataset(params):
    if params.get("type") == "SMOTE":
        params["dataset"]["overwrite_cache"] = params["overwrite_cache"]
        dataset = get_dataset(params["dataset"])

        params["model"]["encoder"]["features"] = dataset.get_input_size()
        params["model"]["encoder"]["edge_dim"] = dataset.get_edge_size()
        model = get_ae_model(params["model"]["encoder"], params["model"]["decoder"])
        model.load(params["model"]["checkpoint"])

        return SMOTEDataset(model, dataset, params)
    if params.get("type") == "Fillup":
        params["dataset"]["overwrite_cache"] = params["overwrite_cache"]
        dataset = get_dataset(params["dataset"])

        params["fill_from"]["overwrite_cache"] = params["overwrite_cache"]
        fill_from = get_dataset(params["fill_from"])

        return FillupDataset(dataset, fill_from, params)
    if params.get("type") == "NodeDrop":
        params["dataset"]["overwrite_cache"] = params["overwrite_cache"]
        dataset = get_dataset(params["dataset"])

        return NodeDropDataset(dataset, params)
    if params.get("type") == "EdgeDrop":
        params["dataset"]["overwrite_cache"] = params["overwrite_cache"]
        dataset = get_dataset(params["dataset"])

        return EdgeDropDataset(dataset, params)
    if params.get("type") == "Autoencoded":
        params["dataset"]["overwrite_cache"] = params["overwrite_cache"]
        dataset = get_dataset(params["dataset"])

        params["model"]["encoder"]["features"] = dataset.get_input_size()
        params["model"]["encoder"]["edge_dim"] = dataset.get_edge_size()
        model = get_ae_model(params["model"]["encoder"], params["model"]["decoder"])
        model.load(params["model"]["checkpoint"])

        return AutoencodedDataset(model, dataset, params)
    if params.get("type") == "Downsampled":
        params["dataset"]["overwrite_cache"] = params["overwrite_cache"]
        return DownsampledDataset(get_dataset(params["dataset"]))
    if params.get("type") == "Combined":
        datasets = []
        for i, _ in enumerate(params["datasets"]):
            params["datasets"][i]["overwrite_cache"] = params["overwrite_cache"]
            datasets.append(get_dataset(params["datasets"][i]))
        return CombinedDataset(datasets)
    return VulnerabilityDataset(**params)


def get_ae_model(encoder_params, decoder_params):
    encoder = get_encoder(encoder_params)
    decoder_params["hidden_channels"] = encoder_params["hidden_channels"]
    decoder_params["features"] = encoder_params["features"]
    decoder_params["edge_dim"] = encoder_params["edge_dim"]
    decoder = get_decoder(decoder_params)
    
    return AutoEncoder(encoder, decoder)


def get_classification_model(params):
    if params.get("variant") == "devign":
        return DevignClassifier(params)
    params["encoder"]["features"] = params["features"]
    params["encoder"]["edge_dim"] = params["edge_dim"]
    encoder = get_encoder(params["encoder"])
    
    params["classifier"]["features"] = params["encoder"]["hidden_channels"]
    params["classifier"]["classes"] = params["classes"]
    classifier = get_classifier(params["classifier"])

    return GraphClassifier(encoder, classifier)


def get_decoder(params):
    if params["type"] == "VAEDecoder":
        params["decoder"]["hidden_channels"] = params["hidden_channels"]
        params["decoder"]["features"] = params["features"]
        return NodeVAEDecoder(get_decoder(params["decoder"]))
    if params.get("type") == "ReGAE":
        return ReGAEDecoder(params)
    if params["type"] == "InnerProductDecoder":
        return InnerProductDecoder()
    if params["type"] == "DirectedInnerProductDecoder":
        return DirectedInnerProductDecoder(params)
    if params["type"] == "MLPAdj":
        return MLPAdjDecoder(params)
    if params["type"] == "CompositeDecoder":
        split = min(params["hidden_channels"] // 2, 128)
        
        params["adj"]["hidden_channels"] = split
        params["adj"]["features"] = params["features"]
        adj_decoder = get_decoder(params["adj"])

        params["feature"]["hidden_channels"] = params["hidden_channels"] - split
        params["feature"]["features"] = params["features"]
        feature_decoder = get_decoder(params["feature"])

        return CompositeDecoder(adj_decoder, feature_decoder)
    if params["type"] == "GNNDecoder":
        if params["layer_type"] == "GAT":
            return GATDecoder(params)
        if params["layer_type"] == "GDN":
            return GDNDecoder(params)
    
    raise ValueError(f"Could not construct decoder for {params}")


def get_encoder(params):
    if params.get("type") == "GraphComposite":
        pooling = get_pooling(params["pooling"])
        params["encoder"]["features"] = params["features"]
        params["encoder"]["edge_dim"] = params["edge_dim"]
        params["hidden_channels"] = params["encoder"]["hidden_channels"]
        encoder = get_encoder(params["encoder"])

        return CompositeGraphLevelEncoder(encoder, pooling)
    if params.get("type") == "ReGAE":
        return ReGAEEncoder(params)
    if params.get("autoencoder_type") == "VAE":
        params["hidden_channels"] = params["mu"]["hidden_channels"]
        params["common"]["features"] = params["features"]
        params["common"]["edge_dim"] = params["edge_dim"]
        common_encoder = get_encoder(params["common"])
        
        params["mu"]["features"] = params["common"]["hidden_channels"]
        params["logstd"]["features"] = params["common"]["hidden_channels"]
        params["mu"]["edge_dim"] = params["common"]["edge_dim"]
        params["logstd"]["edge_dim"] = params["common"]["edge_dim"]
        
        mu_encoder = get_encoder(params["mu"])
        logstd_encoder = get_encoder(params["logstd"])

        return NodeVAEEncoder(common_encoder, mu_encoder, logstd_encoder)
    if params["layer_type"] == "GCN":
        return GCNEncoder(**params)
    if params["layer_type"] == "GAT":
        return GATEncoder(**params)
    if params["layer_type"] == "GDN":
        return GDNEncoder(**params)
    if params["layer_type"] == "GIN":
        return GINEncoder(**params)
    if params["layer_type"] == "GGNN":
        return GGNNEncoder(**params)
    
    raise ValueError(f"Could not construct encoder for {params}")


def get_pooling(params):
    if params.get("type") == "mean":
        return MeanPooling(params)
    if params.get("type") == "max":
        return MaxPooling(params)
    if params.get("type") == "sum":
        return SumPooling(params)
    
    raise ValueError(f"Could not construct pooling for {params}")


def get_classifier(params):
    if params.get("layer_type") == "MLP":
        return MLPClassifier(params)
    if params.get("layer_type") == "identity":
        return IdentityClassifier(params)
    
    raise ValueError(f"Could not construct classifier for {params}")