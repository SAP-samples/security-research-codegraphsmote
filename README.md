# CodeGraphSMOTE - Data Augmentation for Vulnerability Discovery

## Description

This repository contains the source code for our paper "CodeGraphSMOTE - Data Augmentation for Vulnerability Discovery", currently under review at [DBSec 23](https://www.dbsec2023.unimol.it/)

## Requirements

- Python
- PyTorch
- PyTorch Geometric
- NetworkX
- imbalanced-learn
- gensim
- tokenizers
- pandas

For the transformer reconstruction demo:
- dash (for the transformer reconstruction demo)
- [cpg-to-dot](https://github.com/SAP-samples/security-research-taintgraphs)

## Training data

Training data for the various datasets can be obtained at:

- [Devign](https://sites.google.com/view/devign) (FFmpeg+QEMU)
- [ReVeal](https://github.com/VulDetProject/ReVeal) (Chromium+Debian)
- [PatchDB](https://sunlab-gmu.github.io/PatchDB/)

From the commits, methods are extracted as vulnerable prior to the fix commit and as non-vulnerable after the fix commit, as described in Devign and ReVeal. Afterwards, the resulting C code is processed using [Fraunhofer-CPG](https://github.com/Fraunhofer-AISEC/cpg). A single file per method containing the cpg in Graphviz DOT language needs to be placed in the cache folders of this directory (alternatively the paths in `params/dataset_params.py` can be changed). The processed CPG-files can be created ergonomically using [cpg-to-dot](https://github.com/SAP-samples/security-research-taintgraphs).

## Scripts relevant to the reproduction of the results

- `notebooks/`
    - `analyze_cwe.ipynb`
        Visualization of the distances between CWE clusters. Used for the right-hand side of figure 4
    - `degree_vis.ipynb`
        Notebook containing the code for visualizations of the average degree against the number of nodes. Used for figures 2b and 2c.
- `params/`
    Hyperparameters of training, models and datasets as well as paths to the data and various other configuration
- `scripts/`
    - `cpg_reconstruction/`
        - `demo.py`
            Interactive demonstration the reconstruction of code from a CPG  using the trained transformer
        - `demo2.py`
            Interactive demonstration of the interpolation between two code samples and reconstruction using the transformer; used for figure 3
        - `train.py`
            Training of the CPG reconstruction transformer
    - `cwe_distances.py`
        Generates the data needed for `notebooks/analyze_cwe.ipynb`
    - `degree_per_node.py`
        Generates figure 2a
    - `draw_cwes.py`
        Generates left-hand side of figure 4
    - `plot_percentages.py`
        Used for creating figure 5
    - `quick_preprocess.py`
        Parallelized version of data preprocessing. Use this before training on any dataset
    - `view_results.py`
        Creates textual summaries of the cross-validation experiments. Used for table 1
- `cv_classifier.py`
    Used to generate the results on the full dataset with cross-evaluation for table 1
- `cv_subsampling_drop.py`
    Used to generate the subsampled results shown as "Node-Dropping" in figure 5
- `cv_subsampling_sard.py`
    Used to generate the subsampled results shown as "SARD" in figure 5
- `cv_subsampling_smote.py`
    Used to generate the subsampled results shown as "CodeGraphSMOTE" in figure 5
- `cv_subsampling.py`
    Used to generate the subsampled results shown as "Downsampled" in figure 5
- `train_vgae.py`
    Training of the VGAE model used for CodeGraphSMOTE

All implementations of models, training and data processing are in `experiments/`. All other files are utility files to ease implementation of the scripts.

## How to obtain support
[Create an issue](https://github.com/SAP-samples/security-reseearch-codegraphsmote/issues) in this repository if you find a bug or have questions about the content.
 
For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2023 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSE) file.
