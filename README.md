# Boosting Graph Contrastive Learning via Graph Contrastive Saliency

This is the code for Boosting Graph Contrastive Learning via Graph Contrastive Saliency (GCS).
GCS adaptively screens the semantic-related substructure in graphs by capitalizing on the proposed gradient-based Graph Contrastive Saliency (GCS). The goal is to identify the most semantically discriminative structures of a graph via contrastive learning, such that we can generate semantically meaningful augmentations by leveraging on saliency.

## Requirements

To install requirements:

```setup
conda env create -f environment.yaml
```

## Unsupervised Learning

To train the model for unsupervised graph-level tasks:

```setup
python unsupervised.py
```

## Transfer Learning
Please refer to https://github.com/snap-stanford/pretrain-gnns#installation for environment setup and https://github.com/snap-stanford/pretrain-gnns#dataset-download to download dataset.


To pretrain the model(s) in the paper for transfer learning:

```setup
python transfer_pretrain.py
```
> Output: the file "latest.tar"

To finetune the model(s) for downstream tasks:
```setup
python transfer_finetune.py
```

