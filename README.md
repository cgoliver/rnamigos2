# RNAmigos 2.0

## Install

Create an environment if you like and then:

`pip install -r requirements.txt`

## Get Data

These are all tarballs that you should put in the `data/` folder at the respository root and untar.

1. These are networkx graphs, one for each pocket, annotated with the value for the intermolecular term (`INTER`) score from `RDOCK`.

* [Docked pockets test set](https://drive.proton.me/urls/RSZ2V97TXG#z06rtSrHNGxU)
* [Docked pockets train set](https://drive.proton.me/urls/RSZ2V97TXG#z06rtSrHNGxU)


## Configs

All configs are in the `conf/` folder.

* `conf/learning.yaml` controls the model training

## Train a model

```
python experiments/train.py
```

## Pre-train a model

## Load a pretrained model

## Compute Enrichment Factor

```
python experiments/eval.py

```

## Generate figures from paper

