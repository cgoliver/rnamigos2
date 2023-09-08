# RNAmigos 2.0

## Install

Create an environment if you like and then:

`pip install -r requirements.txt`

## Get Data

These are all tarballs that you should put in the `data/` folder at the respository root and untar.

1. These are networkx graphs, one for each pocket, annotated with the value for the intermolecular term (`INTER`) score from `RDOCK`.

* [Docked pockets test set](https://drive.proton.me/urls/RSZ2V97TXG#z06rtSrHNGxU)
* [Docked pockets train set](https://drive.proton.me/urls/929Z2M4YWC#pkwIdM4TZAqR)
* [Pretraining data](https://drive.proton.me/urls/YKNV0M1WBR#s0E0cMSTvpsH)
* [Binding scores](https://drive.proton.me/urls/TZJ7R8T8T0#RCd1LK8uu1MK)


## Generate actives and decoys list


```
python scripts/build_screen_data.py
```


## Configs

All configs are in the `conf/` folder.

* `conf/learning.yaml` controls the model training

## Train a model

```
python experiments/train.py
```

Pass the `--help` flag to get all the options.

## Pre-train a model

```
python experiments/pretrain.py name=default 

```

Pass the `--help` flag to get all the options.

## Load a pretrained model

```
python experiments/train model.use_pretrained=true model.pretrained_path=pretrained/default/model.pth
```

## Compute Enrichment Factor

```
python experiments/eval.py

```

## Generate figures from paper

