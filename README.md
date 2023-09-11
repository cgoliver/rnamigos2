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
* [Decoy Library](https://drive.proton.me/urls/YGHQV867NG#RuVM8TLFOdKH)


## Generate actives and decoys list


```
python scripts/build_screen_data.py
```

NOTE: you will need to install pybel if you want DecoyFinder decoys. This depends on an OpenBabel installation. 
The easiest way is to install openbabel through conda or compile OpenBabel and then pip install openbabel.
To disable DecoyFinder decoys pass the ``-no-decoyfinder`` flag.


## Configs

All configs are in the `conf/` folder.

* `conf/train.yaml` controls the model training
* `conf/pretrain.yaml` controls the model pretraining
* `conf/evaluate.yaml` controls the model virtual screening evaluation 

## Train a model

```
python experiments/train.py
```

Pass the `--help` flag to get all the options.

We have 3 options for training modes (`train.target`):
    * `dock`: predict the docking INTER score (regression)
    * `is_native`: predict whether the given ligand is the native for the given pocket (binary classification)
    * `native_fp`: given only a pocket, predict the native ligand's fingerprint. This is the RNAmigos1.0 setting (multi-label classification)` 

Make sure to set the correct `train.loss` given the target you chose. Optioins:

    * `l1`: L2 loss
    * `l2`: L1 loss
    * `bce`: Binary crossentropy

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

