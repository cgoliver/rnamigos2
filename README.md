# RNAmigos 2.0

## Install

Create an environment if you like and then:

`pip install -r requirements.txt`

## Get Data

These are all tarballs that you should put in the `data/` folder at the respository root and untar.

1. These are networkx graphs, one for each pocket, annotated with the value for the intermolecular term (`INTER`) score from `RDOCK`.

* [Docked pockets test set](https://drive.proton.me/urls/9656ESVF8G#2PQYJZyqcDMs)
* [Docked pockets train set](https://drive.proton.me/urls/1HVNSQG6NM#8G2DytahZ1Pc)


## Configs

All configs are in the `conf/` folder.

* `conf/learning.yaml` controls the model training

## Train a model

```
python experiments/train.py
```


## Generate figures from paper



