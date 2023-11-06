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

Predict docking scores:

```
python experiemnts/train.py train.target='dock' train.loss='l2'
```

Predict whether or not a ligand is native

```
python experiemnts/train.py train.target='is_native' train.loss='bce'
```

Predict the native ligand directly

```
python experiemnts/train.py train.target='native_fp' train.loss='bce'
```

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

The first step to reproduce the results and figures from the paper is to obtain the raw results files and csvs.
Those files are directly available for download at :

[//]: # (TODO)
If you want to reproduce the results from scratch, first you need to set up the environment,
data and model as detailed above. 

[//]: # (Then, you need to pretrain a model that follows RNAmigos1 and one using directed graphs and )
[//]: # (hungarian similarity function, there is a script to pretrain models in *job_scripts/*.)
Then, you need to pretrain a model, by running :
```bash    
python experiments/pretrain.py name=pretrained_hungarian_64
```
Then you need to train models using those pretrained models. 
Finally, once models are trained, you will need to make an inference on the test set for each trained model, resulting 
in output csvs containing the pocket id, ligand id and predicted score.
Scripts are available to run all relevant trainings and inferences.
```bash
bash job_scripts/paper_runs.sh
# TODO add inference script and also for rdock
```

Now we expect the *outputs/* directory to hold paper_{fp, native, dock, rdock}_{, _raw}.csv files.
The first step is to produce ensembling results, by running 
```bash
cd fig_scripts
python mixing.py
```

We now have the table of mixing results as well as the best ensemble models. 
In addition, we have blended our models to produce csvs with mixed and mixed+rdock results.

We can now run : 
```bash
python violins.py
python ef_time.py
```
to produce the remaining Figures of the paper.
