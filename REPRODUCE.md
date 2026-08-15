# Reproducing the results

All the initial and processed files are directly available for download on [Zenodo](https://doi.org/10.5281/zenodo.14803961).

Below are the steps necessary to reproduce most of these files from publicly available databases, training models and using inference and plotting scripts.

## Setting up the environment

If you want to reproduce the results from scratch, first you need to set up the environment,
data and model as detailed in the README.md.

Any data files you download with the links below should be placed at the `/data` folder at the root of this repository.

Make sure to setup the repo:

```
pip install .
```

## Gathering the data

If you want already pre-processed data files just run:

```
cd data
tar -xzvf rnamigos2_data.tar.gz
```

and skip to the _Generate actives and decoys list_ section (ligands stored in our format are a bit heavy).

### Getting initial pocket data

The pocket graphs are in the Zenodo archive, at `data/json_pockets_expanded`.

Initially, we download all pdb containing RNA + small molecule, filter them and save resulting pockets as node ids +
ligands...

[//]: # (TODO : include steps to get the original pockets.)

### Get docking scores

The docking scores are in the Zenodo archive, at `data/rnamigos2_dataset_consolidated.csv`.

[//]: # (We can now proceed to docking all relevant pairs.)
[//]: # (The docking experiment can be launched using :)
[//]: # (TODO : upload docking scripts_prepare)

To split these raw results into csvs adapted for each of our training scenarios, one can run

```bash
python scripts_prepare/build_csvs.py
```

This should take about 3 minutes, and result in data/csvs/{docking_data,fp_data,binary_data}.csv

### Pockets as 2.5d graphs, ligands as graphs and fingerprints

We now want to prepare our pockets and ligands for learning our tool.
This can be obtained using our scripts, and requires downloading all RNA as 2.5D graphs, using _rnaglib_ download tool.

[//]: # (TODO : RIGHT NOW, we need to have json_pockets/ because the node ids are broken...)
[//]: # (TODO : This requires having rnaglib_all data, maybe we should mention how to get that)

```bash
python scripts_prepare/get_pocket_graphs.py
```

### Splitting the data

The splits are directly available through our git repo as pickle files, in data/train_{val,test}_75.p

**These pickles are the split, not the `SPLIT` column of the csvs.** With `train.rnamigos1_split: -2`, which
every shipped config uses, `get_systems` selects train and test pockets by membership in `data/train_test_75.p`
and ignores the `SPLIT` column entirely. That column is a leftover from an older split based on PDB names
(the `train.rnamigos1_split: -1` path), it labels rows rather than pockets, and it disagrees with the split
actually used on most of the test set. Do not use it to check which pockets were held out.

To reproduce these splits, we first need to compute RMScores between all pockets.
The RMScores are in the Zenodo archive, at `data/rmscores/`.

[//]: # (TODO : Add RMscores computations)

Now that we have the file data/rmscore_normalized_by_average_length_complete_dataset.csv, 
we can split the data according to the RMscores.
Simply run:

```bash
python scripts_prepare/split.py
```
### Generate actives and decoys list

The decoy sets used for virtual screening are built from
`data/rnamigos2_dataset_consolidated.csv`, which is included in the Zenodo archive, so there is
nothing extra to download:

```bash
python scripts_prepare/build_ligand_db.py
```

This writes the `pdb`, `pdb_chembl` and `chembl` sets to `data/ligand_db/`. The results in the
paper are reported on `chembl`.

Note that the three sets are not nested the way the names suggest. `chembl` is the ~500 drug-like
ChEMBL compounds docked against that individual pocket, while `pdb_chembl` mixes the PDB ligands
with the full global ChEMBL pool, so `pdb_chembl` is not `pdb` + `chembl`.

The ROBIN and DecoyFinder sets are separate:

```
python scripts_prepare/build_screen_data.py --robin
python scripts_prepare/build_screen_data.py --decoyfinder
```

We save them in `data/ligand_db/` as well.

NOTE: you will need to install pybel if you want DecoyFinder decoys. This depends on an OpenBabel installation.
The easiest way is to install openbabel through conda or compile OpenBabel and then pip install openbabel.
DecoyFinder samples ligands from a given library. In this case we use the ZINC in-vitro bioactive compounds from the
[RNAmigos1 paper](https://academic.oup.com/nar/article/48/14/7690/5870337), whose data is on
[Zenodo](https://zenodo.org/record/8338267). Take `rnamigos_1_data/data/decoys/in-vitro.csv` out of that archive and
put it at `data/decoy_libraries/in-vitro.csv`.

We now have pockets, native ligands and different sets of decoys.

## Model training and inference

The pretrained weights used by the paper are already in `pretrained/`, so this section is only needed if you want to
redo the pretraining yourself.

The whole RNAs come from rnaglib, as the non-redundant annotated graph set:

```bash
rnaglib_download -r nr -a
```

This lands under `~/.rnaglib/datasets/` (e.g. `rnaglib-nr-1.0.0-annotated`). Point `data.pretrain_graphs` at it, or
symlink it to `data/pretrain_data/nr-graphs_annotated`. The RNA-FM embeddings are computed on the fly by rnaglib's
`RNAFMTransform` and cached at `data.rnafm_cache_pretrain`, so there is nothing else to fetch.

Pretrain a model, by running :

```bash    
python rnamigos/pretrain.py name=pretrained_hungarian_64
```

We additionally need to load optimol encoder pretrained weights which are in the `pretrained/optimol/` path already
included in the repository.

Then you need to train models using those pretrained models.
Scripts are available to run all relevant trainings.

```bash
bash scripts_run/train.sh
```

This will train three models and save them in results/trained_models.
Moreover, this will compute the prediction of those models on the test set and different decoy sets.
The result of these predictions are dumped in outputs.
You will get a {model_name}.csv containing the pocket id and AuRoc score for different decoy sets.
You will also get a {model_name}_raw.csv containing the pocket id, ligand id, ligand source (different decoys sources)
and predicted score.

To get results in a similar format for rDock, please run:

```bash
python scripts_run/rdock_output.py
```

Finally, you can generate csvs containing RNAmigos results for the validation on pockets with Chembl decoys and ROBIN
validation by running:

```bash
python scripts_run/rdock_ouptut.py
python scripts_run/chembl_inference.py
python scripts_run/robin_inference.py
```

## Generate the figures from the results

[//]: # (TODO get dl files)


We now have the table of mixing results as well as the best ensemble models. 
To obtain the different plots of the paper, one can now run :

```bash
python scripts_fig/violins.py
python scripts_fig/ef_time.py
python scripts_fig/panel_1.py
python scripts_fig/robin_analyze.py
```
