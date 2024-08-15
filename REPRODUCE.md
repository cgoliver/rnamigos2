# Reproducing the results

If you want to reproduce the results from scratch, first you need to set up the environment,
data and model as detailed in the README.md.

Any data files you download with the links below should be placed at the `/data` folder at the root of this repository.

Make sure to setup the repo:

```
pip install .
```

If you want already pre-processed data files just run:

```
cd data
tar -xzvf rnamigos2_data.tar.gz
```

and skip to the model training section.

## Getting initial pocket data

Initially, we download all pdb containing RNA + small molecule, filter them and save resulting pockets as node ids +
ligands...

[//]: # (TODO : include steps to get the original pockets.)


Download pocket graphs [here](https://drive.proton.me/urls/SC9AQCF2SC#JYQ3K9yNUJ4U)

## Generate actives and decoys list

Once equiped with this initial data, we select the decoys corresponding to our actives.
This is done by running :

```
python scripts_prepare/build_screen_data.py --pdb --decoyfinder
```

We save them in `data/ligand_db/`

NOTE: you will need to install pybel if you want DecoyFinder decoys. This depends on an OpenBabel installation.
The easiest way is to install openbabel through conda or compile OpenBabel and then pip install openbabel.
DecoyFinder samples ligands from a given library. In this case we use ZINC in-vio bioactive compounds which you can
download [here](https://drive.proton.me/urls/CQMXCX5MW4#YQeEEa7VHVcu)

We now have pockets, native ligands and different sets of decoys.

These decoys can be downloaded [here](https://drive.proton.me/urls/6XCM553QBC#1NR2xU9W3CkR)

## Get docking scores

We can now proceed to docking all relevant pairs.
The docking experiment can be launched using :

[//]: # (TODO : upload docking scripts_prepare)

and the corresponding docking scores can be obtained [here](https://drive.proton.me/urls/TZJ7R8T8T0#RCd1LK8uu1MK)

[//]: # (TODO : check that the data is ok)

To split this initial data into csvs adapted for each of our training scenarios, one can run

```bash
python scripts_prepare/build_csvs.py
```

This should take about 3 minutes.

## Pockets as 2.5d graphs, ligands as graphs and fingerprints

We now want to prepare our pockets and ligands for learning our tool.
This can be obtained using our scripts.

[//]: # (TODO : RIGHT NOW, we need to have json_pockets/ because the node ids are broken...)

[//]: # (TODO : This requires having rnaglib_all data, maybe we should mention how to get that)

```bash
python scripts_prepare/get_pocket_graphs.py
```

## Splitting the data

We first need to compute RMScores and then to split the data according to the RMscores.

[//]: # (TODO : Add RMscores computations)

Now that we have the file data/rmscore_normalized_by_average_length_complete_dataset.csv, we can obtain our
final splits. Simply run:

```bash
python scripts_prepare/split.py
```

## Model training

Fetch the whole RNAs for pretraining [here](https://drive.proton.me/urls/Y8TTCWKDVC#vs29rzJ1h9YN)

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

# Generate the figures from the results

If you followed the previous steps, we expect the *outputs/* directory to hold {native, dock, rdock}_{, _raw}.csv files.

Those files are directly available for download at :

[//]: # (TODO get dl files)


The first step is to produce ensembling results, by running

```bash
python scripts_fig/mixing.py
```

We now have the table of mixing results as well as the best ensemble models.
In addition, we have blended our models to produce csvs with mixed and mixed+rdock results.

We can now run :

```bash
python scripts_fig/violins.py
python scripts_fig/ef_time.py
```

to produce the remaining Figures of the paper.

For the ROBIN experiment, we will first need to produce predictions for four ROBIN targets.
To get these predictions, please run (takes about 20 minutes):

```
python scripts_fig/robin_inference.py
```

Finally, you will obtain the ROBIN plot with :

```
python scripts_fig/robin_fig.py
```