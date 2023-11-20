# Reproducting the results
If you want to reproduce the results from scratch, first you need to set up the environment,
data and model as detailed in the README.md. 

## Getting initial pocket data

Initially, we download all pdb containing RNA + small molecule, filter them and save resulting pockets as node ids + ligands...

[//]: # (TODO : include steps to get the original pockets.)


## Generate actives and decoys list
Once equiped with this initial data, we select the decoys corresponding to our actives.
This is done by running : 

```
python scripts/build_screen_data.py
```

We save them in `data/ligand_db/`

NOTE: you will need to install pybel if you want DecoyFinder decoys. This depends on an OpenBabel installation. 
The easiest way is to install openbabel through conda or compile OpenBabel and then pip install openbabel.
To disable DecoyFinder decoys pass the ``-no-decoyfinder`` flag.

We now have pockets, native ligands and different sets of decoys.
These decoys can be downloaded here : TODO [Decoy Library](https://drive.proton.me/urls/6XCM553QBC#1NR2xU9W3CkR)

## Get docking scores

We can now proceed to docking all relevant pairs.
The docking experiment can be launched using :

[//]: # (TODO : upload docking scripts)
 
and the corresonding docking scores can be obtained directly at [Binding scores](https://drive.proton.me/urls/TZJ7R8T8T0#RCd1LK8uu1MK)

[//]: # (TODO : check that the data is ok)

To split this initial data into csvs adapted for each of our training scenarios, one can run 
```bash
python scripts/build_csvs.py
```

## Pockets as 2.5d graphs, ligands as graphs and fingerprints

We now want to prepare our pockets and ligands for learning our tool.
This can be obtained using our scripts 
```bash
python scripts/get_pockets.py
```

[//]: # (TODO : get scripts going)


## Models training
Now that we have all the necessary data, we can start training models.

First, we need to pretrain a model, by running :
```bash    
python experiments/pretrain.py name=pretrained_hungarian_64
```

We additionally need to load optimol encoder pretrained weights 

[//]: # (TODO : include optimol's weights)

Then you need to train models using those pretrained models.
Scripts are available to run all relevant trainings.
```bash
bash job_scripts/paper_runs.sh
```

Finally, once models are trained, you will need to make an inference on the test set for each trained model, resulting 
in output csvs containing the pocket id, ligand id and predicted score.
The corresponding script needs to be run : 

[//]: # (TODO CARLOS add inference script)
[//]: # (TODO script for rdock to get it as csv)

# Generate the figures from the results

If you followed the previous steps, we expect the *outputs/* directory to hold paper_{fp, native, dock, rdock}_{, _raw}.csv files.
Those files are directly available for download at :

[//]: # (TODO get dl files)


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
