# RNAmigos 2.0

Welcome on RNAmigos 2.0 ! 

## Table of Contents: 

- [Description](#description)
- [Using the tool with Collab](#Using-the-tool-with-Collab)
- [Using the tool locally](#Using-the-tool-locally)
- [Reproducting the results](#Reproducting-the-results)
- [Generate the figures from the results](#Generate-the-figures-from-the-results)

## Description

RNAmigos is a virtual screening tool : given the binding site of a target and a library of chemical compounds, it 
ranks the compounds so that better ranked compounds have a higher chance to bind the target.
It is based on a machine learning model using the PyTorch framework and was trained leveraging unsupervised and synthetic data.
It was shown to display similar enrichment factors to docking while running in a fraction of the time.
A detailed description of the tool is available at

[//]: # (**TODO : insert link to publication**)

If you find this tool useful, please cite 

[//]: # (**TODO : add bib**)
```bib

```

## Using the tool with Collab

The easiest way to use the tool is to use Google Colab, provided at this link

[//]: # (TODO : setup link. )
You will need to provide a binding site, in the form of a list of binding pocket nodes. 
If you want to use your tool on unpublished data, you can additionally provide a custom PDB file containing those files.


## Using the tool locally

### Local environment
A local use of the tool is also possible by following the next steps.
First, create an environment if you like and then:

`conda create -n rnamigos2`
`conda activate rnamigos2`
`pip install -r requirements.txt`

### Getting the data
The only data needed to make predictions are the weights of the models.
They are available at :

[//]: # (TODO : get link to model weights.)

and needs to be saved in saved_models.

### Making a prediction

You will need a binding site structure. 
The first thing you need is a list of binding site nodes in the rnaglib format (pdbid.chain.resnumber) in a text file.
* If your RNA is a canonical one (present in the PDB), then it's 2.5d graph representation is already part of the rnaglib release.
To get this release you can use :
`rnaglib download`
* Otherwise, you can input a PDB file of the binding site. 
You should include some context around the binding site (a margin of approximately three residues)

You will also need a ligand file : a text file containing one smiles per line

Then, simply run : 

[//]: # (TODO : build the script)
`python main.py --pocket {pdb_nodes}.txt --ligands {ligands_file}.txt --outname {my_results}.csv`

And a csv called my_results.csv will be dumped with one line per ligand with its smile and corresponding score.

## Reproducting the results
If you want to reproduce the results from scratch, first you need to set up the environment,
data and model as detailed above. 

### Getting initial data

[//]: # (TODO : include steps to get the original csv.)


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

Now that we are equiped with the csv.
* [Docked pockets test set](https://drive.proton.me/urls/RSZ2V97TXG#z06rtSrHNGxU)



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

## Generate the figures from the results

If you followed the previous steps, we expect the *outputs/* directory to hold paper_{fp, native, dock, rdock}_{, _raw}.csv files.
Those files are directly available for download at :

[//]: # (TODO)


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
