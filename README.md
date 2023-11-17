# RNAmigos 2.0

Welcome on RNAmigos 2.0 ! 

<img src="images/vs_fig.png">

## Table of Contents: 

- [Description](#description)
- [Using the tool with Collab](#Using-the-tool-with-Collab)
- [Using the tool locally](#Using-the-tool-locally)
- [Reproducting results and figures](#Reproducting-results-and-figures)

## Description

RNAmigos is a virtual screening tool : given the binding site of a target and a library of chemical compounds, it 
ranks the compounds so that better ranked compounds have a higher chance to bind the target.
It is based on a machine learning model using the PyTorch framework and was trained leveraging unsupervised and synthetic data.
It was shown to display similar enrichment factors to docking while running in a fraction of the time.
A detailed description of the tool is available at :

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

### Setup
A local use of the tool is also possible by following the next steps.
First, create a conda environment:

```bash
conda create -n rnamigos2
conda activate rnamigos2
pip install -r requirements.txt
```

The only data needed to make predictions are the weights of the models.
They are available at :

[//]: # (TODO : get link to model weights and detail how to get them all.)

and needs to be saved in `saved_models/`.

### Making predictions on your targets

To run RNAmigos2.0 on your own target and ligands, use the `experiments/inference.py` script.

You will need to provide the following:

* Path to an mmCif file
* Path to a .txt file with one SMILES string per line
* A list of binding site residue identifiers 

To convert the mmCif to a 2.5D graph you will need to make sure you have the latest rnaglib and an optional dependency of rnaglib.
You can just run the inference script to get a score for each ligand in your SMILES .txt file.

Taking example structure and ligand file from `/sample_files`, selecting residues `16-20` of chain `A` as the binding site, the corresponding command is :
Now you just run the inference script to get a score for each ligand in your SMILES .txt file.
```
python experiments/inference.py cif_path=sample_files/3ox0.cif \
                                pdbid=3ox0 \
                                residue_list=\[A.20,A.19,A.18,A.17,A.16\] \
                                ligands_path=sample_files/test_smiles.txt \
                                out_path=scores.txt
``` 

Once this executes you will have `scores.txt` that looks like this:

```
CCC[S@](=O)c1ccc2[nH]/c(=N\C(=O)OC)[nH]c2c1 0.2639017701148987
O=C(O)[C@@H](O)c1ccccc1 0.6267350912094116
CC(=O)Oc1ccccc1C(=O)O 0.6304176449775696
CN1[C@H]2CC[C@@H]1CC(OC(=O)[C@H](CO)c1ccccc1)C2 0.47674891352653503
...
```

## Reproducting results and figures

The steps necessary to reproduce results and figures are detailed in REPRODUCE.md
