# RNAmigos 2.0

Welcome on RNAmigos 2.0 !

<img src="figs/rnamigos2.png">

## Table of Contents:

- [Description](#description)
- [Using the tool with Collab](#Using-the-tool-with-Collab)
- [Using the tool locally](#Using-the-tool-locally)
- [Reproducting results and figures](#Reproducting-results-and-figures)

## Description

RNAmigos is a virtual screening tool : given the binding site of a target and a library of chemical compounds, it
ranks the compounds so that better ranked compounds have a higher chance to bind the target.
It is based on a machine learning model using the PyTorch framework and was trained leveraging unsupervised and
synthetic data.
It was shown to display similar enrichment factors to docking while running in a fraction of the time.
A detailed description of the tool is available
on [BioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.23.568394v2).

If you find this tool useful, please cite

```bib
@article{carvajal2023rnamigos2,
  title={RNAmigos2: Fast and accurate structure-based RNA virtual screening with semi-supervised graph learning and large-scale docking data},
  author={Carvajal-Patino, Juan G and Mallet, Vincent and Becerra, David and Ni{\~n}o Vasquez, Luis Fernando and Oliver, Carlos and Waldisp{\"u}hl, J{\'e}r{\^o}me},
  journal={bioRxiv},
  pages={2023--11},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Using the tool with Collab

The easiest way to use the tool is to use Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cgoliver/rnamigos2/blob/master/rnamigos_inference.ipynb)

You will need to provide a cif file, a binding site in the form of a list of binding pocket nodes and a list of ligand
smiles.

## Using the tool locally

A local use of the tool is also possible by following the next steps.
NOTE: This has been tested on Linux Ubuntu 24 and Mac OS 13 and 14. No special hardware requirement, inference code runs
on common desktops and laptops.

First, create a conda environment:

```bash
git clone https://github.com/cgoliver/rnamigos2.git
cd rnamigos2/
conda create -n rnamigos2
conda activate rnamigos2
pip install -r requirements.txt
```

To run RNAmigos2.0 on your own target and ligands, use the `rnamigos/inference.py` script.

You will need to provide the following:

* Path to an mmCif file
* Path to a .txt file with one SMILES string per line
* A list of binding site residue identifiers

Now you can just run the inference script to get a score for each ligand in your SMILES .txt file.
Taking example structure and ligand file from `/sample_files`, selecting residues `16-20` of chain `A` as the binding
site, the corresponding command is :

```
python rnamigos/inference.py cif_path=sample_files/3ox0.cif \
                                pdbid=3ox0 \
                                residue_list=\[A.20,A.19,A.18,A.17,A.16\] \
                                ligands_path=sample_files/test_smiles.txt \
                                out_path=scores.txt
``` 

Once this executes (~10 seconds) you will have `scores.txt` that looks like this:

```
CCC[S@](=O)c1ccc2[nH]/c(=N\C(=O)OC)[nH]c2c1 0.2639017701148987
O=C(O)[C@@H](O)c1ccccc1 0.6267350912094116
CC(=O)Oc1ccccc1C(=O)O 0.6304176449775696
CN1[C@H]2CC[C@@H]1CC(OC(=O)[C@H](CO)c1ccccc1)C2 0.47674891352653503
...
```

The scores are between 0 and 1 with a higher score representing a better likelihood of binding.

**NOTE:** inference on user-provided structures has not been validated as it uses fr3d-python as a structure annotation
backend which was not used in training. The models provided were trained on structures annotated by x3dna-dssr.

## Reproducting results and figures

The steps necessary to reproduce results and figures are detailed in `REPRODUCE.md`.
