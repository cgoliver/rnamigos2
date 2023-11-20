"""
Set of functions for creating and handling molecular fingerprints.
"""
import pickle
import math
import sys

import numpy as np
from tqdm import tqdm
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import jaccard
from scipy.stats import entropy
import matplotlib.pyplot as plt
from pybel import *

def smiles_dict(sdf_dir, include_ions=False, bits=False, fptype='FP2'):
    """
    Returns a dictionary where the key is a ligand ID and value is a 1024
    molecular fingerprint object. 
    
    Reads from a folder of SDF files.
    
    Arguments:
        sdf_dir (str): path to folder with sdf files.
       
    Returns:
        dict: dictionary with 3-letter IDs as keys and fingerprint vectors as values.
    """
    smiles = {}
    sdf_list = os.listdir(sdf_dir)
    for s in tqdm(sdf_list):
        sdf = os.path.join(sdf_dir, s)
        mols = readfile("sdf", sdf)
        mol_id = s.split("_")[0]
        print(mol_id)
        for m in mols:
            # mol_id = m.data['ChemCompId']
            if mol_id in smiles:
                continue
            # mol_id = m.data["PUBCHEM_COMPOUND_CID"]
            # mol_name = m.data['Name']
            smiles[mol_id] = m.write(format='smi')
    #these ligands keep failing. hard code them
    smiles['GLY'] = 'NCC(O)=O'
    smiles['LYS'] = 'N[C@@H](CCCC[NH3+])C(O)=O'
    smiles['FME'] = 'CSCC[C@H](NC=O)C(O)=O'
    smiles['CTC'] = 'CN(C)[C@H]1[C@@H]2C[C@H]3C(=C(O)[C@]2(O)C(=O)C(C(N)=O)=C1O)C(=O)c1c(O)ccc(Cl)c1[C@@]3(C)O'
    smiles['TAC'] = 'CN(C)C1[C@@H]2C[C@H]3C(=C(O)[C@]2(O)C(=O)C(C(N)=O)=C1O)C(=O)c1c(O)cccc1[C@@]3(C)O'

    return smiles

def fp_dict(smiles_file, include_ions=False, bits=False, fptype='FP2'):
    """
    Returns a dictionary where the key is a ligand ID and value is a 1024
    molecular fingerprint object.
    Reads from a file with a smiles string and 3-letter code per line.
    
    Arguments;
        smiles_file (str): path to file with smiles
        include_ions (bool): whether to include ions in fingerprint dictionary
        bits (bool): convert index vector to bit array
        fptype (str): 'fp2' 1024 fingerprint or 'maccs' 166 binary.

    Returns:
        dict: dictionary with 3-letter codes as keys and fingerprints as values.

    """

    nbits = {'maccs': 166, 'FP2': 1024}
    fps = {}

    with open(smiles_file, "r") as sms:
        for s in sms:
            smile, name = s.split()
            try:
                mol = readstring('smi', smile)
            except:
                continue
            fp = mol.calcfp(fptype=fptype)
            if bits:
                fp = index_to_vec(fp.bits, nbits=nbits[fptype])
            fps[name] = fp
    #these ligands keep failing. hard code them
    # fps['GLY'] = readstring('smi', 'NCC(O)=O').calcfp(fptype=fptype)
    # fps['LYS'] = readstring('smi', 'N[C@@H](CCCC[NH3+])C(O)=O').calcfp(fptype=fptype)
    # fps['FME'] = readstring('smi', 'CSCC[C@H](NC=O)C(O)=O').calcfp(fptype=fptype)
    # fps['CTC'] = readstring('smi', 'CN(C)[C@H]1[C@@H]2C[C@H]3C(=C(O)[C@]2(O)C(=O)C(C(N)=O)=C1O)C(=O)c1c(O)ccc(Cl)c1[C@@]3(C)O').calcfp(fptype=fptype)
    # fps['TAC'] = readstring('smi', 'CN(C)C1[C@@H]2C[C@H]3C(=C(O)[C@]2(O)C(=O)C(C(N)=O)=C1O)C(=O)c1c(O)cccc1[C@@]3(C)O').calcfp(fptype=fptype)
    return fps

def index_to_vec(fp, nbits=1024):
    """
    Convert list of 1 indices to numpy binary vector.

    Returns:
        `array`: 1x1024 binary numpy vector
    """
    # vec = np.zeros(166)
    vec = np.zeros(nbits)
    vec[fp] = 1
    return vec

if __name__ == "__main__":
    # all_ligs = fp_dict("../data/ligs", bits=True, fptype='maccs')
    # smiles = smiles_dict("../data/ligs")
    # pickle.dump(smiles, open("../data/smiles_ligs_dict.p", "wb"))
    # all_ligs = fp_dict("../data/pdb_rna_smiles.txt", bits=True, fptype='maccs')
    all_ligs = fp_dict("../data/all_ligs_pdb.txt", bits=True, fptype='maccs')
    pickle.dump(all_ligs, open("../data/all_ligs_pdb_maccs.p", "wb"))
