"""
    Build set of decoys for each ligand using DecoyFinder.
"""
import sys
import os
import pickle
import tempfile

from pybel import *

if __name__ == "__main__":
    sys.path.append('..')

from data_processor.pybel_fp import index_to_vec

from decoy_finder import *

def build_decoy_dict(lig_dict, decoy_dict=None):
    """ Build decoys for ligand dictionary.
        Dictionary contains SMILES (value) for each ligand ID (key)
        You can pass an already started decoy_dict, if None starts a new
        decoy dictionary.
    """
    #dict: {'lig_code': [(lig_fp, [decoy_fps])]}
    if not decoy_dict:
        decoy_dict = {}
    for name, sm in lig_dict.items():
        if name in decoy_dict:
            print(f">>> already did {name}")
            continue
        try:
            os.remove('decoys.sdf')
            os.remove('lig.txt')
        except:
            pass
        #call decoy finder
        print(f">>> decoy generation for {name}, {sm}")
        with open('lig.txt', 'w') as l:
            l.write(sm)
        for _ in find_decoys(query_files=['lig.txt'], db_files=['../data/in-vitro.csv'], outputfile='decoys.sdf'):
            pass

        # parse output SDF.
        try:
            mols = readfile("sdf", "decoys.sdf")
        except:
            continue
        ligand_fp = index_to_vec(readstring('smi', sm).calcfp(fptype='maccs').bits, nbits=166)
        decoys = []
        for mol in mols:
            fp = mol.calcfp(fptype="maccs").bits
            fp = index_to_vec(fp, nbits=166)
            decoys.append(fp)
        decoy_dict[name] = (ligand_fp, decoys)

        os.remove('decoys.sdf')
        os.remove('lig.txt')

        pickle.dump(decoy_dict, open('decoys_zinc.p', 'wb'))
    return decoy_dict

if __name__ == "__main__":
    lig_dict = pickle.load(open('../data/smiles_ligs_dict.p', 'rb'))
    decoy_dict = pickle.load(open('decoys_zinc.p', 'rb'))
    decs = build_decoy_dict(lig_dict, decoy_dict=decoy_dict)
    pickle.dump(decs, open('decoys_zinc.p', 'wb'))
    pass
