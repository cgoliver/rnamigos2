""""
Analyzes PDB structures with ligands and counts the number of
RNA vs Protein residues at various sphere radii.

The output of this script can be used to select binding pockets.
"""
import os
import sys
import pickle
import multiprocessing

import numpy as np

import pandas as pd
from Bio.PDB import *

amino_acids = [
    'ALA',
    'ARG',
    'ASN',
    'ASP',
    'CYS',
    'GLU',
    'GLN',
    'GLY',
    'HIS',
    'ILE',
    'LEU',
    'LYS',
    'MET',
    'PHE',
    'PRO',
    'SER',
    'THR',
    'TRP',
    'TYR',
    'VAL'
]


def ligand_center(residue):
    return np.mean(np.array([atom.coord for atom in residue.get_atoms()]), axis=0)


def is_valid_ligand(ligand_residue):
    # no ions
    invalids = ['HOH', 'NCO', 'SO4', 'EPE', 'OHX', 'MPD']
    if ligand_residue.resname in invalids or len(ligand_residue.resname) != 3:
        return False
    return True


def _is_valid_ligand(ligand_residue):
    # no ions
    return ligand_residue.resname.strip() == 'MG'


def pocket_res_count(struct, ligand_residue, radius=12):
    """
        Isolate residues around a ligand for a range of radii and counts
        the number of RNA residues and Protein residues in the radius.
        
        Arguments:
            struct (Bio.PDB.Structure): BioPython object for full structure.
            ligand_residue (Bio.PDB.Residue): BioPython residue object for ligand.
            radius (int): Largest radius around ligand to accept.

        Returns:
            pocket_rescount (list): list of protein/rna residue counts for each cutoff.
    """
    # get all residues within cutoff
    kd = NeighborSearch(list(struct.get_atoms()))
    center = ligand_center(ligand_residue)

    # list of rna and protein concentrations in region (cutoff, rna_res, protein_res)
    pocket_rescount = []
    for c in range(1, radius):
        pocket = kd.search(ligand_center(ligand_residue), c, level='R')

        rna_res = []
        protein_res = []
        # if there is protein pocket is no bueno
        for res in pocket:
            res_name = res.resname.strip()
            info = f"{res.get_parent().id}:{res.resname.strip()}:{res.id[1]}"
            if res_name in ['A', 'U', 'C', 'G']:
                rna_res.append(info)
            elif res_name != ligand_residue.resname and res_name in amino_acids:
                protein_res.append(info)
            else:
                continue
        pocket_rescount.append({'cutoff': c, 'rna': len(rna_res), 'rna_res': rna_res,
                                'protein': len(protein_res),
                                'protein_res': protein_res})

    return pocket_rescount


def full_struc_analyse(strucpath):
    """
    Analyses a full PDB structure for its binding pockets.
    Counts RNA/Protein residue counts at all binding sites.

    Arguments:
        strucpath (str): Path to PDB structure.
    Returns:
        valid_ligs_info (list): List of pocket info for each ligand.
                                `[(lig_res_id, {'rna_count' int, 'prot_count', int, 'cutoff', int})]`

    """
    try:
        # load mmCIF structure
        struc_dict = MMCIF2Dict.MMCIF2Dict(strucpath)
        # load PDB
        parser = MMCIFParser(QUIET=True)
        pdbstruc = parser.get_structure("", strucpath)
    except Exception as e:
        print("Structure loading error {e}")
        return None

    ligand_dict = {}
    try:
        ligand_dict['position'] = struc_dict['_pdbx_nonpoly_scheme.pdb_seq_num']
        ligand_dict['res_name'] = struc_dict['_pdbx_nonpoly_scheme.mon_id']
        ligand_dict['chain'] = struc_dict['_pdbx_nonpoly_scheme.pdb_strand_id']
        ligand_dict['unique_id'] = struc_dict['_pdbx_nonpoly_scheme.asym_id']

    except:
        print("Ligand not detected.")
        return None

    num_models = len(pdbstruc)
    model = pdbstruc[0]
    try:
        ligand_df = pd.DataFrame.from_dict(ligand_dict)
    # pandas complains when dictionary values are not lists
    # this happens when there is only one ligand in PDB
    except ValueError:
        ligand_df = pd.DataFrame(ligand_dict, index=[0])

    ligand_df['position'] = pd.to_numeric(ligand_df['position'])
    pdbid = os.path.basename(strucpath).split(".")[0]
    ligand_df['pdbid'] = pdbid

    # check ligands
    invalid_ligands = 0
    valid_ligands = []
    for ligand in ligand_df.itertuples():
        ligand_res = None
        # find the residue corresponding to ligand
        for res in model[ligand.chain].get_residues():
            if res.id[1] == ligand.position:
                ligand_res = res
                if is_valid_ligand(ligand_res):
                    valid_ligands.append(
                        (ligand_res, f"#{'0' if num_models == 1 else '0.1'}", pocket_res_count(model, ligand_res)))
                else:
                    invalid_ligands += 1
    return valid_ligands, strucpath


def binding_wrapper(strucpath):
    try:
        return full_struc_analyse(strucpath)
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        print(f"Failed on {strucpath} :: {e}")
        return None


def process_all(pdb_path, dump_path, restart=None, parallel=False):
    """
        Main function for building full lig_dict.

        Arguments:
            pdb_path (str): path to PDBs
            dump_path (str): file path to write lig_dict.
            restart (str): path to dictionary to start from.
            parallel (bool): If True runs with multiprocessing
    """
    # PDB_PATH = os.path.join("..", "data", "all_rna_prot_lig_2019")
    pdbs = [os.path.join(pdb_path, p) for p in os.listdir(pdb_path)]
    num_pdbs = len(pdbs)
    done = []
    if not restart is None:
        try:
            r = pickle.load(open(restart, 'rb'))
            done = list(r.keys())
        except FileNotFoundError:
            print(f"Could not load dict {restart}")

    lig_dict = {}

    todo = iter([p for p in pdbs if p not in done])
    num_done = 0

    pool = multiprocessing.Pool(processes=20)
    for result in pool.imap_unordered(full_struc_analyse, todo):
        # for result in map(binding_wrapper, todo):
        if result is None:
            print("failed")
            continue
        ligs, p = result
        num_done += 1
        print(f"done {p}, {num_done} of {num_pdbs}")
        lig_ids = []
        for l, model_id, radii in ligs:
            info = f"{model_id}:{l.get_parent().id}:{l.resname}:{l.id[1]}"
            lig_ids.append((info, radii))
        lig_dict[os.path.basename(p)] = lig_ids
        pickle.dump(lig_dict, open(dump_path, 'wb'))
    pass


if __name__ == "__main__":
    # pdb_path = os.path.join("..", '..', 'carlos_dock', "data", "all_rna_with_lig_2019")
    pdb_path = '/home/mcb/users/jcarva4/rnamigos2/rnamigos1/rnamigos_gcn/data/cif_files'
    process_all(pdb_path, "../data/binding_pocket_analyse.p")
