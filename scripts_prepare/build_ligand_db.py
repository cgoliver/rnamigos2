"""
Build the pocket decoy sets used for virtual screening, into data/ligand_db.

    python scripts_prepare/build_ligand_db.py

Writes data/ligand_db/<pocket>/{pdb,pdb_chembl,chembl}/{actives,decoys}.txt

This supersedes the `--pdb` half of build_screen_data.py, which never wrote the
`chembl` set even though that is the one the paper reports on. The ROBIN and
DecoyFinder sets still come from build_screen_data.py.

The three sets are not nested the way the names suggest:

    pdb         every PDB-derived ligand in the dataset      (264, global)
    chembl      the ChEMBL compounds docked against *this*   (~500, per pocket)
                pocket -- drug-like, diversity-picked
    pdb_chembl  pdb + every ChEMBL ligand in the dataset     (6333, global)

so pdb_chembl is not pdb + chembl, and chembl cannot be recovered by
subtracting one shipped set from the other.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

DECOY_MODES = ("pdb", "pdb_chembl", "chembl")


def build_ligand_db(
    pdb_data_path="data/rnamigos2_dataset_consolidated.csv",
    save_path="data/ligand_db",
    decoy_modes=DECOY_MODES,
):
    df = pd.read_csv(pdb_data_path, usecols=["PDB_ID_POCKET", "LIGAND_SMILES", "LIGAND_SOURCE", "IS_NATIVE"])

    pdb_ligands = set(df.loc[df["LIGAND_SOURCE"] == "PDB", "LIGAND_SMILES"].unique())
    chembl_ligands = set(df.loc[df["LIGAND_SOURCE"] == "CHEMBL", "LIGAND_SMILES"].unique())
    # The ChEMBL compounds docked against each individual pocket, which is what
    # the `chembl` set is; the global pool above is only used for pdb_chembl.
    chembl_per_pocket = (
        df.loc[df["LIGAND_SOURCE"] == "CHEMBL"].groupby("PDB_ID_POCKET")["LIGAND_SMILES"].apply(set).to_dict()
    )
    natives = df.loc[df["IS_NATIVE"] == "YES"]

    counts = {mode: 0 for mode in decoy_modes}
    empty = []
    for pocket in tqdm(natives.itertuples(), total=len(natives)):
        pocket_id, native = pocket.PDB_ID_POCKET, pocket.LIGAND_SMILES
        # Drop the native by whole SMILES. build_screen_data.py writes
        # `- set(pocket.LIGAND_SMILES)`, which is a set of characters and so
        # leaves every native sitting in its own decoys.txt.
        pools = {
            "pdb": pdb_ligands - {native},
            "pdb_chembl": (pdb_ligands | chembl_ligands) - {native},
            "chembl": chembl_per_pocket.get(pocket_id, set()) - {native},
        }
        for mode in decoy_modes:
            decoys = pools[mode]
            if not decoys:
                empty.append((pocket_id, mode))
                continue
            out_dir = Path(save_path, pocket_id, mode)
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "actives.txt", "w") as ac:
                ac.write(native)
            with open(out_dir / "decoys.txt", "w") as de:
                de.write("\n".join(sorted(decoys)))
            counts[mode] += 1

    for mode in decoy_modes:
        print(f"{mode:11s} written for {counts[mode]} pockets")
    if empty:
        print(f"no decoys available for {len(empty)} pocket/mode pairs, e.g. {empty[:5]}")
    return counts


def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_data_path", default="data/rnamigos2_dataset_consolidated.csv")
    parser.add_argument("--save_path", default="data/ligand_db")
    parser.add_argument("--decoy_modes", nargs="+", default=list(DECOY_MODES), choices=list(DECOY_MODES))
    return parser.parse_args()


if __name__ == "__main__":
    args = cline()
    build_ligand_db(
        pdb_data_path=args.pdb_data_path,
        save_path=args.save_path,
        decoy_modes=tuple(args.decoy_modes),
    )
