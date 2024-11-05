from joblib import delayed, Parallel
import itertools
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

ROBIN_POCKETS = {
    "TPP": "2GDI_Y_TPP_100",
    "ZTP": "5BTP_A_AMZ_106",
    "SAM_ll": "2QWY_B_SAM_300",
    "PreQ1": "3FU2_A_PRF_101",
}

RES_DIR = "outputs/robin/"
if __name__ == "__main__":
    df = pd.read_csv(f"{RES_DIR}/big_df_raw.csv")
    scores = ["rdock", "dock_42", "native_42", "rnamigos_42", "combined_42"]
    pass

    all_smiles = list(df["smiles"].unique())

    mols = []
    keep_smiles = []
    for s in all_smiles:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                mols.append(mol)
                keep_smiles.append(s)
        except:
            print(f"failed on {s}")

    fps = [MACCSkeys.GenMACCSKeys(m) for m in mols[:10]]

    def compute_tanimoto(fp_1, fp_2):
        return DataStructs.TanimotoSimilarity(fp_1, fp_2)

    # Step 2: Create pairs of fingerprints
    pairs = itertools.combinations(fps, 2)

    # Step 3: Parallelize the computation of Tanimoto similarities
    tani = Parallel(n_jobs=-1)(delayed(compute_tanimoto)(fp_1, fp_2) for fp_1, fp_2 in pairs)

    square_tani = squareform(tani)
    print(square_tani)
    np.save("robin_tanimotos.npy", square_tani)
    with open("robin_smileslist_tani.txt", "w") as ro:
        for sm in keep_smiles:
            ro.write(sm + "\n")
