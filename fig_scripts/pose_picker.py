import pickle
import pandas as pd
from rdkit import Chem

def main():
    """
    pick 3 ligands for each pocket: random from 10th percentile, random from 90th percentile, native
    """
    df = pd.read_csv("outputs/mixed_raw.csv")
    df_original = pd.read_csv("data/csvs/rnamigos2_dataset_consolidated.csv")

    names_train, names_test, grouped_train, grouped_test = pickle.load(open("data/train_test_75.p", 'rb'))
    print(grouped_test.keys())
    df = df.loc[(df['decoys'] == 'chembl') & (df['pocket_id'].isin(grouped_test)) ]
    for pocket_id, group in df.groupby('pocket_id'):
        try:
            sdf_chembl = Chem.SDMolSupplier(f'outputs/TEST_SET_DOCKING/{pocket_id}/1poseperlig.sdf')
            sdf_pdb = Chem.SDMolSupplier(f'outputs/TEST_SET_DOCKING/{pocket_id}/1poseperlig_final.sdf')
        except OSError:
            print(pocket_id, "fail")
            continue
        group = group.sort_values(by='combined')
        group['rank'] = group['combined'].rank(pct=True, ascending=True)
        samps = group.loc[group['rank'] > 0.9].sample(3)
        native = group.loc[group['is_active'] == 1.0].iloc[0]
        native_id = pocket_id.split("_")[2]
        for samp in samps.itertuples():
            dock_info = df_original.loc[(df_original['LIGAND_SMILES'] == samp.smiles) & (df_original['PDB_ID_POCKET'] == pocket_id)].iloc[0]
            pass

            for mol in sdf_chembl:
                if not mol is None:
                    lig_info = mol.GetPropsAsDict()
                    print(lig_info)
                    if lig_info['Name'] == dock_info['LIGAND_ID']:
                        # write the sdf to its own file
                        writer = Chem.SDWriter(f'outputs/TEST_SET_DOCKING/{pocket_id}/{pocket_id}_{dock_info["LIGAND_ID"]}.sdf')
                        writer.write(mol)
                        break
                    info_df = pd.DataFrame({"lig_id": [dock_info["LIGAND_ID"]], "score": [samp.combined], "smiles": [samp.smiles], "inter": [dock_info["INTER"]], 'rank': [samp.rank]})
                    info_df.to_csv(f"outputs/TEST_SET_DOCKING/{pocket_id}/{pocket_id}_{dock_info['LIGAND_ID']}.csv")
        for mol in sdf_pdb:
            if not mol is None:
                lig_info = mol.GetPropsAsDict()
                if lig_info['Name'] == native_id:
                    # write the sdf to its own file
                    writer = Chem.SDWriter(f'outputs/TEST_SET_DOCKING/{pocket_id}/{pocket_id}_{native_id}_native.sdf')
                    writer.write(mol)

                    info_df = pd.DataFrame({"lig_id": [native_id], "score": [native['combined']], "smiles": [native['smiles']], "inter": [dock_info["INTER"]], 'rank': [native['rank']]})
                    info_df.to_csv(f"outputs/TEST_SET_DOCKING/{pocket_id}/{pocket_id}_{dock_info['LIGAND_ID']}.csv")
                    break


        pass





if __name__ == "__main__":
    main()
    pass
