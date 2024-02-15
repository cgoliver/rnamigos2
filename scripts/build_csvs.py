import os
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

interactions_csv_original = 'data/rnamigos2_dataset_consolidated.csv'
interactions_csv_migos1 = 'data/rnamigos_1_data/rnamigos1_dataset.csv'
os.makedirs('data/csvs', exist_ok=True)

interactions_csv_dock = 'data/csvs/docking_data.csv'
interactions_csv_fp = 'data/csvs/fp_data.csv'
interactions_csv_binary = 'data/csvs/binary_data.csv'

systems = pd.read_csv(interactions_csv_original)
systems = systems.rename({'TYPE': 'SPLIT'}, axis='columns')

# FP : Get PDB, SMILES
natives = systems.loc[systems['IS_NATIVE'] == 'YES']
systems_fp = natives[['PDB_ID_POCKET', 'LIGAND_SMILES', 'SPLIT']]
systems_fp.to_csv(interactions_csv_fp)


# DOCK : Get PDB, SMILES, SCORE
def flatten_values(systems):
    """
    This performs a per-system quantile normalization
    :param systems:
    :return:
    """
    all_pockets = set(systems['PDB_ID_POCKET'].unique())
    all_values = list()
    for i, pocket in enumerate(all_pockets):
        print(f'Processing pocket {i}/{len(all_pockets)}')
        pocket_values = systems.loc[systems['PDB_ID_POCKET'] == pocket][['INTER']]
        qt = QuantileTransformer(random_state=0, n_quantiles=50)
        transformed_values = qt.fit_transform(pocket_values)
        # This is not equivalent to directly using transformed values since we keep the index
        pocket_values['normalized_values'] = transformed_values
        pocket_values = pocket_values[["normalized_values"]]
        all_values.append(pocket_values)
    all_new_values = pd.concat(all_values)
    # Since this concatenation takes the index into account, order is preserved.
    new_systems = pd.concat((systems, all_new_values), axis=1)
    return new_systems


systems_dock = systems[['PDB_ID_POCKET', 'LIGAND_SMILES', 'INTER', 'SPLIT']]
systems_dock_quantiles = flatten_values(systems_dock)
systems_dock_quantiles.to_csv(interactions_csv_dock)

# IS NATIVE Get PDB, SMILES, 0/1
systems_binary = systems[['PDB_ID_POCKET', 'LIGAND_SMILES', 'IS_NATIVE', 'SPLIT']]
systems_binary['IS_NATIVE'] = systems_binary['IS_NATIVE'].apply(lambda x: 1 if x == 'YES' else 0)
systems_binary.to_csv(interactions_csv_binary)
