import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# FIRST GET PDB/CHEMBL MAPPER
in_df = 'data/csvs/binary_data.csv'
in_df = pd.read_csv(in_df)
df_1 = in_df[['LIGAND_SMILES', 'LIGAND_SOURCE']]
pdb_ligs = set(df_1.loc[df_1['LIGAND_SOURCE'] == 'PDB']["LIGAND_SMILES"].unique())
chembl_ligs = set(df_1.loc[df_1['LIGAND_SOURCE'] == 'CHEMBL']["LIGAND_SMILES"].unique())
a = chembl_ligs.intersection(pdb_ligs)

# THEN SPLIT PERF
# out_df = "outputs/pockets/native_pre_rnafm_raw.csv"
# out_df = "outputs/pockets/dock_rnafm_raw.csv"
out_df = "outputs/pockets/rdock_raw.csv"
out_df = "outputs/native_rnafm_onpdb_raw.csv"
out_df = pd.read_csv(out_df)
out_df = out_df.loc[out_df['decoys'] == 'pdb_chembl']
pdb_scores = out_df.loc[out_df['smiles'].isin(pdb_ligs)]['raw_score'].values
chembl_scores = out_df.loc[out_df['smiles'].isin(chembl_ligs)]['raw_score'].values

# PLOT
df_pdb = pd.DataFrame({"score": pdb_scores, "decoy": "PDB"})
df_chembl = pd.DataFrame({"score": chembl_scores, "decoy": "chembl"})
df = pd.concat([df_pdb, df_chembl])
sns.histplot(data=df, x="score", hue="decoy")
plt.show()

# df_2 = df_2[['pocket_id', 'smiles', 'is_active', renamed_score]]
# df_to_use = df_1.merge(df_2, on=['pocket_id', 'smiles', 'is_active'], how='outer')
a = 1
