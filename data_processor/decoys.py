"""
    Functions for generated decoys.
"""
import os
import pickle

def smiles_2_text(smiles_dict):
    keep = {s.split(":")[2] for s in os.listdir("../data/annotated/pockets_nx")}
    with open("decoys_plz_train.txt", "w") as sm:
        for _, v in smiles_dict.items():
            s = v.split()
            if len(s) > 1:
                smile, name = s
                if name in keep:
                    sm.write(smile + '\n')
    pass

if __name__ == "__main__":
    smiles_dict = pickle.load(open("../data/smiles_ligs_dict.p", "rb"))
    smiles_2_text(smiles_dict)
    pass
