import os
import sys
import pickle
"""
if __name__ == '__main__':
    sys.path.append('.')
    predict()
"""

#from tools.learning_utils import inference_on_dir
#graph_dir = "data/pockets_to_predict/pockets_nx_symmetric_orig"


def predict(model_name, pred_fp_name):
    pf_dict = {}
    for folder in os.scandir(graph_dir):
        fp_pred, _ = inference_on_dir(model_name, graph_dir + '/' + folder.name, get_sim_mat=True)
        pf_dict[folder.name] = fp_pred

    pickle.dump(pf_dict, open('models/' + pred_fp_name , 'wb'))
    print(len(pf_dict))


if __name__ == '__main__':
    sys.path.append('.')
    from tools.learning_utils import inference_on_dir
    #graph_dir = "data/pockets_to_predict/pockets_nx_symmetric_orig"
    graph_dir = "data/pockets_to_predict/pockets_nx_annot_fp_predictions"
    #graph_dir = "data/annotated/pockets_nx_symmetric_orig_emb_one_hot"
    mn = sys.argv[1]
    fpn = sys.argv[2]
    predict(mn, fpn)
