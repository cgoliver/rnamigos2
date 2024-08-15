import matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
import numpy as np
import os
import pandas as pd
import pickle
import random
import seaborn as sns
from sklearn.manifold import TSNE, MDS

random.seed(42)
np.random.seed(42)


def get_groups():
    script_dir = os.path.dirname(__file__)
    splits_file = os.path.join(script_dir, '../data/train_test_75.p')
    _, _, train_names_grouped, test_names_grouped = pickle.load(open(splits_file, 'rb'))
    train_group_reps = [random.choice(names) for key, names in train_names_grouped.items()]
    test_group_reps = [random.choice(names) for key, names in test_names_grouped.items()]
    group_reps_path = os.path.join(script_dir, '../data/group_reps_75.p')
    pickle.dump((train_group_reps, test_group_reps), open(group_reps_path, 'wb'))


if __name__ == '__main__':
    get_groups()


def group_df(df):
    """
    Subset rows of a df to only keep one representative for each pocket.
    """
    script_dir = os.path.dirname(__file__)
    splits_file = os.path.join(script_dir, '../data/group_reps_75.p')
    try:
        train_group_reps, test_group_reps = pickle.load(open(splits_file, 'rb'))
    except FileNotFoundError as e:
        raise Exception("To produce this missing file run python scripts_fig/plot_utils.py") from e
    df = df.loc[df['pocket_id'].isin(train_group_reps + test_group_reps)]
    return df


def get_rmscores():
    rm_scores = pd.read_csv("data/rmscores.csv", index_col=0)
    return rm_scores


def get_smooth_order(pockets, rmscores=None):
    """
    Given pockets in a certain order, return a permutation so that similar pockets are close in the permuted list
    """
    if rmscores is None:
        rmscores = get_rmscores()
    rmscores_values = rmscores.values
    rmscores_labels = rmscores.columns

    # Subset queried pockets, this is needed when we have only a few pockets (otherwise indices are over len(pockets))
    selected_pockets = set(pockets)
    test_index = np.array([name in selected_pockets for name in rmscores_labels])
    test_rmscores_labels = rmscores_labels[test_index]
    test_rmscores_values = rmscores_values[test_index][:, test_index]

    # Order following query pockets order
    pocket_to_id = {pocket: i for i, pocket in enumerate(test_rmscores_labels)}
    sorter = [pocket_to_id[pocket] for pocket in pockets]
    test_rmscores_values = test_rmscores_values[sorter][:, sorter]

    # Use the values to re-order the pockets
    distance_symmetrized = (1 - test_rmscores_values + (1 - test_rmscores_values).T) / 2
    # sim_error = distance_symmetrized - (distance_symmetrized).T
    new_coords = TSNE(n_components=1, learning_rate='auto', metric='precomputed', init='random').fit_transform(
        distance_symmetrized)
    # new_coords = MDS(n_components=1, dissimilarity='precomputed').fit_transform(distance_symmetrized)
    return np.argsort(new_coords.flatten())


def rotate_2D_coords(coords, angle=0):
    """
    coords are expected to be (N,2)
    """
    theta = (angle / 180.) * np.pi
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    center = coords.mean(axis=0)
    return (coords - center) @ rot_matrix + center


def setup_plot():
    # SETUP PLOT
    plt.rcParams['text.usetex'] = False
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    # matplotlib.rcParams['font.family'] = 'Helvetica'
    plt.rc('font', size=16)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=13)  # fontsize of the tick labels
    plt.rc('xtick', labelsize=13)  # fontsize of the tick labels
    plt.rc('grid', color='grey', alpha=0.2)

    # raw_hex = ["#61C6E7", "#4F7BF0", "#6183E7", "#FA4828"]
    palette_dict = {'fp': "#3180e0",
                    'native': "#2ba9ff",
                    'dock': "#2957d8",
                    'rdock': "#FA4828",
                    'mixed': "#0a14db",
                    'mixed+rdock': "#803b96"}
    # palette = ["#2ba9ff", "#2957d8", "#FA4828", "#0a14db", "#803b96"]
    # palette = [f"{raw}" for raw in raw_hex]
    # palette = sns.color_palette(palette)
    return palette_dict


PALETTE_DICT = setup_plot()


class CustomScale(mscale.ScaleBase):
    name = 'custom'

    def __init__(self, axis, offset=0.01, sup_lim=1, divider=1):
        mscale.ScaleBase.__init__(self, axis=axis)
        self.offset = offset
        self.divider = divider
        self.sup_lim = sup_lim
        self.thresh = None

    def get_transform(self):
        return self.CustomTransform(thresh=self.thresh,
                                    offset=self.offset,
                                    sup_lim=self.sup_lim,
                                    divider=self.divider)

    def set_default_locators_and_formatters(self, axis):
        pass

    class CustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, offset, thresh, sup_lim, divider):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
            self.offset = offset
            self.sup_lim = sup_lim
            self.divider = divider

        def transform_non_affine(self, a):
            return - np.log((self.sup_lim + self.offset - a) / self.divider)

        def inverted(self):
            return CustomScale.InvertedCustomTransform(thresh=self.thresh,
                                                       offset=self.offset,
                                                       sup_lim=self.sup_lim,
                                                       divider=self.divider)

    class InvertedCustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, offset, thresh, sup_lim, divider):
            mtransforms.Transform.__init__(self)
            self.offset = offset
            self.thresh = thresh
            self.sup_lim = sup_lim
            self.divider = divider

        def transform_non_affine(self, a):
            return self.sup_lim - np.exp(-a * self.divider) + self.offset

        def inverted(self):
            return CustomScale.CustomTransform(offset=self.offset,
                                               thresh=self.thresh,
                                               sup_lim=self.sup_lim,
                                               divider=self.divider)


# X = type('CustomScale', (object,), dict(offset=0.01, name = 'custom'))

mscale.register_scale(CustomScale)
