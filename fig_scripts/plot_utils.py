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
    _, _, _, test_names_grouped = pickle.load(open(splits_file, 'rb'))
    group_reps = [random.choice(names) for key, names in test_names_grouped.items()]
    group_reps_path = os.path.join(script_dir, '../data/group_reps_75.p')
    pickle.dump(group_reps, open(group_reps_path, 'wb'))


def group_df(df):
    """
    Subset rows of a df to only keep one representative for each pocket.
    """
    script_dir = os.path.dirname(__file__)
    splits_file = os.path.join(script_dir, '../data/group_reps_75.p')
    group_reps = pickle.load(open(splits_file, 'rb'))
    df = df.loc[df['pocket_id'].isin(group_reps)]
    return df


def get_rmscores():
    rm_scores = pd.read_csv("data/rmscores.csv", index_col=0)
    return rm_scores


def get_smooth_order(rmscores):
    distance_symmetrized = (1 - rmscores + (1 - rmscores).T)/2
    # sim_error = distance_symmetrized - (distance_symmetrized).T
    new_coords = TSNE(n_components=1, learning_rate='auto', metric='precomputed', init='random').fit_transform(
        distance_symmetrized)
    # new_coords = MDS(n_components=1, dissimilarity='precomputed').fit_transform(distance_symmetrized)
    return np.argsort(new_coords.flatten())


def setup_plot():
    # SETUP PLOT
    plt.rcParams['text.usetex'] = True
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    plt.rc('font', size=16)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=13)  # fontsize of the tick labels
    plt.rc('xtick', labelsize=13)  # fontsize of the tick labels
    plt.rc('grid', color='grey', alpha=0.2)

    # raw_hex = ["#61C6E7", "#4F7BF0", "#6183E7", "#FA4828"]
    raw_hex = ["#3180e0", "#2ba9ff", "#2957d8", "#FA4828", "#0a14db", "#803b96"]
    hex_plt = [f"{raw}" for raw in raw_hex]
    palette = sns.color_palette(hex_plt)
    return palette


#PALETTE = setup_plot()


class CustomScale(mscale.ScaleBase):
    name = 'custom'

    def __init__(self, axis, offset=0.03, sup_lim=1):
        mscale.ScaleBase.__init__(self, axis=axis)
        self.offset = offset
        self.sup_lim = sup_lim
        self.thresh = None

    def get_transform(self):
        return self.CustomTransform(thresh=self.thresh, offset=self.offset, sup_lim=self.sup_lim)

    def set_default_locators_and_formatters(self, axis):
        pass

    class CustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, offset, thresh, sup_lim):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
            self.offset = offset
            self.sup_lim = sup_lim

        def transform_non_affine(self, a):
            return - np.log(self.sup_lim + self.offset - a)

        def inverted(self):
            return CustomScale.InvertedCustomTransform(thresh=self.thresh, offset=self.offset, sup_lim=self.sup_lim)

    class InvertedCustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, offset, thresh, sup_lim):
            mtransforms.Transform.__init__(self)
            self.offset = offset
            self.thresh = thresh
            self.sup_lim = sup_lim

        def transform_non_affine(self, a):
            return self.sup_lim - np.exp(-a) + self.offset

        def inverted(self):
            return CustomScale.CustomTransform(offset=self.offset, thresh=self.thresh, sup_lim=self.sup_lim)


# X = type('CustomScale', (object,), dict(offset=0.01, name = 'custom'))

mscale.register_scale(CustomScale)
