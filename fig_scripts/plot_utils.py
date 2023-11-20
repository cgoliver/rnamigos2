import matplotlib.ticker as ticker
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
import matplotlib.pyplot as plt
import matplotlib.pyplot
import numpy as np
import seaborn as sns

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
PALETTE = sns.color_palette(hex_plt)


class CustomScale(mscale.ScaleBase):
    name = 'custom'

    def __init__(self, axis):
        mscale.ScaleBase.__init__(self, axis=axis)
        self.offset = 0.01
        self.thresh = None

    def get_transform(self):
        return self.CustomTransform(thresh=self.thresh, offset=self.offset)

    def set_default_locators_and_formatters(self, axis):
        pass

    class CustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, offset, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
            self.offset = offset

        def transform_non_affine(self, a):
            return - np.log(1 + self.offset - a)

        def inverted(self):
            return CustomScale.InvertedCustomTransform(thresh=self.thresh, offset=self.offset)

    class InvertedCustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, offset, thresh):
            mtransforms.Transform.__init__(self)
            self.offset = offset
            self.thresh = thresh

        def transform_non_affine(self, a):
            return 1 - np.exp(-a) + self.offset

        def inverted(self):
            return CustomScale.CustomTransform(offset=self.offset, thresh=self.thresh)


mscale.register_scale(CustomScale)
