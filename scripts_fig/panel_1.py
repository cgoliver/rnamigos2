import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn.palettes import dark_palette, light_palette, blend_palette

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts_fig.plot_utils import group_df, get_smooth_order


def custom_diverging_palette(
    h_neg, h_pos, s_neg=75, s_pos=75, l_neg=50, l_pos=50, sep=1, n=6, center="light", as_cmap=False  # noqa
):
    """
    Make a diverging palette between two HUSL colors.I added the possibility of asymetry in endpoints saturation
    """
    palfunc = dict(dark=dark_palette, light=light_palette)[center]
    n_half = int(128 - (sep // 2))
    neg = palfunc((h_neg, s_neg, l_neg), n_half, reverse=True, input="husl")
    pos = palfunc((h_pos, s_pos, l_pos), n_half, input="husl")
    midpoint = dict(light=[(0.95, 0.95, 0.95)], dark=[(0.133, 0.133, 0.133)])[center]
    mid = midpoint * sep
    pal = blend_palette(np.concatenate([neg, mid, pos]), n, as_cmap=as_cmap)
    return pal


def barcodes(transposed=False, decoy_mode="chembl"):
    name_runs = {
        "COMPAT": "native_42.csv",
        "AFF": "dock_42.csv",
        "rDock": "rdock.csv",
        "MIXED": "docknat_42.csv",
    }

    rows = []
    prev_pockets = None
    for csv_name in name_runs.values():
        df = pd.read_csv(f"outputs/pockets/{csv_name}")
        df = group_df(df)
        row = df[df["decoys"] == decoy_mode].sort_values(by="pocket_id")
        all_pockets = row["pocket_id"].values
        if prev_pockets is None:
            prev_pockets = all_pockets
        else:
            assert (prev_pockets == all_pockets).all(), print(prev_pockets, all_pockets)
        rows.append(row["score"])

    # FIND SMOOTHER PERMUTED VERSION
    # smooth = True
    smooth = False
    if smooth:
        order = get_smooth_order(prev_pockets)
        for i in range(len(rows)):
            new_row = rows[i].values[order]
            rows[i] = new_row

    n_over = 15
    # sns.heatmap(rows, cmap='binary_r')
    # cmap = sns.color_palette("vlag_r", as_cmap=True)
    # cmap = sns.diverging_palette(0, 245, s=100, l=50, as_cmap=True)
    # cmap = custom_diverging_palette(0, 245, s_neg=100, l_neg=50, s_pos=90, l_pos=80, as_cmap=True)
    # blue_pal = sns.light_palette('#5c67ff', n_colors=30)[:10] # too grey/violet
    # blue_pal = sns.light_palette('#9dabe1', n_colors=10) # a bit violet and also lot of color
    # blue_pal = sns.light_palette('#a5b0d9', n_colors=10) # close
    # blue_pal = sns.light_palette('#7689d5', n_colors=10) # nice blue but a bit dense
    # blue_pal = sns.light_palette('#ccd6ff', n_colors=10) # brighter less blue
    # blue_pal = sns.light_palette('#d6ecff', n_colors=10) # almost white
    # blue_pal = sns.light_palette('#ebf5ff', n_colors=10) # whiter
    # blue_pal = sns.color_palette("light:b", n_colors=10) # hardcode blue
    blue_pal = sns.light_palette("#fff", n_colors=n_over)  # white
    red_pal = sns.light_palette("#CF403E", reverse=True, n_colors=128 - n_over)
    cmap = blend_palette(np.concatenate([red_pal, blue_pal]), 1, as_cmap=True)

    if not transposed:
        ax = sns.heatmap(rows, cmap=cmap)
    else:
        ax = sns.heatmap(np.array(rows).T, cmap=cmap)

    # Handle spine
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("grey")

    # Handle ticks
    if not transposed:
        xticks = np.arange(0, len(rows[0]), 10)
        xticks_labels = xticks + 1
        plt.xticks(xticks, xticks_labels, va="center")
        plt.tick_params(axis="x", bottom=False, labelbottom=True)
        plt.yticks(np.arange(len(name_runs)) + 0.5, [name for name in name_runs.keys()], rotation=0, va="center")
        plt.tick_params(axis="y", left=False, right=False, labelleft=True)
    else:
        plt.xticks(np.arange(len(name_runs)) + 0.5, [name for name in name_runs.keys()], rotation=0, va="center")
        plt.tick_params(axis="x", bottom=False, labelbottom=True)
        yticks = np.arange(0, len(rows[0]), 10)
        yticks_labels = yticks + 1
        # In heatmaps, id0 is at the top, it looks weird
        yticks_positions = len(rows[0]) - yticks - 1
        plt.yticks(yticks_positions, yticks_labels, va="center")
        plt.tick_params(axis="y", left=False, labelleft=True)

    if not transposed:
        plt.xlabel(r"Pocket")
        plt.ylabel(r"Method")
    else:
        plt.xlabel(r"Method")
        plt.ylabel(r"Pocket")

    fig_name = f"figs/barcode{'_transposed' if transposed else ''}{'_chembl' if decoy_mode == 'chembl' else ''}.pdf"
    plt.savefig(fig_name, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    transposed = False
    # transposed = True
    # decoy_mode = 'pdb_chembl'
    decoy_mode = "chembl"
    barcodes(transposed=transposed, decoy_mode=decoy_mode)
