import os, sys
import pickle

import networkx as nx
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    sys.path.append("..")

from post.rna_layout import circular_layout



params= {'text.latex.preamble' : [r'\usepackage{fdsymbol}\usepackage{xspace}']}
plt.rc('font', family='serif')
plt.rcParams.update(params)

labels = {
    'CW': r"$\medblackcircle$\xspace",
    'CS': r"$\medblacktriangleright$\xspace",
    'CH': r"$\medblacksquare$\xspace",
    'TW': r"$\medcircle$\xspace",
    'TS': r"$\medtriangleright$\xspace",
    'TH': r"$\medsquare$\xspace"
}


make_label = lambda s: labels[s[:2]] + labels[s[0::2]] if len(set(s[1:])) == 2\
    else labels[s[:2]]

def rna_draw(nx_g, title="", highlight_edges=None, node_colors=None, num_clusters=None):
    # pos = circular_layout(nx_g)
    pos = nx.spring_layout(nx_g)

    nodes = nx.draw_networkx_nodes(nx_g, pos, node_size=150,  node_color='grey', linewidths=2)

    nodes.set_edgecolor('black')

    # plt.title(r"{0}".format(title))
    edge_labels = {}
    for n1,n2,d in nx_g.edges(data=True):
        try:
            symbol = make_label(d['label'])
            edge_labels[(n1, n2)] = symbol
        except:
            if d['label'] == 'B53':
                edge_labels[(n1, n2)] = ''
            else:
                edge_labels[(n1, n2)] = r"{0}".format(d['label'])
            continue

    non_bb_edges = [(n1,n2) for n1,n2,d in nx_g.edges(data=True) if d['label'] != 'B53']
    bb_edges = [(n1,n2) for n1,n2,d in nx_g.edges(data=True) if d['label'] == 'B53']

    nx.draw_networkx_edges(nx_g, pos, edgelist=non_bb_edges)
    nx.draw_networkx_edges(nx_g, pos, edgelist=bb_edges, width=2)

    if not highlight_edges is None:
        nx.draw_networkx_edges(nx_g, pos, edgelist=highlight_edges, edge_color='y', width=8, alpha=0.5)

    nx.draw_networkx_edge_labels(nx_g, pos, font_size=16,
                                 edge_labels=edge_labels)
    plt.axis('off')
    # plt.savefig('fmn_' + title + '.png', format='png')
    # plt.clf()
    plt.show()

def ablation_draw():
    g_name = "1fmn_#0.1:A:FMN:36.nx_annot.p"
    modes = ['', '_bb-only', '_wc-bb', '_wc-bb-nc', '_no-label', '_label-shuffle']
    for m in modes:
        g_dir = "../data/annotated/pockets_nx" + m
        g,_,_,_ = pickle.load(open(os.path.join(g_dir, g_name), 'rb'))
        rna_draw(g, title=m)
    pass

if __name__ == "__main__":
    ablation_draw()
