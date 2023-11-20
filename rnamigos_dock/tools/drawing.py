import os, sys
import pickle

import networkx as nx
import matplotlib

matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    sys.path.append("..")

from tools.rna_layout import circular_layout

params = {'text.latex.preamble': [r'\usepackage{fdsymbol}\usepackage{xspace}']}
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

make_label = lambda s: labels[s[:2]] + labels[s[0::2]] if len(set(s[1:])) == 2 \
    else labels[s[:2]]


def rna_draw(nx_g, title="", highlight_edges=None, nt_info=False, node_colors=None, num_clusters=None):
    """
    Draw an RNA with the edge labels used by Leontis Westhof
    :param nx_g:
    :param title:
    :param highlight_edges:
    :param node_colors:
    :param num_clusters:
    :return:
    """
    # pos = circular_layout(nx_g)
    pos = nx.spring_layout(nx_g)

    if node_colors is None:
        nodes = nx.draw_networkx_nodes(nx_g, pos, node_size=150, node_color='white', linewidths=2)
    else:
        nodes = nx.draw_networkx_nodes(nx_g, pos, node_size=150, node_color=node_colors, linewidths=2)

    nodes.set_edgecolor('black')
    if nt_info:
        nx.draw_networkx_labels(nx_g, pos, font_color='black')

    # plt.title(r"{0}".format(title))
    edge_labels = {}
    for n1, n2, d in nx_g.edges(data=True):
        try:
            symbol = make_label(d['label'])
            edge_labels[(n1, n2)] = symbol
        except:
            if d['label'] == 'B53':
                edge_labels[(n1, n2)] = ''
            else:
                edge_labels[(n1, n2)] = r"{0}".format(d['label'])
            continue

    non_bb_edges = [(n1, n2) for n1, n2, d in nx_g.edges(data=True) if d['label'] != 'B53']
    bb_edges = [(n1, n2) for n1, n2, d in nx_g.edges(data=True) if d['label'] == 'B53']

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


def rna_draw_pair(graphs, estimated_value=None, highlight_edges=None, node_colors=None, num_clusters=None,
                  similarity=False,
                  true_value=None):
    fig, ax = plt.subplots(1, len(graphs), num=1)
    for i, g in enumerate(graphs):
        pos = nx.spring_layout(g)

        if not node_colors is None:
            nodes = nx.draw_networkx_nodes(g, pos, node_size=150, node_color=node_colors[i], linewidths=2, ax=ax[i])
        else:
            nodes = nx.draw_networkx_nodes(g, pos, node_size=150, node_color='grey', linewidths=2, ax=ax[i])

        nodes.set_edgecolor('black')

        # plt.title(r"{0}".format(title))
        edge_labels = {}
        for n1, n2, d in g.edges(data=True):
            try:
                symbol = make_label(d['label'])
                edge_labels[(n1, n2)] = symbol
            except:
                if d['label'] == 'B53':
                    edge_labels[(n1, n2)] = ''
                else:
                    edge_labels[(n1, n2)] = r"{0}".format(d['label'])
                continue

        non_bb_edges = [(n1, n2) for n1, n2, d in g.edges(data=True) if d['label'] != 'B53']
        bb_edges = [(n1, n2) for n1, n2, d in g.edges(data=True) if d['label'] == 'B53']

        nx.draw_networkx_edges(g, pos, edgelist=non_bb_edges, ax=ax[i])
        nx.draw_networkx_edges(g, pos, edgelist=bb_edges, width=2, ax=ax[i])

        if not highlight_edges is None:
            nx.draw_networkx_edges(g, pos, edgelist=highlight_edges, edge_color='y', width=8, alpha=0.5, ax=ax[i])

        nx.draw_networkx_edge_labels(g, pos, font_size=16,
                                     edge_labels=edge_labels, ax=ax[i])
        ax[i].set_axis_off()

    plt.axis('off')
    title = 'similarity : ' if similarity else 'distance : ' + str(estimated_value)
    if true_value is not None:
        title = title + f' true : {true_value}'

    plt.title(title)
    plt.show()


def generic_draw_pair(graphs, title="", highlight_edges=None, node_colors=None, num_clusters=None):
    fig, ax = plt.subplots(1, len(graphs), num=1)
    for i, g in enumerate(graphs):
        pos = nx.spring_layout(g)

        if not node_colors is None:
            nodes = nx.draw_networkx_nodes(g, pos, node_size=150, node_color=node_colors[i], linewidths=2, ax=ax[i])
        else:
            nodes = nx.draw_networkx_nodes(g, pos, node_size=150, node_color='grey', linewidths=2, ax=ax[i])

        nodes.set_edgecolor('black')

        # plt.title(r"{0}".format(title))
        edge_labels = {}
        for n1, n2, d in g.edges(data=True):
            edge_labels[(n1, n2)] = str(d['label'])

        if not highlight_edges is None:
            nx.draw_networkx_edges(g, pos, edgelist=highlight_edges, edge_color='y', width=8, alpha=0.5, ax=ax[i])

        nx.draw_networkx_edge_labels(g, pos, font_size=16,
                                     edge_labels=edge_labels, ax=ax[i])
        ax[i].set_axis_off()

    plt.axis('off')
    plt.title(f"distance {title}")
    plt.show()


def generic_draw(graph, title="", highlight_edges=None, node_colors=None):
    fig, ax = plt.subplots(1, 2, num=1)
    pos = nx.spring_layout(graph)

    if not node_colors is None:
        nodes = nx.draw_networkx_nodes(graph, pos, node_size=150, cmap=plt.cm.Blues, node_color=node_colors,
                                       linewidths=2, ax=ax[0])
    else:
        nodes = nx.draw_networkx_nodes(graph, pos, node_size=150, node_color='grey', linewidths=2, ax=ax[0])

    nodes.set_edgecolor('black')

    # plt.title(r"{0}".format(title))
    edge_labels = {}
    for n1, n2, d in graph.edges(data=True):
        edge_labels[(n1, n2)] = str(d['label'])

    if not highlight_edges is None:
        nx.draw_networkx_edges(graph, pos, edgelist=highlight_edges, edge_color='y', width=8, alpha=0.5, ax=ax[0])

    nx.draw_networkx_edges(graph, pos, ax=ax[0])
    nx.draw_networkx_edge_labels(graph, pos, font_size=16,
                                 edge_labels=edge_labels, ax=ax[0])
    ax[0].set_axis_off()

    plt.axis('off')
    plt.title(f"motif {title}")
    plt.show()


def ablation_draw():
    g_name = "1fmn_#0.1:A:FMN:36.nx_annot.p"
    modes = ['', '_bb-only', '_wc-bb', '_wc-bb-nc', '_no-label', '_label-shuffle']
    for m in modes:
        g_dir = "../data/annotated/pockets_nx" + m
        g, _, _, _ = pickle.load(open(os.path.join(g_dir, g_name), 'rb'))
        rna_draw(g, title=m)
    pass


if __name__ == "__main__":
    ablation_draw()
