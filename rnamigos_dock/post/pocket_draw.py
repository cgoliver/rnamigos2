import os
import sys
import pickle
import networkx as nx
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib import cm

from .rna_layout import circular_layout


params= {'text.latex.preamble' : [r'\usepackage{fdsymbol}\usepackage{xspace}']}
plt.rcParams.update(params)

def graph_draw(G_p, save=False, show=False, node_colors=[], title="", color='grey'):

    labels = {
        'CW': r"$\medblackcircle$\xspace",
        'CS': r"$\medblacktriangleright$\xspace",
        'CH': r"$\medblacksquare$\xspace",
        'TW': r"$\medcircle$\xspace",
        'TS': r"$\medtriangleright$\xspace",
        'TH': r"$\medsquare$\xspace"
    }
    nucs = ['A', 'U', 'C', 'G', 'L', 'N']
    nt_color = {n:i for i,n in enumerate(nucs)}
    make_label = lambda s: labels[s[:2]] + labels[s[0::2]] if len(set(s[1:])) == 2\
        else labels[s[:2]]

    #place loops in sequence order
    print(G_p.nodes.data())
    loop_labels = {}

    G_p = nx.relabel_nodes(G_p, loop_labels)
    G = nx.DiGraph()
    G.add_edges_from([(*sorted((u,v), key=lambda x:x[1]), d) for u,v,d in G_p.edges.data()])

    G = nx.relabel_nodes(G, loop_labels)

    edge_labels = {(e1, e2):
        make_label(d['label']) if d['label'] != 'B53' else '' for e1, e2, d in G.edges.data()}

    pos = circular_layout(G)
    # pos = nx.spring_layout(G)

    # if len(node_colors) > 0:
        # nx.draw_networkx(G, pos, node_size=800, width=2, node_color=node_colors)
    # else:
        # nx.draw(G, pos, node_size=800, width=2, node_color='grey')

    bb = [(u,v) for u,v in G.edges if G[u][v]['label'] == 'B53']
    bp = [(u,v) for u,v in G.edges if G[u][v]['label'] != 'B53']
    # lr = [(u,v) for u,v in bp if G[u][v]['long_range'] == True]
    # sr = [(u,v) for u,v in bp if G[u][v]['long_range'] == False]

    nx.draw_networkx_nodes(G, pos, node_size=500, width=2,
                            node_color=[cm.Set3(nt_color[G_p.nodes.data()[n]['nt']]) for n in G.nodes])

    nx.draw_networkx_edges(G, pos, edgelist=bb, arrows=True, width=1.5, arrowsize=25)
    nx.draw_networkx_edges(G, pos, edgelist=bp, arrows=False, width=2)
    # nx.draw_networkx_edges(G, pos, edgelist=lr, arrows=False, width=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=25)
    # nx.draw_networkx_labels(G, pos, labels={n:f"{n[0]}{n[1]}-{G_p.nodes.data()[n]['nt']}" for n in G.nodes()})
    nx.draw_networkx_labels(G, pos, labels={n:f"{G_p.nodes.data()[n]['nt']}" for n in G.nodes()},
        font_size=20)

    plt.axis('off')


    plt.title(title)

    if save:
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.savefig(save, format='pdf')
        # plt.savefig(save.replace('pdf', 'png'), format='png', dpi=300)
    if show:
        plt.show()
    plt.clf()
if __name__ == "__main__":
    g = nx.read_gpickle('../data/sample_graphs/3ds7_GNG_P.nxpickle')
    graph_draw(g, show=True)
