if __name__ == "__main__":
    import sys
    sys.path.append("..")
import os
import sys
import pickle
import networkx as nx
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib import cm

from tools.rna_layout import circular_layout


params= {'text.latex.preamble' : [r'\usepackage{fdsymbol}\usepackage{xspace}']}
plt.rcParams.update(params)

def graph_draw(G, ax=None, save=False, show=False, node_colors=[], title="", color='grey'):

    labels = {
        'CW': r"$\medblackcircle$\xspace",
        'CS': r"$\medblacktriangleright$\xspace",
        'CH': r"$\medblacksquare$\xspace",
        'TW': r"$\medcircle$\xspace",
        'TS': r"$\medtriangleright$\xspace",
        'TH': r"$\medsquare$\xspace",
    }
    nucs = ['A', 'U', 'C', 'G']
    nt_color = {n:i for i,n in enumerate(nucs)}
    make_label = lambda s: labels[s[:2]] + labels[s[0::2]] if len(set(s[1:])) == 2\
        else labels[s[:2]]

    edge_labels = {(e1, e2):
        make_label(d['label']) if d['label'] != 'B53' else '' for e1, e2, d in G.edges.data()}
    # edge_labels = {(e1, e2):
        # d['label'] if d['label'] != 'B53' else '' for e1, e2, d in G.edges.data()}

    # pos = circular_layout(G)
    pos = nx.spring_layout(G)

    bb = [(u,v) for u,v,d in G.edges(data=True) if d['label'] == 'B53']
    bp = [(u,v) for u,v,d in G.edges(data=True) if d['label'] != 'B53']
    # lr = [(u,v) for u,v in bp if G[u][v]['long_range'] == True]
    # sr = [(u,v) for u,v in bp if G[u][v]['long_range'] == False]

    # nx.draw_networkx_nodes(G, pos, node_size=500, width=2, ax=ax,
                            # node_color=[cm.Set3(nt_color[G.nodes.data()[n]['nt']]) for n in G.nodes])

    nx.draw_networkx_nodes(G, pos, node_size=500, width=2, ax=ax)

    nx.draw_networkx_edges(G, pos, edgelist=bb, arrows=True, width=1.5, arrowsize=25, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=bp, arrows=False, width=2, ax=ax)
    # nx.draw_networkx_edges(G, pos, edgelist=lr, arrows=False, width=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=25,ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n:f"{n[0]}{n[1]}" for n in G.nodes()})
    # nx.draw_networkx_labels(G, pos, labels={n:f"{G_p.nodes.data()[n]['nt']}" for n in G.nodes()},
        # font_size=20, ax=ax)

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
    # graphdir = '../Data/pockets_beta_nt'
    # graphdir = '../Data/non_binding_graphs'
    graphdir = '../Data/pockets_gamma'
    # g = nx.read_gpickle('../Data/pockets_beta_nt/1et4_CNC_G.nxpickle')
    n = len(os.listdir(graphdir))
    for i, f in enumerate(os.listdir(graphdir)):
        g = nx.read_gpickle(os.path.join(graphdir, f))
        print(f"{i} of {n}")
        # graph_draw(g, show=False, save=f"../Data/pockets_beta_images_nt/{f.replace('.nxpickle', '.pdf')}")
        # graph_draw(g, show=True)
        graph_draw(g, show=False, save=f"../Data/pockets_gamma_images/{f.replace('.nxpickle', '.pdf')}")
        # graph_draw(g, show=True, save='hello.pdf')
        # sys.exit()
