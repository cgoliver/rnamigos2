"""
Take a PDB with binding site residues and produces binding pocket graph based
on carnaval fr3d graphs.
"""

import os,sys
import pickle
import itertools
from queue import Queue
from collections import Counter


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import rna_layout
from pocket_draw import *


def node_search(v, w, G, pos_map, broaden=False, k=6):
    """
    connect residues v and w using BFS.

    takes fr3d nodes and a map from fr3d nodes to carnaval nodes

    returns carnaval nodes
    """

    visited = []

    q = Queue()
    depth = 0


    #add all nodes adjacent to v
    try:
        v_carna = pos_map[v]
        dest = pos_map[w]
    except KeyError:
        print(f"NODE {v} or {w} NOT FOUND")
        return []
    else:
        q.put(v_carna)
        while not q.empty():
            u = q.get()
            if u == dest and not broaden or depth >= k:
                return visited
            elif depth == k and broaden:
                return visited
            else:
                depth += 1
                for n in G.neighbors(u):
                    if (u, n) not in visited:
                        q.put(n)
                        visited.append((u, n))
    return visited
def chain_filter(G, chain):
    """
    Filter graph to include only nodes within a given chain.
    Returns new graph with residues only in chain.
    """

    G_chain = nx.Graph()
    edges = [e for e in G.edges.data() if e[0][0] == chain and e[1][0] == chain]
    nodes = [(n[0], n[1]) for n in G.nodes.data() if n[0][0] == chain]
    print(f"found {len(nodes)} nodes in chain {chain}")
    G_chain.add_edges_from(edges)
    G_chain.add_nodes_from(nodes)
    return G_chain

    pass
def _graph_draw(pocket_G, G, anchors, title=""):
        node_color = []
        #only draw nodes for current chain
        # chain_nodes = [n for n in G.nodes if n[0] == chain]
        """
        for n in pocket_G.nodes:
            if G.node[n]['fr3d'] in anchors:
                node_color.append('green')
            elif n in pocket_G.nodes:
                node_color.append('blue')
            else:
                node_color.append('red')
        """
        # pos = nx.circular_layout(G)
        pos = nx.spring_layout(pocket_G)
        # pos = rna_layout.circular_layout(G)
        edge_labels = {(e[0], e[1]):e[2]['label'] for e in pocket_G.edges.data()}
        # node_labels = {n: (G.nodes[n]['fr3d'], n[1]) for n in\
            # pocket_G.nodes}
        node_labels = {n:n for n in\
            pocket_G.nodes}
        nx.draw_networkx_edge_labels(pocket_G, pos, edge_labels)
        nx.draw_networkx(pocket_G, pos)
        # nx.draw_networkx(pocket_G, pos, node_color=node_color)
        plt.title(title)
        # plt.title(f"{pdbid}, {lig_name}, {ligand_id}")
        plt.show()

def build_graph(df, carna_dir, bfs=True, remove_stacks=True, draw=True, save=False, remove_dangles=True,
        loop_collapse=True):
    """
    Build graph for each binding pocket in DataFrame.
    """
    #drop all ions 
    df = df[df['ion'] != True]
    pdb_group = df.groupby(['pdb_id'])
    lost = 0
    sizes = []
    sizes_dangles = []
    for pdbid, pocket in pdb_group:
        try:
            G = nx.read_gpickle(os.path.join(carna_dir,\
                f"{pdbid.upper()}.nxpickled"))
            #build mapping from carnaval to fr3d positions
        except FileNotFoundError:
            print(f"carnaval file {pdbid} not found")
            continue

        for name, p in pocket.groupby(['ligand', 'ligand_id']):
            #carnaval fr3d attribute has position as string!
            lig_name, ligand_id= name
            print(name)
            if lig_name == "NCO":
                continue
            # print(f"filtering chain {chain}")
            # G = chain_filter(G_full, chain)

            #fr3d to carnaval dictionary
            pos_map = {G.node[n]['fr3d']:n for n in G.nodes}
            #map fr3d to carnaval
            carna_map = {n:G.node[n]['fr3d'] for n in G.nodes}
            anchors = {(res.chain, str(res.position)) for res in p.itertuples()}
            bind_colors = lambda G: ['green' if carna_map[n] in anchors else 'grey'
                for n in G.nodes]
            pocket_G = nx.Graph()
            pocket_G.graph = G.graph
            edges = []
            for v, w in itertools.combinations(anchors, 2):
                if bfs:
                    if len(anchors) < 2:
                        edges += node_search(v, v, G, pos_map, broaden=True)
                    else:
                        edges += node_search(v, w, G, pos_map, k=2)
                else:
                    try:
                        #convert to carnaval nodes 
                        c1 = pos_map[v]
                        c2 = pos_map[w]
                        G[c1][c2]
                        edges.append((c1, c2))
                        for n in G.neighbors(c1):
                            edges.append((c1, n))
                        for n in G.neighbors(c2):
                            edges.append((c2, n))
                    except KeyError:
                        continue
                    pass

            pocket_G.add_edges_from((e[0], e[1], G.edges[e]) for e in edges)

            if remove_dangles:
                pocket_G = dangle_trim(pocket_G)
                pocket_G = reconnect(G, pocket_G)
            if remove_stacks:
                pocket_G = _stack_trim(pocket_G)
            interactions = [e for e in pocket_G.edges.data() if e[2]['label'] != 'B53']
            pocket_G = reconnect(G, pocket_G)
            pocket_G = floaters(pocket_G)

            if len(interactions) < 2:
                # _graph_draw(pocket_G, G, anchors, title="deleted")
                lost += 1
                continue
            else:
                # graph_draw(pocket_G, node_colors=bind_colors(pocket_G), title=f"done: {pdbid, lig_name}")
                # graph_draw(pocket_G)
                pass
            if loop_collapse:
                pocket_G = _loop_collapse(pocket_G)

            # stack_collapse(pocket_G)
            nx.set_node_attributes(pocket_G,{n:d for n,d in G.nodes.data()})
            if save:
                print(f"../Data/{save}/{pdbid}_{lig_name}_{ligand_id}.nxpickle")
                nx.write_gpickle(pocket_G,\
                    f"../Data/{save}/{pdbid}_{lig_name}_{ligand_id}.nxpickle")
            if draw:
                # graph_draw(pocket_G, show=False, save=f"../Data/pockets_beta_images_nt/{pdbid}_{lig_name}_{ligand_id}.pdf")
                graph_draw(pocket_G, show=True, save=False)
    print(f"lost: {lost} graphs because they were too small.")
    pass
def floaters(G):
    """
    Try to connect floating base pairs. (Single base pair not attached to backbone).
    Otherwise remove.
    """
    deg_ok = lambda H, u, v, d: (H.degree(u) == d) and (H.degree(v) == d)
    floaters = []
    for u,v in G.edges():
        if deg_ok(G, u, v, 1):
            floaters.append((u,v))

    G.remove_edges_from(floaters)

    return G
def reconnect(G, pocket_G):
    """
    Connect missing links.
    """
    H = pocket_G.copy()
    for u, v in itertools.combinations(H.nodes(), 2):
        try:
            G[u][v]
        except KeyError:
            pass
        else:
            # print(f"ADDING {u, v}")
            H.add_edge(u, v, **G[u][v])
        try:
            G[v][u]
        except KeyError:
            pass
        else:
            H.add_edge(v, u, **G[v][u])

    return H

def _stack_trim(G):
    open_e = []
    edges_todo = set([(u,v) for u,v in G.edges if G[u][v]['label'] == 'CWW'])
    cur_G = G.copy()
    deg_ok = lambda H, u, v, d: (H.degree(u) == d) and (H.degree(v) == d)
    non_bb = lambda H, u: len([u for u,info in H[u].items() if info['label'] != 'B53'])
    # print(f"initial {len(cur_G.nodes)}")
    while True:
        # graph_draw(cur_G)
        # print(len(cur_G.nodes))
        # print(open_e)
        #find potential stack openings
        # print(cur_G.nodes)
        # print(f"todo: {edges_todo}")
        for u,v in edges_todo:
            inter = non_bb(cur_G, u) == 1 and non_bb(cur_G, v) == 1
            if deg_ok(cur_G, u,v, 2) and (cur_G[u][v]['label'] == 'CWW') and inter:
                if (u,v) not in open_e:
                    open_e.append((u,v))
                    # print(f"candidate: {u,v, cur_G.nodes}")
        # print(open_e)
        #see if need to remove an edge
        try:
            u,v = open_e.pop()
        except IndexError:
            break

        edges_todo.remove((u,v))
        #find stacking partners ndoes b, m
        b,m = (None, None)
        for node, info in cur_G[u].items():
            if info['label'] == 'B53':
                b = node
        for node, info in cur_G[v].items():
            if info['label'] == 'B53':
                m = node
        if None in (b, m):
            continue
        try:
            #check if partner nodes are adjacent
            cur_G[b][m]
        except KeyError:
            continue
        inter = non_bb(cur_G, b) == 1 and non_bb(cur_G, m) == 1
        # # print(f"partners: {u,v,b,m}")
        # print(f"NODES: {cur_G[b]}")
        # print(f"partner test: {u, v, b,m, deg_ok(cur_G, b,m, 3), inter, cur_G[b][m]['label'] == 'CWW'}")
        if deg_ok(cur_G, b, m, 3) and cur_G[b][m]['label'] == 'CWW' and inter:
            # print(f"REMOVING ({u}, {v})")
            # cur_G.remove_nodes_from([u,v])
            # cur_G = cur_G.copy()
            cur_G = cur_G.subgraph([n for n in cur_G.nodes if n not in [u, v]]).copy()
            # print(f"new nodes {cur_G.nodes}")
            # print(len(cur_G.nodes))
        if len(open_e) == 0 or len(edges_todo) == 0:
            break
    return cur_G

def stack_trim(G):
    """
    Remove stacks from graph.
    """
    is_ww = lambda n,G: 'CWW' in [info['label'] for node,info in G[n].items()]
    degree = lambda i, G, nodelist: np.sum(nx.to_numpy_matrix(G, nodelist=nodelist)[i])
    cur_G = G.copy()
    while True:
        stacks = []
        for n in cur_G.nodes:
            if cur_G.degree(n) == 2 and is_ww(n, cur_G):
                #potential stack opening
                partner = None
                stacker = None
                for node, info in cur_G[n].items():
                    if info['label'] == 'B53':
                        stacker = node
                    elif info['label'] == 'CWW':
                        partner = node
                    else:
                        pass
                if cur_G.degree(partner) > 3:
                    continue
                partner_2 = None
                stacker_2 = None
                for node, info in cur_G[partner].items():
                    if info['label'] == 'B53':
                        stacker_2 = node
                    elif info['label'] == 'CWW':
                        partner_2 = node
                try:
                    if cur_G[stacker][stacker_2]['label'] == 'CWW':
                        stacks.append(n)
                        stacks.append(partner)
                except KeyError:
                    continue
        if len(stacks) == 0:
            break
        else:
            cur_G.remove_nodes_from(stacks)
            cur_G = cur_G.copy()
    return cur_G

def dangle_trim(G):
    """
    Recursively remove dangling nodes from graph.
    """
    is_backbone = lambda n,G: sum([G[n][nei]['label'] != 'B53' for nei in G.neighbors(n)]) == 0
    degree = lambda i, G, nodelist: np.sum(nx.to_numpy_matrix(G, nodelist=nodelist)[i])
    cur_G = G.copy()
    while True:
        dangles = []
        for n in cur_G.nodes:
            # node_deg = degree(i, G, current_nodeset)
            # print(node_deg)
            # if node_deg == 2 and is_backbone(n, G):
            if cur_G.degree(n) == 1 and is_backbone(n, cur_G) or cur_G.degree(n) == 0:
                dangles.append(n)
        if len(dangles) == 0:
            break
        else:
            cur_G.remove_nodes_from(dangles)
            cur_G = cur_G.copy()
    return cur_G

def in_stem(G, u,v):
    non_bb = lambda G, n: len([info['label'] for node, info in G[n].items() if info['label'] != 'B53'])
    is_ww = lambda G, u, v: G[u][v]['label'] == 'CWW'
    if is_ww(G,u,v) and (non_bb(G,u) in (1, 2)) and (non_bb(G,v) in (1, 2)):
        return True
    return False
    
def stack_collapse(G):
    """
    Collapse each stacking region to single node.
    """

    stems = []
    for u,v in G.edges():
            if in_stem(G, u, v):
                #potentially in stem, try to expand.
                while True:
                    vicinity = set()
                    v_nodes = []
                    for n in G.neighbors(u):
                        if n == v:
                            continue
                        v_nodes.append(n)
                    for n in G.neighbors(v):
                        if n == u:
                            continue
                        v_nodes.append(n)
                    for w,z in itertools.combinations(v_nodes, 2):
                        try:
                            stem_continue = G[w][z]['label'] == 'CWW' and in_stem(G, w,z)
                        except KeyError:
                            pass
                        else:
                            if stem_continue:
                                vicinity.add((w,z))
                    print(f"VICINITY: {u,v}: {vicinity}")
                    for v in vicinity:
                        pass
                    break
    pass
def _loop_collapse(G, draw=False):
    """
    Collapse backbone regions in nodes with no other interactions to
    single node with weight counting the number of collapsed nodes.

    Algo:
        - per chain sort nodes in ascending order
        - if find contiguous stretch with no other interactions, collapse
    """
    G = nx.to_undirected(G)
    nodelist = sorted(sorted(G.nodes, key=lambda x:x[1]), key=lambda x:x[0])
    is_backbone = lambda n,G: not sum([G[n][nei]['label'] != 'B53' 
        for nei in G.neighbors(n)])
    current_chain = nodelist[0][0]
    in_loop = False
    loop_size = 0
    loops = []
    current_loop = []
    prev_node =  None
    for n in nodelist:
        if n[0] != current_chain:
            current_chain = n[0]
        try:
            if is_backbone(n, G) and not in_loop:
                #opening a loop
                in_loop = True
                current_loop.append(n)
            elif not is_backbone(n, G) and in_loop:
                #ending a loop
                loops.append(current_loop)
                current_loop = []
                in_loop = False
            elif is_backbone(n,G) and in_loop:
                #extending a loop
                if n[1] == current_loop[-1][1] + 1 \
                    and current_chain == n[0]:
                    current_loop.append(n)
            else:
                continue
        except KeyError:
            pass
    #build graph with collapsed nodes
    loop_nodes = {n for l in loops for n in l if len(l) > 1}
    # collapsed = G.subgraph([n for n in G.nodes if n not in loop_nodes])
    C = nx.Graph()
    for e1,e2,d in G.edges.data():
        if not ((e1 in loop_nodes) or (e2 in loop_nodes)):
            C.add_edge(e1, e2, **d)
    for i,loop in enumerate(loops):
        if len(loop) < 2:
            continue
        try:
            left = [n for n in G.neighbors(loop[0]) if not is_backbone(n,G)][0]
        except:
            print("COLLAPSE FAIL")
            return G 
        right = [n for n in G.neighbors(loop[-1]) if not is_backbone(n,G)][0]
        #find midpoint position to place loop
        mid = int((left[1] + right[1]) / 2)
            # graph_draw(pocket_G)
        lnode = (left[0], mid, 'L')
        C.add_node(lnode, nt='LOOP',num=i,size=len(loop))
        C.add_edges_from([(lnode, left, {'label':'B53'}),
                          (lnode, right, {'label':'B53'})])
    if draw:
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos)
        edge_labels = {(e[0], e[1]):e[2]['label'] for e in G.edges.data()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
        plt.title("BEFORE")
        plt.show()
        pos = nx.spring_layout(C)
        nx.draw_networkx(C, pos)
        edge_labels = {(e[0], e[1]):e[2]['label'] for e in C.edges.data()}
        nx.draw_networkx_edge_labels(C, pos, edge_labels)
        plt.title("AFTER")
        plt.show()
    return C

if __name__ == "__main__":
    data_dir = os.path.join("..", "Data")
    carnaval_dir = os.path.join(data_dir, "all_graphs_pickled")
    # binding_residues = pd.read_csv("binding_residues_5A_interchain.csv")
    binding_residues = pd.read_csv("bind_pockets_residues_prot.csv")

    # build_graph(binding_residues, carnaval_dir, draw=True,
        # save='../Data/pockets_beta_nt', bfs=True, loop_collapse=True)
    build_graph(binding_residues, carnaval_dir, draw=False,
        save="../Data/graphs_prot", bfs=True)
    sys.exit()
    modules = pickle.load(open('../Data/r3dm_modules.pickle', 'rb'))
    collapsed = []
    for i,m in enumerate(modules):
        g = dangle_trim(m)
        g = loop_collapse(g)
        g = stack_trim(g)
        g.title = str(i)
        collapsed.append(g)
    pickle.dump(collapsed, open('../Data/r3dm_clean.pickle', 'wb'))
    # sizes = [len(g.nodes) for g in collapsed]
    # sizes_before = [len(g.nodes) for g in modules]
    # sns.distplot(sizes, label="after")
    # sns.distplot(sizes_before, label="before")
    # plt.legend()
    # plt.show()
