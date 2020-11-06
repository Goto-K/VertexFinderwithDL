import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

event5167 = np.load("Full_plus_Predict_vfdnn_03_5167.npy")
G = nx.Graph()

bnames = ['nevent', 'ntr1track', 'ntr2track', 
         # Track 1 : 22 input variables 
          'tr1d0', 'tr1z0', 'tr1phi', 'tr1omega', 'tr1tanlam', 'tr1charge', 'tr1energy', 
          'tr1covmatrixd0d0', 'tr1covmatrixd0z0', 'tr1covmatrixd0ph', 'tr1covmatrixd0om', 'tr1covmatrixd0tl', 
          'tr1covmatrixz0z0', 'tr1covmatrixz0ph', 'tr1covmatrixz0om', 'tr1covmatrixz0tl', 'tr1covmatrixphph', 
          'tr1covmatrixphom', 'tr1covmatrixphtl', 'tr1covmatrixomom', 'tr1covmatrixomtl', 'tr1covmatrixtltl',         
         # Track 2 : 22 input variables 
          'tr2d0', 'tr2z0', 'tr2phi', 'tr2omega', 'tr2tanlam', 'tr2charge', 'tr2energy', 
          'tr2covmatrixd0d0', 'tr2covmatrixd0z0', 'tr2covmatrixd0ph', 'tr2covmatrixd0om', 'tr2covmatrixd0tl', 
          'tr2covmatrixz0z0', 'tr2covmatrixz0ph', 'tr2covmatrixz0om', 'tr2covmatrixz0tl', 'tr2covmatrixphph', 
          'tr2covmatrixphom', 'tr2covmatrixphtl', 'tr2covmatrixomom', 'tr2covmatrixomtl', 'tr2covmatrixtltl',  
         # fitter feature value 
          'vchi2', 'vposx', 'vposy', 'vposz', 'mass', 'mag', 'vec', 'tr1selection', 'tr2selection', 'v0selection', 
          'connect', 'lcfiplustag',  
          'tr1id', 'tr1pdg', 'tr1ssid', 'tr1sspdg', 'tr1ssc', 'tr1ssb', 'tr1oth', 'tr1pri',  
          'tr2id', 'tr2pdg', 'tr2ssid', 'tr2sspdg', 'tr2ssc', 'tr2ssb', 'tr2oth', 'tr2pri']

for event in event5167: 
    G.add_edge(event[1], event[2])
    G[event[1]][event[2]].update({"nc":event[-5], "pv":event[-4], "svbb":event[-3], "svcc":event[-2], "svbc":event[-1]})

edge_nc = [ d["nc"] for (u,v,d) in G.edges(data=True)]
edge_pv = [ d["pv"] for (u,v,d) in G.edges(data=True)]
edge_svbb = [ d["svbb"] for (u,v,d) in G.edges(data=True)]
edge_svcc = [ d["svcc"] for (u,v,d) in G.edges(data=True)]
edge_svbc = [ d["svbc"] for (u,v,d) in G.edges(data=True)]

pos = nx.layout.spring_layout(G)

nodes = nx.draw_networkx_nodes(G, pos, node_size=1, node_color='blue')
edges = nx.draw_networkx_edges(G, pos, node_size=1, edge_color=edge_pv, alpha=0.5, arrows=False, edge_cmap=plt.cm.Blues, width=2)

for i, event in enumerate(event5167):
    edges[i].set_alpha(event[-5])

pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
pc.set_array(edge_colors)
plt.colorbar(pc)

ax = plt.gca()
ax.set_axis_off()
plt.show()
