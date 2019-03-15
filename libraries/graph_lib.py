import numpy as np
import scipy as sp
import pygsp as gsp
from pygsp import graphs
import libraries.graph_utils as graph_utils
import os

def real(N, graph_name, connected=True):
    r""" 
    A convenience method for loading toy graphs that have been collected from the internet.
 
	Parameters:
	----------
	N : int 
	    The number of nodes.

	graph_name : a string 
        Use to select which graph is returned. Choices include 
            * airfoil
                Graph from airflow simulation
                http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.9217&rep=rep1&type=pdf
                http://networkrepository.com/airfoil1.php
            * yeast
                Network of protein-to-protein interactions in budding yeast.
                http://networkrepository.com/bio-yeast.php
            * minnesota
                Minnesota road network.
                I am using the version provided by the PyGSP software package (initially taken from the MatlabBGL library.)
            * bunny
                The Stanford bunny is a computer graphics 3D test model developed by Greg Turk and Marc Levoy in 1994 at Stanford University
                I am using the version provided by the PyGSP software package.
	connected : Boolean
        Set to True if only the giant component is to be returned.    
"""

    directory = os.path.dirname(graph_utils.__file__) + '/data/'
    
    tries = 0    
    while True:
        tries = tries + 1

        if graph_name == 'airfoil': 
            G = graphs.Airfoil()
            G = graphs.Graph(W=G.W[0:N,0:N], coords=G.coords[0:N,:])

        elif graph_name == 'yeast': 
            file = directory + 'bio-yeast.npy'  
            W = np.load(file)
            G = graphs.Graph(W = W[0:N,0:N])

        elif graph_name == 'minnesota': 
            G = graphs.Minnesota()
            W = G.W.astype(np.float)
            G = graphs.Graph(W=W[0:N,0:N], coords=G.coords[0:N,:])

        elif graph_name == 'bunny':
            G = graphs.Bunny()
            W = G.W.astype(np.float)
            G = graphs.Graph(W=W[0:N,0:N], coords=G.coords[0:N,:])
            
        if connected==False or G.is_connected(): break 
        if tries > 1: 
            print('WARNING: disconnected graph.. trying to use the giant component')
            G,_ = graph_utils.get_giant_component(G)
            break
    return G

def models(N, graph_name, connected=True, default_params=False, k=12, sigma=0.5):

    tries = 0    
    while True:
        tries = tries + 1
        if graph_name == 'regular':         
            if default_params: k = 10; 
            offsets = []
            for i in range(1,int(k/2)+1):
                offsets.append(i) 
                offsets.append(-(N-i)) 

            offsets = np.array(offsets)
            vals = np.ones_like(offsets)
            W  = sp.sparse.diags(vals, offsets, shape=(N, N), format='csc', dtype=np.float)
            W = (W + W.T)/2
            G = graphs.Graph(W = W)    

        else: print('ERROR: uknown model'); return 
            
        if connected==False or G.is_connected(): break 
        if tries > 1: 
            print('WARNING: disconnected graph.. trying to use the giant component')
            G = graph_utils.get_giant_component(G)
            break
    return G


