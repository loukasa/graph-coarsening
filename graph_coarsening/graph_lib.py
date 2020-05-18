import numpy as np
import scipy as sp
import pygsp as gsp

import os
import tempfile
import zipfile

from . import graph_utils

from pygsp import graphs
from scipy import sparse
from urllib import request

_YEAST_URL = "http://nrvis.com/download/data/bio/bio-yeast.zip"
_MOZILLA_HEADERS = [("User-Agent", "Mozilla/5.0")]


def download_yeast():
    r"""
    A convenience method for loading a network of protein-to-protein interactions in budding yeast.

    http://networkrepository.com/bio-yeast.php
    """
    with tempfile.TemporaryDirectory() as tempdir:
        zip_filename = os.path.join(tempdir, "bio-yeast.zip")
        with open(zip_filename, "wb") as zip_handle:
            opener = request.build_opener()
            opener.addheaders = _MOZILLA_HEADERS
            request.install_opener(opener)
            with request.urlopen(_YEAST_URL) as url_handle:
                zip_handle.write(url_handle.read())
        with zipfile.ZipFile(zip_filename) as zip_handle:
            zip_handle.extractall(tempdir)
        mtx_filename = os.path.join(tempdir, "bio-yeast.mtx")
        with open(mtx_filename, "r") as mtx_handle:
            _ = next(mtx_handle)  # header
            n_rows, n_cols, _ = next(mtx_handle).split(" ")
            E = np.loadtxt(mtx_handle)
    E = E.astype(int) - 1
    W = sparse.lil_matrix((int(n_rows), int(n_cols)))
    W[(E[:, 0], E[:, 1])] = 1
    W = W.tocsr()
    W += W.T
    return W


def real(N, graph_name, connected=True):
    r"""
    A convenience method for loading toy graphs that have been collected from the internet.

	Parameters:
	----------
	N : int
	    The number of nodes. Set N=-1 to return the entire graph.

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

    directory = os.path.join(
        os.path.dirname(os.path.dirname(graph_utils.__file__)), "data"
    )

    tries = 0
    while True:
        tries = tries + 1

        if graph_name == "airfoil":
            G = graphs.Airfoil()
            G = graphs.Graph(W=G.W[0:N, 0:N], coords=G.coords[0:N, :])

        elif graph_name == "yeast":
            W = download_yeast()
            G = graphs.Graph(W=W[0:N, 0:N])

        elif graph_name == "minnesota":
            G = graphs.Minnesota()
            W = G.W.astype(np.float)
            G = graphs.Graph(W=W[0:N, 0:N], coords=G.coords[0:N, :])

        elif graph_name == "bunny":
            G = graphs.Bunny()
            W = G.W.astype(np.float)
            G = graphs.Graph(W=W[0:N, 0:N], coords=G.coords[0:N, :])

        if connected == False or G.is_connected():
            break
        if tries > 1:
            print("WARNING: Disconnected graph. Using the giant component.")
            G, _ = graph_utils.get_giant_component(G)
            break
            
    if not hasattr(G, 'coords'): 
        try:
            import networkx as nx
            graph = nx.from_scipy_sparse_matrix(G.W)
            pos = nx.nx_agraph.graphviz_layout(graph, prog='neato')  
            G.set_coordinates(np.array(list(pos.values()))) 
        except ImportError:
            G.set_coordinates()
        
    return G


def models(N, graph_name, connected=True, default_params=False, k=12, sigma=0.5):

    tries = 0
    while True:
        tries = tries + 1
        if graph_name == "regular":
            if default_params:
                k = 10
            offsets = []
            for i in range(1, int(k / 2) + 1):
                offsets.append(i)
                offsets.append(-(N - i))

            offsets = np.array(offsets)
            vals = np.ones_like(offsets)
            W = sp.sparse.diags(
                vals, offsets, shape=(N, N), format="csc", dtype=np.float
            )
            W = (W + W.T) / 2
            G = graphs.Graph(W=W)

        else:
            print("ERROR: uknown model")
            return

        if connected == False or G.is_connected():
            break
        if tries > 1:
            print("WARNING: disconnected graph.. trying to use the giant component")
            G = graph_utils.get_giant_component(G)
            break
    return G
