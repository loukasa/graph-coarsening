# graph-coarsening package

Multilevel graph coarsening algorithm with spectral and cut guarantees.

The code accompanies paper [Graph reduction with spectral and cut guarantees](http://www.jmlr.org/papers/volume20/18-680/18-680.pdf) by Andreas Loukas published at JMLR/2019 ([bibtex](http://www.jmlr.org/papers/v20/18-680.bib)).

## Paper asbtract 
Can one reduce the size of a graph without significantly altering its basic properties? The graph reduction problem is hereby approached from the perspective of restricted spectral approximation, a modification of the spectral similarity measure used for graph sparsification. This choice is motivated by the observation that restricted approximation carries strong spectral and cut guarantees, and that it implies approximation results for unsupervised learning problems relying on spectral embeddings. The article then focuses on coarsening - the most common type of graph reduction. Sufficient conditions are derived for a small graph to approximate a larger one in the sense of restricted approximation. These findings give rise to algorithms that, compared to both standard and advanced graph reduction methods, find coarse graphs of improved quality, often by a large margin, without sacrificing speed.

## Contents

There are five python notebooks included under `examples`:

* `coarsening_demo.ipynb` demonstrates how the code can be used with a toy example (see also [blogpost](https://andreasloukas.blog/2018/11/05/multilevel-graph-coarsening-with-spectral-and-cut-guarantees/)).
* `coarsening_methods.ipynb` shows the effect of different coarsening methods on a toy example.
* `experiment_approximation.ipynb` reproduces the results of Section 5.1.
* `experiment_spectrum.ipynb` reproduces the results of Section 5.2.
* `experiment_scalability.ipynb` reproduces the results of Section 5.3.

Since I have not fixed the random seed, some small variance should be expected in the experiment output.

## Installation instructions: 
`
git clone git@github.com:loukasa/graph-coarsening.git
cd graph-coarsening
pip install .
`
(depedencies: pygsp, matplotlib, numpy, scipy, networkx, sortedcontainers)

## Acknowledgements

This work was kindly supported by the Swiss National Science Foundation (grant number PZ00P2 179981). I would like to thank [Scott Gigante](https://cbb.yale.edu/people/scott-gigante) for helping package the code.

15 May 2020

[Andreas Loukas](https://andreasloukas.blog)

[![DOI](https://zenodo.org/badge/175851068.svg)](https://zenodo.org/badge/latestdoi/175851068)

Released under the Apache license 2.0
