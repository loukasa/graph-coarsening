# graph-coarsening
Multilevel graph coarsening algorithm with spectral and cut guarantees

The code accompanies the paper: [Graph reduction with spectral and cut guarantees](https://arxiv.org/abs/1808.10650) published in JMLR/2019.

There are four python notebooks included: 

* The "coarsening_demo.ipynb" demonstrates how the code can be used with a toy example (see also [this blogpost](https://andreasloukas.blog/2018/11/05/multilevel-graph-coarsening-with-spectral-and-cut-guarantees/)).
* The "experiment_approximation.ipynb" reproduces the results of Section 5.1
* The "experiment_spectrum.ipynb" reproduces the results of Section 5.2
* The "experiment_scalability.ipynb" reproduces the results of Section 5.3

Since I have not fixed the random seed, some small variance should be expected in the experiment output. 
 

Depedencies: pygsp, matplotlib, numpy, scipy, networkx, sortedcontainers

This work was kindly supported by the Swiss National Science Foundation (grant number PZ00P2 179981).

15 March 2019

[Andreas Loukas](https://andreasloukas.wordpress.com)

Released under the MIT license 
