# OT: a BN learning method based on high-order HSIC
 we develop a two-phase algorithm namely optimal-tuning (OT) algorithm to locally amend the global optimization. In the optimal phase, an optimization problem based on first-order Hilbert-Schmidt independence criterion (HSIC) gives an estimated skeleton as the initial determined parents subset. In the tuning phase, the skeleton is locally tuned by deletion, addition and DAG-formalization strategies using the theoretically proved incremental properties of high-order HSIC.
### Requires
tensorflow 1.15.1
networkx 2.6.3
### Running the tests
Clone the package.

`git clone https://github.com/YafeiannWang/optimal-tune-algorithm.git`

Run the generate_data.py file for generating sythetic data.

Run the OT.py file for learning the estimated graph.
### Publication
Wang Y, Liu J. Learning nonparametric DAGs with incremental information via high-order HSIC[J]. arXiv preprint arXiv:2308.05969, 2023.
