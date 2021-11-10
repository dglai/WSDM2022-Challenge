# WSDM 2022 Large-scale Temporal Graph Link Prediction - Baseline and Initial Test Set

[WSDM Cup Website link](https://www.wsdm-conference.org/2022/call-for-wsdm-cup-proposals/)

[Link to this challenge](https://www.dgl.ai/WSDM2022-Challenge/)

This branch offers

* An initial test set having a small number of test examples for each dataset, together with their labels in `exist` column.  Note that this test set only serves for development purposes.  So
  * The intermediate and final dataset will **not** contain the `exist` column.
  * This is **not** the intermediate dataset we will be using for ranking solutions.
* A simple baseline that trains on both datasets.

Download links to initial test set: [Dataset A](https://data.dgl.ai/dataset/WSDMCup2022/input_A_initial.csv.gz) [Dataset B](https://data.dgl.ai/dataset/WSDMCup2022/input_B_initial.csv.gz)

## Baseline description

The baseline is only a minimal working example for both datasets, and it is certainly not optimal.  **You are encouraged to tweak it or propose your own solutions from scratch!**

Here we summarize our baseline:
The baseline is an [RGCN](https://arxiv.org/abs/1703.06103)-like GNN model trained on the entire graph.
Event timestamps on the graph are encoded by decomposing the 10-digit decimal integers into 10-dimensional vectors, each element representing a digit.
We train the model as binary classification using a negative-sampling-like strategy.
Given a ground truth event `(s, d, r, t)` with source node `s`, destination node `d`, event type `r` and timestamp `t`, we perturb `t` to obtain a new value `t'`.
We label the quadruplet with 1 if the new timestamp is larger than the original timestamp, and 0 otherwise.  The model is essentially trained to
predict `p(t < t' | s, d, r)`, i.e. the probability that an edge with type `r` exists from source `s` and destination `d` before timestamp `t'`.

## Baseline usage

To use the baseline you need to install [DGL](https://www.dgl.ai).

You also need at least 64GB of CPU memory.  GPU is not required.

1. Convert csv file to DGL graph objects.

   ```
   python csv2DGLgraph.py --dataset [A or B]
   ```

2. Training.

   ```
   python base_pipeline.py --dataset [A or B]
   ```

## Performance on Initial Test Set

The baseline got AUC of 0.511 on Dataset A and 0.510 on Dataset B.
