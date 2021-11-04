# A Simple Baseline for WSDM 2022 Temporal Link Prediction Challenge
https://www.wsdm-conference.org/2022/call-for-wsdm-cup-proposals/

https://www.dgl.ai/WSDM2022-Challenge/

1. convert csv file to dgl.heterograph.

  python csv2DGLgraph.py --dataset [A or B]

2. training using DGL library.

  python base_pipeline.py --dataset [A or B]


# Problem Formulating:
## Original problem: 
Given historical information, estimating the probability ***p*** of link (src,dst,etype) existing during the time span (start,end), aka, $p(src,dst,etype | t \in [start,end])$ 

## Equal to
Given historical information, estimating two probabilities: $p_s(src,dst,etype | t <= start)$ the link happened before start timestamp, 

and $p_e(src,dst,etype | t <= end)$ the link happened before end timestamp.

Therefore, the target probability p can be computed using $p = p_e - p_s$.

# Model description:
## Node Emebdding 
We construct a RGCN-like Heterogenous GNN model using native DGL API, to generate node embedding.
## Timestamp Encoding
For an unix timestamp (e.g., 1234567890), we split it into 10-dimension vector [1,2,3,4,5,6,7,8,9,0], and then the vector is divided by 10, resulting in final time encoding vector [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.0].
## Probability Estimating
We put a triplet (src_embeding, dst_embedding, time_encoding) into an MLP, predicting the probability that the members of this triplet are matched well.
## Negative Sampling
For each triplet we generate one negative triplet. We randomly replace time_encoding by other one that is earlier than the original one.


