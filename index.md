## WSDM 2022 Challenge - Temporal Link Prediction

### Description

We are DGL Team, the developers of [DGL](https://www.dgl.ai), one of the leading libraries dealing with deep learning on graphs.  We are deeply interested in methods of machine learning on graphs, as real world datasets can often be expressed as graphs, with entities as nodes and interactions as edges.  Examples include user-user interaction in social networks, user-item interactions in recommender systems, etc.  Moreover, these graphs are often in practice temporal, with new edges coming in with timestamps.  Therefore, our next focus is to support learning on large-scale temporal graphs, for which we are actively seeking out solutions that works well in general.

Temporal Link Prediction is one of the classical tasks on temporal graphs.  Contrary to link prediction which asks if an edge exists between two nodes on a partially observed graph, Temporal Link Prediction asks if an edge will exist between two nodes within a given time span.  It is more useful than traditional link prediction as one can then build multiple applications around the model, such as forecasting the demand of customers in E-commerce, or forecasting what event will happen in a social network, etc.

We are expecting an approach that works well on large-scale temporal graphs in general.  In this challenge, we expect a single model that works well on two kinds of data simultaneously:

* Dataset A: a dynamic event graph with entities as nodes and events as edges.
* Dataset B: a user-item graph with users and items as nodes and various interactions as edges.

The task will be predicting whether an edge of a given type will exist between two given nodes before a given timestamp.

### Description of Dataset A

(to be announced)

### Description of Dataset B

(to be announced)

### Test set and submission format

Please complete the submission in [this form](todo).

For Dataset A, we will release a CSV file `input_A.csv` containing the following five columns:

* `src_id`: The source node ID
* `dst_id`: The destination node ID
* `event_type`: The event type ID
* `start_time`: The starting timestamp in Unix Epoch
* `end_time`: The ending timestamp in Unix Epoch

We expect a submission file `output_A.csv` with all the above columns, and additionally the following column:

* `exists`: 1 if an edge with type `event_type` from node ID `src_id` to node ID `dst_id` is predicted to exist between timestamp `start_time` and `end_time`.  0 otherwise.

A sample `output_A.csv` file can be downloaded [here](todo).

For Dataset B, we will release a CSV file `input_B.csv` containing the following five columns:

`user_id`: The user ID
`item_id`: The item ID
`interaction_type`: The interaction type ID
`start_time`: The starting timestamp in Unix Epoch
`end_time`: The ending timestamp in Unix Epoch

We expect a submission file `output_B.csv` with all the above columns, and additionally the following column:

`exists`: 1 if an edge with type `interaction_type` from user ID `user_id` to item ID `item_id` is predicted to exist between timestamp `start_time` and `end_time`.  0 otherwise.

A sample `output_B.csv` file can be downloaded [here](todo).

### Competition Terms and Conditions

We expect all the participants to open source their submissions.

Participants are allowed to participate only once, with no concurrent submissions or code sharing between the teams.

Participants are not allowed to use external datasets or pretrained models.

### Evaluation Criteria

We use Area Under ROC (AUC) as evaluation metric for both datasets, and use the harmonic average of the two AUCs as the score of the submission.  Specifically, let `AUC_A` and `AUC_B` be the AUC for Dataset A and Dataset B respectively, the final score is `2 / (1 / AUC_A + 1 / AUC_B)`.  This is to encourage the submissions to work well on both tasks, instead of working extremely well on one while sacrificing the other.

### Schedule

(to be announced)

### Support or Contact

(to be announced)

### Links

WSDM call for cup proposals: https://www.wsdm-conference.org/2022/call-for-wsdm-cup-proposals/
