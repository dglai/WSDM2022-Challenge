## WSDM 2022 Challenge - Temporal Link Prediction

### Description

Temporal Link Prediction is one of the classical tasks on temporal graphs.  Contrary to link prediction which asks if an edge exists between two nodes on a partially observed graph, Temporal Link Prediction asks if an edge will exist between two nodes within a given time span.  It is more useful than traditional link prediction as one can then build multiple applications around the model, such as forecasting the demand of customers in E-commerce, or forecasting what event will happen in a social network, etc.

We are expecting an approach that works well on large-scale temporal graphs in general.  In this challenge, we expect a single model (hyperparameters can vary) that works well on two kinds of data simultaneously:

* Dataset A: a dynamic event graph with entities as nodes and different types of events as edges.
* Dataset B: a user-item graph with users and items as nodes and different types of interactions as edges.

The task will be predicting whether an edge of a given type will exist between two given nodes before a given timestamp.

### Description of Dataset A

Dataset A contains the following files:

* `edges_train_A.csv` ([Download here](https://data.dgl.ai/dataset/WSDMCup2022/edges_train_A.csv.gz)): the file containing the temporal edges between the nodes.  Each row represents an edge with the following four columns:
  * `src_id`: the source node ID.
  * `dst_id`: the destination node ID.
  * `edge_type`: the edge type ID.
  * `timestamp`: the timestamp in Unix Epoch.
* `node_features.csv` ([Download here](https://data.dgl.ai/dataset/WSDMCup2022/node_features.csv.gz)): the file whose rows represent node features.  The first column is the node ID while the rest of the columns are all anonymized categorical features (which are not necessarily consecutive integers, and do not have an order).  -1 means that the categorical feature is missing.
  * **11/1 Update:** We have removed the nodes that do not appear in the edge set.  This reduced the number of nodes from 69,992 to 19,442.  The test set will not involve any nodes that do not already appear in the given edge set.
* `edge_type_features.csv` ([Download here](https://data.dgl.ai/dataset/WSDMCup2022/edge_type_features.csv.gz)): the file whose rows represent features of edge types.  The first column is the edge type ID while the rest of the columns are all anonymized categorical features.

### Description of Dataset B

Dataset B contains a single file:

* `edges_train_B.csv` ([Download here](https://data.dgl.ai/dataset/WSDMCup2022/edges_train_B.csv.gz)): the file containing the temporal edges between the users and items.  Each row contains the following columns:
  * `src_id`: the source node ID.
  * `dst_id`: the destination node ID.
  * `edge_type`: the interaction type ID.
  * `timestamp`: the timestamp in Unix Epoch.
  * `feat`: a string of comma-separated floating point values representing the edge's anonymized features.  An empty string means that the edge's feature is unavailable.

Note that in this dataset the nodes and edge types do not have features, unlike Dataset A.

### Test set and submission guidelines

We will release two CSV file `input_A.csv` and `input_B.csv` representing the test queries for dataset A and B respectively.  Each file contains the following five columns:

* `src_id`: The source node ID
* `dst_id`: The destination node ID
* `edge_type`: The event type ID
* `start_time`: The starting timestamp in Unix Epoch
* `end_time`: The ending timestamp in Unix Epoch

We expect two files `output_A.csv` and `output_B.csv` representing your predictions on each test query.  Each file should contain the same number of lines as the given input files.  Each line should contain a single number representing the predicted probability that the edge connecting from node ID `src_id` to node ID `dst_id` with type `event_type` will be added to the graph at some time between `start_time` and `end_time` (inclusive of both endpoints).

It is guaranteed that the timestamps in the test set will be always later than the training set.  This is to match a more realistic setting where one learns from the past and predicts the future.

During competition we will release an intermediate test set and a final test set.  The prizes will only depend on the performance on the final test set, and you will need to submit supplementary materials such as your code repository URL.  You can optionally submit your prediction on the intermediate test set and see how your model performs.

**12/23 Update:** **[The intermediate leaderboard](https://data.dgl.ai/dataset/WSDMCup2022/results.xlsx) has been announced.**  Note that the ranking on the intermediate test set will not impact the ranking on the final test set and the prize in any way.

Due to the difference between the difficulties of two datasets, we have also **changed the ranking metric from harmonic average of AUC to the average of T-scores to encourage balancing the performance on both dataset, instead of sacrificing the performance on one for the other.  See the Evaluation Criteria section below for details.**

We have also made available a [quick evaluation platform](http://eval-env.eba-5u39qmpg.us-west-2.elasticbeanstalk.com/) for the intermediate test set.  During the rest of the competition you can submit your prediction on the intermediate test set there as many times as you like.

**The final test set for [dataset A](https://data.dgl.ai/dataset/WSDMCup2022/final/input_A.csv.gz) and [dataset B](https://data.dgl.ai/dataset/WSDMCup2022/final/input_B.csv.gz) has been released as well.  Submission is open with [this Google form](https://forms.gle/xm2AsikFgV9qvDDE7) until January 20th 2022 23:59:59PM AoE.**

**12/13 Update:**

After inspection of the data and submissions we have found that there are some test set records already appearing in the training set.  Moreover, some of the test set records are labeled 0 while they actually appear in the training set.  Because of this, we have updated the [input_A_initial.csv.gz](https://data.dgl.ai/dataset/WSDMCup2022/input_A_initial.csv.gz) and [input_B_initial.csv.gz](https://data.dgl.ai/dataset/WSDMCup2022/input_B_initial.csv.gz) files, as well as the intermediate test set [input_A.csv.gz](https://data.dgl.ai/dataset/WSDMCup2022/intermediate/input_A.csv.gz) and [input_B.csv.gz](https://data.dgl.ai/dataset/WSDMCup2022/intermediate/input_B.csv.gz), in the original URL, removing the test records with issues.  As a result:

* 1802 records in input_A_initial.csv.gz are removed.
* 97 records in input_A.csv.gz are removed.
* 1211 records in input_B_initial.csv.gz are removed.
* 60 records in input_B.csv.gz are removed.

The old intermediate test set are kept here for [dataset A](https://data.dgl.ai/dataset/WSDMCup2022/intermediate-old/input_A.csv.gz) and [dataset B](https://data.dgl.ai/dataset/WSDMCup2022/intermediate-old/input_B.csv.gz).

As a result:

* **Intermediate test set prediction submission remains available through [this Google form](https://forms.gle/X8xkMmSyq3iZXYUi8) until ~~December 11th 2021~~ December 20th 2021 23:59:59PM AoE.  You will need a Google account to upload your prediction file there.**  If you cannot do so for any reason, feel free to reach out to us in the Slack channel so we can figure out alternatives.
  * You can choose not to resubmit; in this case, only the predictions of the remaining test records will be evaluated.
* The intermediate leaderboard result will be released on ~~December 16th 2021~~ December 22nd 2021.  There will be no real-time leaderboards.  If the same team made multiple edits or submitted multiple forms, only the last submission will be evaluated.

**11/11 Update:**
* **We now release the intermediate test set for [Dataset A](https://data.dgl.ai/dataset/WSDMCup2022/intermediate/input_A.csv.gz) and [Dataset B](https://data.dgl.ai/dataset/WSDMCup2022/intermediate/input_B.csv.gz).**

**11/10 Update:** We have updated the initial test set for Dataset B so that the nodes not appearing in the training set are removed.  The resulting number of test examples is 5,074.

**11/4 Update:** We have released an initial test set for [Dataset A](https://data.dgl.ai/dataset/WSDMCup2022/input_A_initial.csv.gz) and [Dataset B](https://data.dgl.ai/dataset/WSDMCup2022/input_B_initial.csv.gz) for developing your solutions, as well as a [simple baseline](https://github.com/dglai/WSDM2022-Challenge).

#### Example

Say that an edge with type 0 from node 0 to node 1 will appear at timestamp 15000000:

| `src_id` | `dst_id` | `edge_type` | `timestamp` |
|:--------:|:--------:|:-----------:|:-----------:|
| 0        | 1        | 0           | 15000000    |

You should predict some probability close to 1 for the following query since the timestamp 15000000 is between 14000000 and 16000000:

| `src_id` | `dst_id` | `edge_type` | `start_time` | `end_time` |
|:--------:|:--------:|:-----------:|:------------:|:----------:|
| 0        | 1        | 0           | 14000000     | 16000000   |

However, you should predict some probability close to 0 for both test queries below:

| `src_id` | `dst_id` | `edge_type` | `start_time` | `end_time` |
|:--------:|:--------:|:-----------:|:------------:|:----------:|
| 0        | 1        | 0           | 13000000     | 14000000   |

| `src_id` | `dst_id` | `edge_type` | `start_time` | `end_time` |
|:--------:|:--------:|:-----------:|:------------:|:----------:|
| 0        | 1        | 0           | 16000000     | 17000000   |

### Competition Terms and Conditions

At the end of the challenge, each team is encouraged to open source the source code that was used to generate their final challenge solution under the MIT license. To be eligible for the leaderboard or prizes, winning teams are also required to submit papers describing their method to the WSDM Cup Workshop, and present their work at the workshop.  Refer to the "Call for Papers" section on the WSDM Cup 2022 webpage for more details.

Participants are allowed to participate only once, with no concurrent submissions or code sharing between the teams.  The same team can submit multiple times, with only the last submission being evaluated.

Participants are not allowed to use external datasets or pretrained models.

We welcome any kinds of model in this challenge, regardless of whether it is a deep learning model or some graph learning algorithm.

### Evaluation Criteria

~~We use Area Under ROC (AUC) as evaluation metric for both datasets, and use the *harmonic average* of the two AUCs as the score of the submission.  Specifically, let `AUC_A` and `AUC_B` be the AUC for Dataset A and Dataset B respectively, the final score is `2 / (1 / AUC_A + 1 / AUC_B)`~~

~~This is to encourage the submissions to work well on both tasks, instead of working extremely well on one while sacrificing the other.~~

**12/23 Update:** In order to balance the difficulty between two datasets, we have decided to change the ranking metric from the harmonic average: `2 / (1 / AUC_A + 1 / AUC_B)` to **the average of T-scores**, to better promote our goal on "encouraging the submissions to work well on both instead of working extremely well on one while sacrificing the other".  The T-score of a dataset is computed as

```
TScore = (AUC - mean(AUC)) / std(AUC) * 0.1 + 0.5
```

where `mean(AUC)` and `std(AUC)` represents the mean and standard deviation of AUC of all participants.  The score for ranking will be `(TScore_A + TScore_B) / 2`.

For reference, we also kept the original harmonic average of AUC in the leaderboard.

### Leaderboard for Intermediate Test Set

This is an excerpt from the complete leaderboard; for score computation details please refer [here](https://data.dgl.ai/dataset/WSDMCup2022/results.xlsx).  Note that this will not impact the ranking on the final test set and the prize.

| Team Name | AUC (Dataset A) | AUC (Dataset B) | Harmonic Average of AUC | Average of T/100  | Rank on Harmonic Average of AUC | Rank on Average of T/100 |
|:---------:|:---------------:|:---------------:|:-----------------------:|:-----------------:|:-------------------------------:|:------------------------:|
| IDEAS Lab UT | 0.582272719 | 0.872039558 | 0.698288604 | 0.813037237 | 1 | 1 |
| DIVE@TAMU | 0.495906631 | 0.756680212 | 0.599148453 | 0.585081611 | 2 | 2 |
| nothing here | 0.500971508 | 0.632476354 | 0.559095205 | 0.534215147 | 3 | 3 |
| Graphile | 0.52824279 | 0.501279235 | 0.51440792 | 0.523983091 | 8 | 4 |
| smallhand | 0.516251289 | 0.539024256 | 0.527392051 | 0.518700616 | 6 | 5 |
| /tmp/graph | 0.519933859 | 0.519243432 | 0.519588416 | 0.516306306 | 7 | 6 |
| HUST_D5 | 0.496201592 | 0.572936262 | 0.531815208 | 0.495548282 | 5 | 7 |
| HappyICES | 0.507947137 | 0.500995797 | 0.504447521 | 0.48357129 | 10 | 8 |
| TopoLab | 0.501385353 | 0.525133491 | 0.51298472 | 0.482389216 | 9 | 9 |
| 10000 Monkeys | 0.505363661 | 0.499368274 | 0.50234808 | 0.47764665 | 11 | 10 |
| AntGraph | 0.503390663 | 0.5 | 0.501689602 | 0.474041453 | 13 | 11 |
| zhang | 0.501189056 | 0.502050738 | 0.501619527 | 0.470678591 | 14 | 12 |
| neutrino | 0.500645569 | 0.501220483 | 0.500932861 | 0.469192943 | 15 | 13 |
| Tencent_2022 | 0.499652798 | 0.500876566 | 0.500263933 | 0.467054303 | 16 | 14 |
| MegaTron | 0.498004853 | 0.505898455 | 0.50192062 | 0.466247299 | 12 | 15 |
| marble | 0.49767288 | 0.499622357 | 0.498645713 | 0.462510404 | 18 | 16 |
| beauty | 0.495978453 | 0.503765895 | 0.499841844 | 0.461180372 | 17 | 17 |
| luozhhh | 0.502644132 | 0.453959254 | 0.477062822 | 0.449979073 | 20 | 18 |
| no_free_lunch | 0.438312773 | 0.681825615 | 0.533599918 | 0.434084762 | 4 | 19 |
| NodeInGraph | 0.47341036 | 0.5 | 0.486342019 | 0.414551355 | 19 | 20 |

### Schedule

| Date                         | Event                                                                        |
|:----------------------------:|:----------------------------------------------------------------------------:|
| Oct 15 2021                  | Website ready and training set available for download.                       |
| Nov 11 2021                  | Intermediate test set release and intermediate submission starts.            |
| ~~Dec 11 2021~~ Dec 20 2021                  | Intermediate submission ends.                                                |
| ~~Dec 16 2021~~ Dec 22 2021                  | Intermediate leaderboard result announcement.                                |
| ~~Dec 17 2021~~ Dec 23 2021                  | Final test set release and final submission starts.                          |
| Jan 20 2022                  | Final submission ends.                                                       |
| Jan 24 2022                  | Final leaderboard result announcement.                                       |
| Jan 25 2022                  | Invitations to top 3 teams for short papers.                                 |
| Feb 15 2022                  | Short paper deadline.                                                        |
| Feb 21-25 2022               | WSDM Cup conference presentation.                                            |

### Prizes

The prizes will be determined solely by the performance on the final test set.

* 1st place: $2,000 + one WSDM Cup conference registration
* 2nd place: $1,000 + one WSDM Cup conference registration
* 3rd place: $500 + one WSDM Cup conference registration

We would like to thank Intel for kindly sponsoring this event.

### Support or Contact

If you have questions or need clarifications, feel free to join the channel **wsdm22-challenge** in [DGL's Slack workspace](https://join.slack.com/t/deep-graph-library/shared_invite/zt-eb4ict1g-xcg3PhZAFAB8p6dtKuP6xQ).

### Links

WSDM call for cup proposals: https://www.wsdm-conference.org/2022/call-for-wsdm-cup-proposals/
