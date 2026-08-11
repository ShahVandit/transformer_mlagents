# CQL training: wbc

**Not part of Cheng et al. (2019).** An alternative learner on the same MDP, run with `--learner cql`.

`alpha=1.0` `steps=4,000` `gamma=0.9` `lr=0.0001` `batch=1024`

## Scalarized reward

CQL is scalar-Q, so the 4-vector reward is collapsed by a fixed weighting:

| component | weight |
|---|---|
| r_sofa | 1.0 |
| r_treat | 1.0 |
| r_info | 1.0 |
| neg_r_cost | 1.0 |

This is the step the paper's vector-valued Q and Pareto pruning exist to avoid. With a scalarization the preference among objectives is fixed by hand up front, so results from this arm are not a multi-objective claim and should not be presented as one.

## Training

| step | loss | td_loss | conservative_loss | q_mean |
|---|---|---|---|---|
| 1 | 0.7182 | 0.0251 | 0.6931 | 0.0000 |
| 2,000 | 0.3924 | 0.0980 | 0.2944 | -0.4535 |
| 4,000 | 0.3514 | 0.0828 | 0.2686 | -0.3198 |

`conservative_loss` is `logsumexp_a Q(s,a) - Q(s,a_behavior)`. With a binary action space and both actions well represented at every hour, there is little out-of-distribution room for the Q-function to hallucinate into, so this term is expected to stay small. A near-inert conservative penalty here is a property of the action space, not a failure of the implementation.

## Order-count calibration (val)

Raw argmax order rate 0.0000 against a clinician rate of 0.0686. A bias of +0.58460 is added to `Q(order)` and bisected until the counts match (13,795 vs 13,795), which is the CQL analogue of the paper's `eps_cost` tuning and puts both arms at a comparable ordering volume.
