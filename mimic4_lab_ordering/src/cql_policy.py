"""
Picklable pieces of the CQL arm: the Q-network and the greedy policy wrapper.

These live here rather than in s4b_train_cql.py for one concrete reason. Pickle
stores a class by module path, and a class defined in a script that runs as
`__main__` is recorded as `__main__.GreedyQPolicy`. Stages 5, 6 and 7 are
different `__main__` modules, so unpickling there fails with

    AttributeError: Can't get attribute 'GreedyQPolicy' on <module '__main__' ...>

Defining them in an ordinary importable module records `cql_policy.GreedyQPolicy`,
which resolves anywhere that has src/ on sys.path.
"""
import numpy as np
import torch
import torch.nn as nn

import config as cfg

N_ACTIONS = 2


def scalarize(reward, weights=None):
    """Collapse the 4-vector reward into the single number CQL trains on.

    This is where the multi-objective claim is given up, so it lives in one
    function and both the trainer and the evaluator call it. Evaluating a CQL
    policy against the 4-vector reward would score it on objectives it never
    optimized; stage 5 therefore uses this for the CQL arm.
    """
    import numpy as _np
    w = weights or cfg.REWARD_WEIGHTS
    vec = _np.array([w[d] for d in cfg.REWARD_DIMS], dtype=_np.float64)
    return (reward.astype(_np.float64) * vec).sum(axis=1).astype(_np.float32)


class QNet(nn.Module):
    """Scalar Q, one value per action.

    The output layer is zero-initialized so training starts from Q == 0 rather
    than from noise: random initial values propagate through the Bellman target
    and can settle the fit at the wrong sign entirely. Same reasoning as the FQE
    network in stage 5.
    """

    def __init__(self, state_dim, hidden=cfg.CQL_HIDDEN, n_actions=N_ACTIONS):
        super().__init__()
        self.n_actions = int(n_actions)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, self.n_actions)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, s):
        return self.head(self.trunk(s))


class GreedyQPolicy:
    """argmax over the learned Q, exposing the same `.predict(states)` signature
    as the extra-trees classifier the MO-FQI arm saves, so stages 5 to 7 need no
    special-casing.

    Carries the normalization statistics because the network trains on
    standardized states while every caller passes raw ones, and a `bias` added to
    Q(order) before the argmax, which is how the order count gets calibrated.
    """

    def __init__(self, net, mu, sd, bias=0.0):
        self.net = net
        self.mu = mu
        self.sd = sd
        self.bias = bias

    def q_values(self, states):
        z = (np.asarray(states, dtype=np.float32) - self.mu) / self.sd
        self.net.eval()
        with torch.no_grad():
            q = self.net(torch.tensor(z, dtype=torch.float32)).numpy()
        if np.isscalar(self.bias):
            if q.shape[1] > 1:
                q[:, 1] += float(self.bias)
        else:
            q += np.asarray(self.bias, dtype=np.float32)
        return q

    def predict(self, states):
        q = self.q_values(states)
        return q.argmax(axis=1).astype(np.int64)
