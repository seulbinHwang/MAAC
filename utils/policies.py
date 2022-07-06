import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import onehot_from_logits, categorical_sample

class BasePolicy(nn.Module):
    """
    Base policy network
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 norm_in=True, onehot_dim=0):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(BasePolicy, self).__init__()

        if norm_in:  # normalize inputs
            # BatchNorm1d: (10, 2, 180) 에서, data_scan_dist 끼리 (10,180)에 대해 batchnorm 을 실시함
            self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
        else:
            # lambda 함수를 한줄로 표현해주는 것
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim + onehot_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations (optionally a tuple that
                                additionally includes a onehot label)
        Outputs:
            out (PyTorch Matrix): Actions
        """
        onehot = None
        if type(X) is tuple:
            X, onehot = X # observation / one_hot
        inp = self.in_fn(X)  # don't batchnorm onehot
        if onehot is not None:
            inp = torch.cat((onehot, inp), dim=1) # onehot
        h1 = self.nonlin(self.fc1(inp))
        h2 = self.nonlin(self.fc2(h1))
        out = self.fc3(h2)
        return out


class DiscretePolicy(BasePolicy):
    """
    Policy Network for discrete action spaces
    """
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, obs, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False):
        out = super(DiscretePolicy, self).forward(obs) # out_dim # ( , 5)
        probs = F.softmax(out, dim=1) # 확률 값들 # (, 5)
        on_gpu = next(self.parameters()).is_cuda
        if sample:
            int_act, act = categorical_sample(probs, use_cuda=on_gpu) # int_act: 고른 action (_, 1) / act: one hot (_, 5)
        else:
            # (, 5)
            act = onehot_from_logits(probs)
        rets = [act] # (_, 5)
        if return_log_pi or return_entropy:
            log_probs = F.log_softmax(out, dim=1)  # (_, 5)
        if return_all_probs:
            rets.append(probs) # (_, 5)
        if return_log_pi:
            # return log probability of selected action
            rets.append(log_probs.gather(1, int_act)) # (_, 1)
        if regularize:
            rets.append([(out**2).mean()]) # ([value])
        if return_entropy:
            rets.append(-(log_probs * probs).sum(1).mean())
        if len(rets) == 1:
            return rets[0]
        return rets
