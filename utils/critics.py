import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain


class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent -> [(obsp.shape[0], acsp.n), ... ] = 8
            hidden_dim (int): Number of hidden dimensions = 128
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by) = 4
        """
        super(AttentionCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList() # encoder * n 마리
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterate over agents
        for sdim, adim in sa_sizes:
            idim = sdim + adim
            odim = adim
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)

            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
                 - [ (1024, 86), (1024, 5) ] * 8 -> st+1, at+1_new

            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        # next_qs = self.target_critic(trgt_critic_in) # Q(st+1, at+1_new)
        if agents is None:
            agents = range(len(self.critic_encoders)) # [0, 1, 2, 3, 4, 5, 6, 7]
        states = [s for s, a in inps] # [ (1024, 86)] * 8 -> st+1
        actions = [a for s, a in inps] # [ (1024, 5) ] * 8 -> at+1_new
        inps = [torch.cat((s, a), dim=1) for s, a in inps] # [(1024, 86+5)] * 8 # st+1, at+1_new

        # extract state-action encoding for each agent
        # input: st+1, at+1_new
        # sa_encodings: [(1024, hidden_dim=128 )] * 8
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]

        # extract state encoding for each agent that we're returning Q for
        # input: st+1
        # s_encodings: [(1024, hidden_dim=128)] * 8
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]

        # extract keys for each head for each agent
        # 흐름: st+1, a_t+1 -> [critic_encoders] -> [key_extractors]
        # all_head_keys = [ [ (1024, 128/4) ]*8 ] * 4
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]

        # extract sa values for each head for each agent
        # 흐름: st+1, a_t+1 -> [critic_encoders] -> [value_extractors]
        # all_head_values = [ [ (1024, 128/4) ]*8 ] * 4
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]

        # extract selectors for each head for each agent that we're returning Q for
        # state_encoder 의 출력값을 인풋으로 넣는다. (selector_extractors 에)
        # 흐름: st+1 -> [state_encoders] -> [selector_extractors]
        # all_head_selectors = [ [ (1024, 128/4) ]*8 ] * 4
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors] #  selector_extractors = query

        other_all_values = [[] for _ in range(len(agents))] # [ [ (1024, 128/4) ] *4]*8
        all_attend_logits = [[] for _ in range(len(agents))] # [ [ (1024, 1, 7) ] *4]*8
        all_attend_probs = [[] for _ in range(len(agents))] # [ [ (1024, 1, 7) ] *4]*8

        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # curr_head_keys: [ (1024, 128/4) ]*8
            # curr_head_values: [ (1024, 128/4) ]*8
            # curr_head_selectors:  [ (1024, 128/4) ]*8

            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                # i : 0 -> 1 -> -> 7
                # a_i: 0 -> 1 -> -> 7
                # selector: (1024, 128/4)
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i] # 나를 제외한 agent -> [ (1024, 128/4) ]*7
                values = [v for j, v in enumerate(curr_head_values) if j != a_i] # 나를 제외한 agent -> [ (1024, 128/4) ]*7
                # calculate attention across agents
                # 흐름: selector(query) * keys
                # selector (1024, 1, 128/4) * keys (7, 1028, 128/4) -> (1024, 128/4, 7) = attend_logits ([1024, 1, 7])
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1]) # ([1024, 1, 7]) # 32로 나눴다.
                attend_weights = F.softmax(scaled_attend_logits, dim=2) # ([1024, 1, 7]) #

                # value * attention score
                #  (1024, 128/4, 7) * [1024, 1, 7] = (1024, 128/4, 7) --(sum)--> (1024, 128/4)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2) # (1024, 128/4)
                other_all_values[i].append(other_values) # (1024, 128/4)
                all_attend_logits[i].append(attend_logits) # (1024, 1, 7)
                all_attend_probs[i].append(attend_weights) # (1024, 1, 7)


        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            # i, ai: 0-> 1 -> ... -> 7
            # attention weight 에 대한 엔트로피
            # all_attend_probs : [ [ (1024, 1, 7) ] *4 ]*8
            # probs: [ (1024, 1, 7) ]
            # head_entropies -> [값] * 4
            head_entropies = [ ( -( (probs + 1e-8).log() * probs ).squeeze().sum(1)
                               .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            # 내 state encoding과 (나와 상대방의 관계를 함축한 embeddings) 를 concatenate
            # s_encodings: [(1024, hidden_dim=128)] * 8
            # other_all_values: [ [ (1024, 128/4) ] *4] * 8
            # (1024, hidden_dim=128) + (1024, 128/4) ] *4 = (1024, 128 + 128/4*4) = (1024, 256)
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1) # (1024, 256)
            all_q = self.critics[a_i](critic_in) # all_q: (1024, 5)
            # int_acs: (1024, 1) -> action의 index 정보
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1] # actions: [ (1024, 5) ] * 8 -> at+1_new
            q = all_q.gather(1, int_acs) # (1024, 5) -> (1204, 1)
            if return_q:
                agent_rets.append(q) # [(1024, 1)]
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                # regularize magnitude of attention logits ( selector(query) * keys ) # [[] for _ in range(len(agents))]
                # all_attend_logits: [ [ (1024, 1, 7) ] *4]*8
                # logit (1024, 1, 7)  이 num_head 개 씩 있는데, 각각을 제곱한 후 모두 더해서 10*(-3) 을 곱해줍니다.
                # attend_mag_reg: value
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,) # attention logit을 regularize 하려는 시도 (value)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))
            if logger is not None:
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0] # (1024, 1)
        else:
            return all_rets
