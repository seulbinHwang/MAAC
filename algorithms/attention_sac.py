import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import AttentionAgent
from utils.critics import AttentionCritic

MSELoss = torch.nn.MSELoss()

class AttentionSAC(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent
    task
    """
    def __init__(self, agent_init_params, sa_size,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.,
                 pol_hidden_dim=128,
                 critic_hidden_dim=128, attend_heads=4,
                 **kwargs):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
            sa_size (list of (int, int)): Size of state and action space for
                                          each agent -> sa_size.append((obsp.shape[0], acsp.n))
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            reward_scale (float): Scaling for reward (has effect of optimal
                                  policy entropy)
            hidden_dim (int): Number of hidden dimensions for networks
        """
        self.nagents = len(sa_size)
        # agent_init_params: [{'num_in_pol': obsp.shape[0], 'num_out_pol': acsp.n} , ... * n]
        # discrete AttentionAgent : actor 여러 개
        self.agents = [AttentionAgent(lr=pi_lr,
                                      hidden_dim=pol_hidden_dim,
                                      **params)
                         for params in agent_init_params]
        # sa_size -> [(obsp.shape[0], acsp.n), ... ]
        self.critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                      attend_heads=attend_heads)
        self.target_critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                             attend_heads=attend_heads)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,
                                     weight_decay=1e-3)
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        # critic에 사용
        return [a.target_policy for a in self.agents]

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                               observations)]

    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
            - obs: [1024, 86] * 8
            - acs: [1024, 5] * 8
            - rews: [1024] * 8 -> reward를 normalized 했다.
            - next_obs: [1024, 86] * 8
            - dones: [1024] * 8
        """
        obs, acs, rews, next_obs, dones = sample
        # Q loss
        next_acs = [] # [(1024, 5)] * 8
        next_log_pis = [] # [(1024, 1)] * 8
        # loop per agent = 8
        for pi, ob in zip(self.target_policies, next_obs):
            # pi: target policy: DiscretePolicy,  ob: [1024, 86]
            # curr_next_ac: (1024, 5) onehot / curr_next_log_pi: (1024, 1)
            curr_next_ac, curr_next_log_pi = pi(ob, return_log_pi=True)
            next_acs.append(curr_next_ac) # at+1_new
            next_log_pis.append(curr_next_log_pi) # at+1_new 이 나올 log 확률
        trgt_critic_in = list(zip(next_obs, next_acs)) # [ (1024, 86), (1024, 5) ] * 8 -> st+1, at+1_new
        critic_in = list(zip(obs, acs)) # -> [ (1024, 86), (1024, 5) ] * 8 -> st, at

        # next_qs: [(1024, 1)] * 8 # Q(st+1)을 구한 후, at+1_new 에 해당하는 값을 도출한다.
        next_qs = self.target_critic(trgt_critic_in) # Q(st+1, at+1_new)

        # critic_rets : [[current_qs(1024, 1), attend_mag_reg (value)] * 8]
        # current_qs Q(st)을 구한 후, at 에 해당하는 값을 도출한다.
        # attend_mag_reg : attention logit 을 제곱 평균해서 합친 값 / 1000
        critic_rets = self.critic(critic_in, regularize=True,
                                  logger=logger, niter=self.niter) # [pq: Q(st,at), regs: attention logit regularize]
        q_loss = 0
        for a_i, nq, log_pi, (pq, regs) in zip(range(self.nagents), next_qs,
                                               next_log_pis, critic_rets):
            """
            a_i: 0, 1, ,,, -> 7
            nq: agent per Q(st+1, at+1_new) -> (1024, 1)
            log_pi: agent per at+1_new 이 나올 확률 -> (1024, 1) 
            pq: agent per Q(st, at) -> (1024, 1)
            regs: agent per attend_mag_reg (attention logit 을 제곱 평균해서 합친 값 / 1000) -> (value)
            
            rews: [1024] * 8 -> reward를 normalized 했다.
            """
            # r(1024, 1) + gamma * Q(st+1, at+1_new) (1024, 1) = (1024, 1)
            target_q = (rews[a_i].view(-1, 1) +
                        self.gamma * nq *
                        (1 - dones[a_i].view(-1, 1))) # Q(st+1, at+1_new)
            if soft:
                # log_pi: (1024, 1)
                # self.reward_scale = 100
                # entropy_coefficient = 1 / self.reward_scale
                target_q -= log_pi / self.reward_scale # SAC
            q_loss += MSELoss(pq, target_q.detach()) # Q(st, at) 와 target 의 차
            for reg in regs:
                # q_loss 에 reg를 더한다.
                q_loss += reg  # attention_logit 의 특정 값만 크지 않게 regularize 하려는 시도
        q_loss.backward()
        # 8 마리 agent 에 대해 backward를 다 한 후, grad / num_agent 를 한다.
        self.critic.scale_shared_grads()
        # gradient exploding을 방지하여 학습의 안정화를 도모하기 위해 사용하는 방법이다.
        grad_norm = torch.nn.utils.clip_grad_norm(
            self.critic.parameters(), 10 * self.nagents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1

    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        """
        :param sample:
        :param soft:
        :param logger:
        :param kwargs:
        :return:

            - obs: [1024, 86] * 8
            - acs: [1024, 5] * 8
            - rews: [1024] * 8 -> reward를 normalized 했다.
            - next_obs: [1024, 86] * 8
            - dones: [1024] * 8
        """
        obs, acs, rews, next_obs, dones = sample
        samp_acs = []
        all_probs = []
        all_log_pis = []
        all_pol_regs = []

        for a_i, pi, ob in zip(range(self.nagents), self.policies, obs):
            """
            a_i: 0 -> 1 -> --> 7
            pi: DiscretePolicy
            ob: [1024, 86]
            """
            """
            curr_ac: (1024, 5) : onehot -----> samp_acs [(1024, 5)] * 8  at_new onehot
            probs: (1024, 5): all probs  -----> all_probs [(1024, 5)] * 8 모든 at_new 의 확률
            log_pi: (1024, 1): log selected probs -----> all_log_pis [(1024, 1)] * 8 at_new 의 log 확률
            pol_regs: policy actor logit (1024, 5)의 제곱 평균 -> [(1024)] -----> all_pol_regs [(1024, 5)] * 8
            ent: (value): entropy
            """
            curr_ac, probs, log_pi, pol_regs, ent = pi(
                ob, return_all_probs=True, return_log_pi=True,
                regularize=True, return_entropy=True)
            logger.add_scalar('agent%i/policy_entropy' % a_i, ent,
                              self.niter)
            samp_acs.append(curr_ac)
            all_probs.append(probs)
            all_log_pis.append(log_pi)
            all_pol_regs.append(pol_regs)

        critic_in = list(zip(obs, samp_acs)) # critic_in: [ [1024, 86], (1024, 5) ] * 8 (st, at_new)
        # [q(1024, 1) , all_q(1024, 5) ] * 8
        critic_rets = self.critic(critic_in, return_all_q=True)
        for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(range(self.nagents), all_probs,
                                                            all_log_pis, all_pol_regs,
                                                            critic_rets):
            """
            a_i: 0 -> 1 -> --> 7
            probs: (1024, 5) 모든 at_new 의 확률
            log_pi: (1024, 1) at_new 의 log 확률
            pol_regs: policy actor logit (1024, 5)의 제곱 평균 -> [value]
            q: (1024, 1) Q(st, at_new)
            all_q:  (1024, 5) Q(st)
            """
            curr_agent = self.agents[a_i]
            # Q(st) (1024, 5) * (1024, 5) 모든 at_new 의 확률 => (1024, 1)
            v = (all_q * probs).sum(dim=1, keepdim=True)
            #  Q(st, at_new) (1024, 1) - (1024, 1)
            pol_target = q - v
            if soft:
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean() # 상수
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()
            for reg in pol_regs: # policy actor logit (1024, 5)의 제곱 평균 -> [value]
                pol_loss += 1e-3 * reg  # policy regularization
            # don't want critic to accumulate gradients from policy loss
            disable_gradients(self.critic) # requires_grad = False
            pol_loss.backward() #
            enable_gradients(self.critic) # requires_grad = True

            # gradient exploding을 방지하여 학습의 안정화를 도모하기 위해 사용하는 방법이다.
            grad_norm = torch.nn.utils.clip_grad_norm(
                curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

            if logger is not None:
                logger.add_scalar('agent%i/losses/pol_loss' % a_i,
                                  pol_loss, self.niter)
                logger.add_scalar('agent%i/grad_norms/pi' % a_i,
                                  grad_norm, self.niter)


    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)

    def prep_training(self, device='gpu'):
        """
        :param device:
        :return:

        Objectives
            - policy / target_policy / critic / target_critic 모두 train()
            - gpu를 사용하도록 설정해줌
        """
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()

        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device

        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        """
        :param device:
        :return:

        Objectives
            - actor의 policy를 eval 상태로 변경
            - actor을 사용하는 장치가 바뀌었을 때, 이를 적용시킨다.
        """
        for a in self.agents: # each actor
            a.policy.eval() # each.actor > discrete policy
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device: # gpu 일떄
            for a in self.agents:
                a.policy = fn(a.policy) #
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01,
                      reward_scale=10.,
                      pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4,
                      **kwargs):
        """
        Instantiate instance of this class from multi-agent environment

        env: Multi-agent Gym environment
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks

        classmethod
            - 클래스를 instance 화 하지 않아도 호출이 가능
            - static method 와 차이점이라면
                - 다른 method 및 class 속성에 access 가능
        """
        agent_init_params = []
        sa_size = []
        for acsp, obsp in zip(env.action_space,
                              env.observation_space):
            agent_init_params.append({'num_in_pol': obsp.shape[0],
                                      'num_out_pol': acsp.n})
            sa_size.append((obsp.shape[0], acsp.n))

        init_dict = {'gamma': gamma, 'tau': tau,
                     'pi_lr': pi_lr, 'q_lr': q_lr,
                     'reward_scale': reward_scale,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'attend_heads': attend_heads,
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance