import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC


def make_parallel_env(env_id, n_rollout_threads, seed):
    """
    :param env_id: config.env_id : fullobs_collect_treasure
    :param n_rollout_threads: config.n_rollout_threads = 12
    :param seed: run_num
    :return:

    DummyVecEnv
        - 환경 만들고, 환경에 대한 seed 와 np seed를 설정한다.
        - agent_types : 'adversary(적)' / 'agent'

    SubprocVecEnv
        - n_rollout_threads 개 만큼의 Process 를 만든다. -> env = cloudpickle.dumps(init_env)
    """
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=True)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    """
    :param config:
    :return:

    Objectives
        - env_id: fullobs_collect_treasure
        - model_dir:
    """
    model_dir = Path('./models') / config.env_id / config.model_name
    print('model_dir:', model_dir)
    if not model_dir.exists():
        # 해당 폴더를 처음 만든다.
        run_num = 1
    else:
        # 이미 해당 폴더가 만들어 져 있으면, RUN이 몇 번째까지 있었는지 확인해서, exst_run_nums 리스트로 만든다.
        # ex. 4번째 run을 실행하면 ->  exst_run_nums = [1, 2, 3]
        print('model_dir.iterdir():', model_dir.iterdir())
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    # tensorboardX
    logger = SummaryWriter(str(log_dir))

    # 정해진 seed
    torch.manual_seed(run_num)
    # random seed
    np.random.seed(run_num)
    """
        DummyVecEnv
            - 환경 만들고, 환경에 대한 seed 와 np seed를 설정한다.
            - agent_types : 'adversary(적)' / 'agent'
    
        SubprocVecEnv
            - n_rollout_threads 개 만큼의 Process 를 만든다. -> env = cloudpickle.dumps(init_env)
    """
    env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num)
    # model instance 생성
    """
    - model.agents = [AttentionAgent] * agent 수(=n_rollout_threads)
        - AttentionAgent
            - self.policy : DiscretePolicy (num_in_pol, num_out_pol)
                - nn.BatchNorn1d
                - nn.Linear(num_in_pol, hidden_dim) + F.leaky_relu
                - nn.Linear(hidden_dim, hidden_dim) + F.leaky_relu
                - nn.Linear(hidden_dim, num_out_pol)
                
            - self.target_policy : DiscretePolicy
            
    - model.critic, model.target_critic: AttentionCritic 1개
        - AttentionCritic
            - self.critic_encoders : ModuleList ------> shared_modules
                - encoder: Sequential * agent 수(=nagents)
                    - enc_bn : nn.BatchNorm1d (idim=sdim+adim)
                    - enc_fc1 : nn.Linear (idim, hidden_dim)
                    - enc_nl : nn.LeakyReLU
                    
            - self.critics : ModuleList
                - critic : Sequential * agent 수(=nagents)
                    - critic_fc1 : nn.linear(2 * hidden_dim, hidden_dim)
                    - critic_nl : nn.LeakyReLU
                    - critic_fc2 : nn.linear(hidden_dim, (odim=adim) )
            
            #########    
            - self.state_encoders : ModuleList
                - state_encoder: Sequential * agent 수(=nagents)
                    - s_enc_bn : nn.BatchNorm1d (sdim)
                    - s_enc_fc1 : nn.Linear(sdim, hidden_dim)
                    - s_enc_nl : nn.LeakyReLU()
            
            - self.key_extractors : ModuleList ------> shared_modules
                - nn.Linear(hidden_dim, attend_dim) (* attend_heads 만큼)
            
            - self.selector_extractors : ModuleList ------> shared_modules
                - nn.Linear(hidden_dim, attend_dim) (* attend_heads 만큼)
                
            - self.value_extractors : ModuleList ------> shared_modules
                - nn.Linear(hidden_dim, attend_dim) (* attend_heads 만큼)

    """
    model = AttentionSAC.init_from_env(env,
                                       tau=config.tau,
                                       pi_lr=config.pi_lr,
                                       q_lr=config.q_lr,
                                       gamma=config.gamma,
                                       pol_hidden_dim=config.pol_hidden_dim,
                                       critic_hidden_dim=config.critic_hidden_dim,
                                       attend_heads=config.attend_heads,
                                       reward_scale=config.reward_scale)
    # buffer_length: 100,000
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    # n_episodes = 50000
    # episode_length = 12
    # n_rollout_threads = 12
    # nagents = 8
    # epi_i: 0, 12, 24, 36 , ...
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        # ep_i: 0, 12, 24, 36,
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset() # (12, 8)
        model.prep_rollouts(device='cpu')

        for et_i in range(config.episode_length): # 25 -> 0~24
            # rearrange observations to be per agent, and convert to torch Variable
            # torch_obs = [(12,86)] * nagents 8
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(model.nagents)]
            # get actions as torch Variables
            # torch_agent_actions, agent_actions: [(12, 5)] * 8
            torch_agent_actions = model.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            # actions : [[(5)] * 8] * 12
            actions = [ [ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]

            # next_obs , rewards, dones : [(8)] * 12
            # infos: {'n': [{}, {}, {}, {}, {}, {}, {}, {}]} * 12
            next_obs, rewards, dones, infos = env.step(actions)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads # 0, 12, 24, 36 ,,,,
            # batch_size 보다 replay_buffer이 크고, steps_per_update = 100
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config.num_updates): # 0, 1, 2, 3
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=config.use_gpu)
                    """
                    sample: 길이 5 list ( obs_buffs / ac_buffs / ret_rews / next_obs_buffs / done_buffs)
                        - obs_buffs: [1024, 86] * 8
                        - ac_buffs: [1024, 5] * 8
                        - ret_rews: [1024] * 8 -> reward를 normalized 했다.
                        - next_obs_buffs: [1024, 86] * 8
                        - done_buffs: [1024] * 8
                    """
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')
        # episode 동안 모든 env의 step 수 합인 25 * 12 에 대한 평균 reward
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i,
                              a_ep_rew * config.episode_length, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')

    model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--n_rollout_threads", default=12, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=50000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')

    config = parser.parse_args()

    run(config)
