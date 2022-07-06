"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrapper):
    """
    :param remote:
    :param parent_remote:
    :param env_fn_wrapper: CloudpickleWrapper(init_env)
        - Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    :return:

    Env_fn_wrapper
        - init_env
            - env = make_env(env_id, discrete_action=True)
            - env.seed(seed + rank * 1000)
            - np.random.seed(seed + rank * 1000)

    Objectives
        - env = cloudpickle.dumps(init_env)
    """
    parent_remote.close()
    env = env_fn_wrapper.x() # cloudpickle.dumps(init_env)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if all(done):
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_agent_types':
            if all([hasattr(a, 'adversary') for a in env.agents]):
                remote.send(['adversary' if a.adversary else 'agent' for a in
                             env.agents])
            else:
                remote.send(['agent' for _ in env.agents])
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses

        env_fns: [ init_env , init_env , init_env , init_env , init_env ,  ... n_rollout_threads 개]
            - init_env
                - env = make_env(env_id, discrete_action=True)
                - env.seed(seed + rank * 1000)
                - np.random.seed(seed + rank * 1000)

        Objectives
            - n_rollout_threads 개 만큼의 Process 를 만든다. -> env = cloudpickle.dumps(init_env)

        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns) # 12
        # Pipe -> 파이프로 연결된 한 쌍의 연결 객체를 돌려주는데 기본적으로 양방향(duplex)입니다.
        # Pipe() 가 반환하는 두 개의 연결 객체는 파이프의 두 끝을 나타냅니다.
        # 각 연결 객체에는 (다른 것도 있지만) send() 및 recv() 메서드가 있습니다.
        # 두 프로세스 (또는 스레드)가 파이프의 같은 끝에서 동시에 읽거나 쓰려고 하면 파이프의 데이터가 손상될 수 있습니다.
        # 물론 파이프의 다른 끝을 동시에 사용하는 프로세스로 인해 손상될 위험은 없습니다.
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        # n_rollout_threads 개 만큼의 Process 를 만든다. -> env = cloudpickle.dumps(init_env)
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        """
        n_rollout_threads 가 1개 일 때만 DummyVecEnv 를 쓴다.

        :param env_fns: [init_env]
        def init_env():
            env = make_env(env_id, discrete_action=True)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env

        Objectives
            - 환경 만들고, 환경에 대한 seed 와 np seed를 설정한다.
            - agent_types : 'adversary(적)' / 'agent'
        """
        self.envs = [fn() for fn in env_fns]

        # 환경 만들고, 환경에 대한 seed 와 np seed를 설정한다.
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        # agent_types : 'adversary(적)' / 'agent'
        if all([hasattr(a, 'adversary') for a in env.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                env.agents]
        else:
            self.agent_types = ['agent' for _ in env.agents]
        self.ts = np.zeros(len(self.envs), dtype='int')        
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.ts += 1
        for (i, done) in enumerate(dones):
            if all(done): 
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):        
        results = [env.reset() for env in self.envs]
        return np.array(results)

    def close(self):
        return