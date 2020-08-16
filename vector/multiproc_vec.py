import multiprocessing as mp
import numpy as np
import traceback
import multiprocessing as mp
import gym

class SharedArray:
    def __init__(self, shape, dtype):
        self.shared_arr = mp.Array(np.ctypeslib.as_ctypes_type(dtype),int(np.prod(shape)),lock=False)
        self.dtype = dtype
        self.shape = shape
        self._set_np_arr()

    def _set_np_arr(self):
        self.np_arr = np.frombuffer(self.shared_arr, dtype=self.dtype).reshape(self.shape)

    def __getstate__(self):
        return (self.shared_arr,self.dtype,self.shape)

    def __setstate__(self, state):
        self.shared_arr = state
        self._set_np_arr()

class SpaceWrapper:
    def __init__(self, space):
        if isinstance(space, gym.spaces.Discrete):
            self.shape = ()
            self.dtype = np.dtype(np.int32)
        elif isinstance(space, gym.spaces.Box):
            self.shape = space.shape
            self.dtype = np.dtype(space.dtype)
        else:
            assert False, "ProcVectorEnv only support Box and Discrete types"


def compress_info(infos):
    non_empty_infs = [(i,info) for i,info in enumerate(infos) if info]
    return non_empty_infs


def decompress_info(num_envs, idx_starts, comp_infos):
    all_info = [{}]*num_envs
    for idx_start, comp_infos in zip(idx_starts, comp_infos):
        for i,info in comp_infos:
            all_info[idx_start+i] = info
    return all_info


def async_loop(vec_env_constr, pipe, shared_obs, shared_actions, shared_rews, shared_dones):
    try:
        vec_env = vec_env_constr()

        pipe.send((vec_env.num_envs))
        env_start_idx = pipe.recv()
        env_end_idx = env_start_idx + vec_env.num_envs
        while True:
            instr = pipe.recv()
            if instr == "reset":
                obs = vec_env.reset()
                shared_obs.np_arr[env_start_idx:env_end_idx] = obs
                shared_dones.np_arr[env_start_idx:env_end_idx] = False
                shared_rews.np_arr[env_start_idx:env_end_idx] = 0.
                comp_infos = []
            elif instr == "step":
                actions = shared_actions.np_arr[env_start_idx:env_end_idx]
                observations, rewards, dones, infos = vec_env.step(actions)
                shared_obs.np_arr[env_start_idx:env_end_idx] = observations
                shared_dones.np_arr[env_start_idx:env_end_idx] = dones
                shared_rews.np_arr[env_start_idx:env_end_idx] = rewards
                comp_infos = compress_info(infos)
            elif instr == "terminate":
                return
            pipe.send(comp_infos)
    except BaseException as e:
        tb = traceback.format_exc()
        pipe.send((e,tb))

class ProcConcatVec:
    def __init__(self, vec_env_constrs, observation_space, action_space, tot_num_envs):
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs = tot_num_envs

        self.shared_obs = SharedArray((num_envs,)+self.observation_space.shape, dtype=self.observation_space.dtype)
        act_space_wrap = SpaceWrapper(self.action_space)
        self.shared_act = SharedArray((num_envs,)+act_space_wrap.shape, dtype=act_space_wrap.dtype)
        self.shared_rews = SharedArray((num_envs,), dtype=np.float32)
        self.shared_dones = SharedArray((num_envs,), dtype=np.bool)

        pipes = []
        procs = []
        for constr in vec_env_constrs:
            inpt,outpt = mp.Pipe()
            proc = mp.Process(target=async_loop, args=(constr, outpt, self.shared_obs, self.shared_act, self.shared_rews, self.shared_dones))
            proc.start()
            pipes.append(inpt)
            procs.append(proc)

        self.pipes = pipes
        self.procs = procs

        num_envs = 0
        env_nums = self._receive_info()
        idx_starts = []
        for pipe,cnum_env in zip(self.pipes,env_nums):
            cur_env_idx = num_envs
            num_envs += cnum_env
            pipe.send(cur_env_idx)
            idx_starts.append(cur_env_idx)

        assert num_envs == tot_num_envs
        self.idx_starts = idx_starts

    def reset(self):
        for pipe in self.pipes:
            pipe.send("reset")

        self._receive_info()

        observations = self.shared_obs.np_arr
        return observations

    def step_async(self, actions):
        self.shared_act.np_arr[:] = actions
        for pipe in self.pipes:
            pipe.send("step")

    def _receive_info(self):
        all_data = []
        for cin in self.pipes:
            data = cin.recv()
            if isinstance(data, tuple):
                e, tb = data
                print(tb)
                raise e
            all_data.append(data)
        return all_data

    def step_wait(self):
        compressed_infos = self._receive_info()
        infos = decompress_info(self.num_envs, self.idx_starts, compressed_infos)
        observations = self.shared_obs.np_arr
        rewards = self.shared_rews.np_arr
        dones = self.shared_dones.np_arr
        return observations, rewards, dones, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def __del__(self):
        for pipe in self.pipes:
            try:
                pipe.send("terminate")
            except BrokenPipeError:
                pass
        for proc in self.procs:
            proc.join()
