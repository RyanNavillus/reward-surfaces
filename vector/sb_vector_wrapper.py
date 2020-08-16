from stable_baselines.common.vec_env.base_vec_env import VecEnv
import warnings

class VecEnvWrapper(VecEnv):
    def __init__(self, rlflow_venv):
        self.rlflow_venv = rlflow_venv
        self.num_envs = rlflow_venv.num_envs
        self.observation_space = rlflow_venv.observation_space
        self.action_space = rlflow_venv.action_space

    def reset(self):
        return self.rlflow_venv.reset()

    def step_async(self, actions):
        self.rlflow_venv.step_async(actions)

    def step_wait(self):
        return self.rlflow_venv.step_wait()

    def step(self, actions):
        return self.rlflow_venv.step(actions)

    def close(self):
        del self.rlflow_venv

    def seed(self, seed):
        if seed is not None:
            warnings.warn("tried to seed environment, not possible with rlflow's vector environments")

    def get_attr(self, attr_name, indices=None):
        raise NotImplementedError()

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError()
