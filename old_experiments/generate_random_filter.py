import numpy as np
from .extract_params import ParamLoader
import os
#from utils import ALGOS
#from test_env import create_env

def gen_rand_dir(param):
    ndims = len(param.shape)
    if ndims == 1 or ndims == 0:
        # don't do any random direction for scalars
        return np.zeros_like(param)
    elif ndims == 2:
        dir = np.random.normal(size=param.shape)
        dir /= np.sqrt(np.sum(np.square(dir),axis=0,keepdims=True))
        dir *= np.sqrt(np.sum(np.square(param),axis=0,keepdims=True))
        return dir
    elif ndims == 4:
        dir = np.random.normal(size=param.shape)
        dir /= np.sqrt(np.sum(np.square(dir),axis=(0,1,2),keepdims=True))
        dir *= np.sqrt(np.sum(np.square(param),axis=(0,1,2),keepdims=True))
        return dir
    else:
        assert False, "only 1, 2, 4 dimentional filters allowed, got {}".format(param.shape)

def generate_random_directions(param_path, out_folder):#, algo_name, env_name):
    baseparams = ParamLoader(param_path)
    # if isinstance(baseparams.params, list):
    #     load_env, test_env = create_env(env_name, algo_name, n_envs=1)
    #     num_envs = test_env.num_envs
    #     env = test_env
    #     model = ALGOS[algo_name].load(param_path,env=load_env,n_cpu_tf_sess=1)
    #     params = dict()
    #     for i, param_name in enumerate(model._param_load_ops.keys()):
    #         params[param_name] = baseparams.params[i]
    # else:
    #     params = base_params.params
    # print({name:arg.shape for name, arg in params.items()})
    # for name, val in params.items():
    #     kern_name = None
    #     if "bias:" in name:
    #         idx = name.index("bias:")
    #         cropped = name[:idx]
    #         kern_name = cropped + "kernel:0"
    #     elif "b:" in name:
    #         idx = name.index("b:")
    #         cropped = name[:idx]
    #         kern_name = cropped + "w:0"
    #     if kern_name:
    #         print("bias:",np.mean(np.abs(val)),"\t\t","kern:",np.mean(np.abs(params[kern_name])))
    # return
    params = baseparams.get_params()
    x_dir = [gen_rand_dir(param) for param in params]
    y_dir = [gen_rand_dir(param) for param in params]
    baseparams.set_params(x_dir)
    baseparams.save(os.path.join(out_folder, f"x_dir{baseparams.extension()}"))
    baseparams.set_params(y_dir)
    baseparams.save(os.path.join(out_folder, f"y_dir{baseparams.extension()}"))

if __name__ == "__main__":
    # loader = ParamLoader()
    # print([p.shape for p in loader.get_params()])
    # print(loader.data)
    param_path = "trained_agents/ppo2/LunarLander-v2.pkl"
    generate_random_directions(param_path, "arg", "ppo2", "LunarLander-v2")
    param_path = "trained_agents/ddpg/BipedalWalker-v2.pkl"
    generate_random_directions(param_path, "arg", "ddpg", "BipedalWalker-v2")
    param_path = "trained_agents/ppo2/QbertNoFrameskip-v4.pkl"
    generate_random_directions(param_path, "arg", "ppo2", "QbertNoFrameskip-v4")
# arr = np.concatenate([np.ones([4,4]),np.ones([4,4])*10],axis=0)
# print(arr)
# print(gen_rand_dir(arr))
