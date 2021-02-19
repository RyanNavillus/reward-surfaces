from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
import collections
import torch
import numpy as np


class ParamLoader:
    def __init__(self, load_path):
        self.load_path = load_path
        assert load_path.endswith(".zip"), "bad file name for sb3 load"
        self.data, self.params, self.pytorch_vars = load_from_zip_file(load_path,device="cpu")

    def get_params(self):
        param_list = []
        assert isinstance(self.params['policy'],collections.OrderedDict)
        for name, param in self.params['policy'].items():
            nd_arr = param.numpy()
            param_list.append(nd_arr)
        return param_list

    def set_params(self, params):
        for new_param, old_param in zip(params, self.params['policy'].values()):
            old_param.data = torch.tensor(new_param)

    def save(self, save_path):
        save_to_zip_file(save_path, self.data, self.params, self.pytorch_vars)
