import cloudpickle
import zipfile
from stable_baselines.common.save_util import data_to_json, json_to_data, params_to_bytes, bytes_to_params
import json


class ParamLoader:
    def __init__(self, load_path):
        self.load_path = load_path
        if load_path.endswith(".pkl"):
            with open(load_path, "rb") as file_:
                self.data, self.params = cloudpickle.load(file_)

        elif load_path.endswith(".zip"):
            with zipfile.ZipFile(load_path, "r") as file_:
                namelist = file_.namelist()
                # If data or parameters is not in the
                # zip archive, assume they were stored
                # as None (_save_to_file allows this).
                data = None
                params = None
                custom_objects = None
                load_data = True
                if "data" in namelist and load_data:
                    # Load class parameters and convert to string
                    # (Required for json library in Python 3.5)
                    json_data = file_.read("data").decode()
                    data = json_to_data(json_data, custom_objects=custom_objects)
                    self.data = data
                if "parameters" in namelist:
                    # Load parameter list and and parameters
                    parameter_list_json = file_.read("parameter_list").decode()
                    parameter_list = json.loads(parameter_list_json)
                    serialized_params = file_.read("parameters")
                    params = bytes_to_params(
                        serialized_params, parameter_list
                    )
                    self.params = params
        else:
            raise RuntimeError("bad file path for stable baselines load")

    def extension(self):
        return self.load_path[-4:]

    def is_pickle(self):
        return self.load_path.endswith(".pkl")

    def get_params(self):
        if self.is_pickle():
            return self.params if isinstance(self.params, list) else list(self.params.values())
        else:
            return [val for name,val in self.params.items()]

    def set_params(self, params):
        if self.is_pickle():
            if isinstance(self.params, list):
                self.params = params
            else:
                for name, param in zip(self.params, params):
                    self.params[name] = param
        else:
            for name, param in zip(self.params, params):
                self.params[name] = param

    def save(self, save_path):
        if self.is_pickle():
            assert save_path.endswith(".pkl")

            with open(save_path, "wb") as file_:
                cloudpickle.dump((self.data, self.params), file_)
        else:
            assert save_path.endswith(".zip")

            data = self.data
            params = self.params

            if data is not None:
                serialized_data = data_to_json(data)
            if params is not None:
                serialized_params = params_to_bytes(params)
                # We also have to store list of the parameters
                # to store the ordering for OrderedDict.
                # We can trust these to be strings as they
                # are taken from the Tensorflow graph.
                serialized_param_list = json.dumps(
                    list(params.keys()),
                    indent=4
                )

            # Create a zip-archive and write our objects
            # there. This works when save_path
            # is either str or a file-like
            with zipfile.ZipFile(save_path, "w") as file_:
                # Do not try to save "None" elements
                if data is not None:
                    file_.writestr("data", serialized_data)
                if params is not None:
                    file_.writestr("parameters", serialized_params)
                    file_.writestr("parameter_list", serialized_param_list)
