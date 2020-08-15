from extract_params import ParamLoader
import sys
import json

fname = sys.argv[1]
loader = ParamLoader(fname)
hyperparams = dict()
for name, value in loader.data.items():
    try:
        json.dumps(value)
        hyperparams[name] = value
    except TypeError:
        pass
print(json.dumps(hyperparams,indent=4))
