import subprocess
import sys
import os

for fname in sys.argv[1:]:
    _, algo, env_name, bname = fname.split("/")
    basename = "vis/"+env_name+algo+bname[:-4]
    arglist = ["python","visualize.py",basename,fname]
    print(" ".join(arglist))
    subprocess.Popen(arglist)
