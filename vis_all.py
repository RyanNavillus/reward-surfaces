import subprocess
import sys
import os

for fname in sys.argv[1:]:
    parts = fname.split("/")[-5:]
    bname = "_".join(parts)[:-4]
    basename = "vis/"+bname
    arglist = ["python","visualize.py",basename,fname]
    print(" ".join(arglist))
    subprocess.Popen(arglist)
