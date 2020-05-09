import os
import numpy as np
from glob import glob


files_0 = glob('/localscratch/merge/*.npy')
files_1 = glob('/media/data/merge/*.npy')
files = files_0 + files_1
files = np.array(files)
plane = [int(f.split('x')[1].split('.')[0]) for f in files]
files = files[np.argsort(plane)]

out = []
for f in files:
    out.append('/opt/anaconda2/envs/powerAIlab/bin/ipython get_region_props.py {} /localscratch/rps/{}\n'.format(f, f.split(os.path.sep)[-1]))

f = open('run_region_props.sh','w')
f.writelines(out)
f.close()

