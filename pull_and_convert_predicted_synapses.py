import os
import numpy as np
from db import db
import xml.etree.ElementTree as ET
from copy import deepcopy
from tqdm import tqdm


CUBE_LEN = 128
version = 'v0'
synapse_list = db.get_predicted_synapses()
template = ET.parse('synapses/template.nml')
root = template.getroot()

# Change experiment name
exp_name = '{}_{}'.format(root[0][0].get('name'), version)
root[0][0].set('name', exp_name)

# Add nodes/edges to root[1]
counts = [0, 0]
for idx, synapse in tqdm(enumerate(synapse_list), total=len(synapse_list)):
    st = synapse['type']
    if st == 'ribbon':
        st_idx = 0
    else:
        st_idx = 1
    x = synapse['x']
    y = synapse['y']
    z = synapse['z']
    if counts[st_idx] > 0:
        node = deepcopy(root[st_idx + 1][0][0])
    else:
        node = root[st_idx + 1][0][0]
    node.set('id', str(counts[st_idx] + 1))
    node.set('x', str(x * CUBE_LEN))
    node.set('y', str(y * CUBE_LEN))
    node.set('z', str(z * CUBE_LEN))
    if counts[st_idx] > 0:
        root[st_idx + 1][0].append(node)
    else:
        root[st_idx + 1][0][0] = node
    counts[st_idx] += 1
template.write(os.path.join('synapses', '{}.nml'.format(exp_name)))

