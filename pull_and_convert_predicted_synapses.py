import os
import numpy as np
from db import db
import xml.etree.ElementTree as ET
from copy import deepcopy
from tqdm import tqdm


CUBE_LEN = 1
MIN_CUBE_LEN = 0


def convert_synapse_predictions(
        synapse_list=None,
        version="v0",
        offset=None,
        template_ribbon="synapses/template.nml",
        template_amacrine="synapses/template.nml",
        exp_name="ding",
        out_ribbon_file="synapses/pred_synapses_{}.nml",
        out_amacrine_file="synapses/amacrine_{}.nml"):
    """Convert coordinates into a Knossos node file."""
    if synapse_list is None:
        synapse_list = db.get_predicted_synapses()
    template_ribbon = ET.parse(template_ribbon)
    template_amacrine = ET.parse(template_amacrine)
    root_ribbon = template_ribbon.getroot()
    root_amacrine = template_amacrine.getroot()

    # Change experiment name
    # exp_name = '{}_{}'.format(root_ribbon[0][0].get('name'), version)
    root_ribbon[0][0].set('name', exp_name)
    root_amacrine[0][0].set('name', exp_name)

    # Add nodes/edges to root[1]
    all_xs, all_ys, all_zs = [], [], []
    counts = [0, 0]
    for idx, synapse in tqdm(enumerate(synapse_list), total=len(synapse_list)):
        st = synapse['type']
        if st == 'ribbon':
            st_idx = 0
            root = root_ribbon
        else:
            st_idx = 1
            root = root_amacrine
        x = synapse['x']
        y = synapse['y']
        z = synapse['z']
        if offset is not None:
            x -= offset['x']
            y -= offset['y']
            z -= offset['z']
        if counts[st_idx] > 0:
            node = deepcopy(root[st_idx + 1][0][0])
        else:
            node = root[st_idx + 1][0][0]
        node.set('id', str(idx + 1))
        node.set('x', str((x * CUBE_LEN) - MIN_CUBE_LEN))
        node.set('y', str((y * CUBE_LEN) - MIN_CUBE_LEN))
        node.set('z', str((z * CUBE_LEN) - MIN_CUBE_LEN))
        all_xs += [x * CUBE_LEN]
        all_ys += [y * CUBE_LEN]
        all_zs += [z * CUBE_LEN]
        if counts[st_idx] > 0:
            root[st_idx + 1][0].append(node)
        else:
            root[st_idx + 1][0][0] = node
        counts[st_idx] += 1
    print(('xmin: {} xmax: {}'.format(np.min(all_xs), np.max(all_xs))))
    print(('ymin: {} ymax: {}'.format(np.min(all_ys), np.max(all_ys))))
    print(('zmin: {} zmax: {}'.format(np.min(all_zs), np.max(all_zs))))
    """
    template_ribbon_unnorm = ET.parse(template_ribbon)
    template_amacrine_unnorm = ET.parse(template_amacrine)
    template_ribbon_unnorm.write(os.path.join('synapses', out_ribbon_file.format(exp_name)))
    template_amacrine_unnorm.write(os.path.join('synapses', out_amacrine_file.format(exp_name)))
    """
    template_ribbon._setroot(root)
    template_ribbon.write(os.path.join(out_ribbon_file.format(exp_name)))
    print("Finished")


if __name__ == '__main__':
    convert_synapse_predictions()

