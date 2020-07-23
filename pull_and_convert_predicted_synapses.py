import os
import colorsys
import wknml
import numpy as np
from db import db
import xml.etree.ElementTree as ET
from copy import deepcopy
from tqdm import tqdm
from wknml import NMLParameters, Group, Edge, Node, Tree, NML, Branchpoint, Comment
from typing import Tuple, List, Dict, Union, Any
import gzip


CUBE_LEN = 1
MIN_CUBE_LEN = 0


def random_color_rgba():
    # https://stackoverflow.com/a/43437435/783758

    h, s, l = np.random.random(), 0.5 + np.random.random() / 2.0, 0.4 + np.random.random() / 5.0
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b, 1)


def convert_synapse_predictions(
        synapse_list=None,
        version="v0",
        offset=None,
        template_ribbon="/media/data_cifs_lrs/projects/prj_connectomics/ffn_membrane_v2/synapses/ribbon_ding_fixed.nml",
        out_ribbon_file="{}_proc_synapses.nml",
        exp_name="ding"):
    """Convert coordinates into a Knossos node file."""
    if synapse_list is None:
        synapse_list = db.get_predicted_synapses()

    # Change experiment name
    groups = [wknml.Group(id=1, name='Ribbon synapses', children=[])]
    trees = []
    # synapse_list = synapse_list[:int(len(synapse_list) * 0.1)]
    for idx, synapse in tqdm(enumerate(synapse_list), total=len(synapse_list)):
        st = synapse['type']
        x = synapse['x']
        y = synapse['y']
        z = synapse['z']
        if offset is not None:
            x -= offset['x']
            y -= offset['y']
            z -= offset['z']
        tree = wknml.Tree(
            id=idx + 1,
            color=random_color_rgba(),
            name="Ribbon {}".format(idx + 1), 
            nodes=[wknml.Node(id=idx + 1, position=(x, y, z), radius=1)],
            edges=[],
            groupId=1)
        trees.append(tree)
    nml = wknml.NML(
        parameters=wknml.NMLParameters(
            name=exp_name,
            scale=[13.199999809265137, 13.199999809265137, 26],
        ),
        trees=trees,
        branchpoints=[],
        comments=[],
        groups=groups,
    )
    with open(os.path.join(out_ribbon_file.format(exp_name)), "wb") as f:
        wknml.write_nml(f, nml)

    # Compress to .gz
    with open(os.path.join(out_ribbon_file.format(exp_name)), "rb") as src, gzip.open("{}{}".format(os.path.join(out_ribbon_file.format(exp_name)), ".gz"), "wb") as dst:
        dst.writelines(src)
    print("Finished")


if __name__ == '__main__':
    convert_synapse_predictions()

