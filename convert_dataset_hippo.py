import matplotlib.pyplot as plt
from skimage import io
import os
import numpy as np 
import h5py
from scipy import ndimage
import skimage.feature
import skimage.measure
import skimage.morphology 
import sys

name = 'hippocampus'
out_root = '/media/data_cifs/connectomics/datasets/third_party/wide_fov'
### 'volume', 'gt' <--labels, 'label' <--membrane
### [z,y,x] ~ [64, 1280, 1280]

data = np.load('/media/data_cifs/connectomics/datasets/kharris_0.npz')
instances = np.array(data['gt'])
volume = data['volume']
print(name + ' inst: ' + str(instances.shape))
print(name + ' vol: ' + str(volume.shape))

# RELABEL FROM ONE
print('before re-labeling: ' + str(len(np.unique(instances))))
instances_new, _, _ = skimage.segmentation.relabel_sequential(instances, offset=1)
print('after re-labeling: ' + str(len(np.unique(instances_new))))

write_dir = os.path.join(out_root,name,'train')
if not os.path.isdir(write_dir):
	os.makedirs(write_dir)
writer = h5py.File(os.path.join(write_dir,'groundtruth.h5'), 'w')
writer.create_dataset('stack', data=instances_new, dtype='<i8')
writer.close()
writer = h5py.File(os.path.join(out_root,name,'train','grayscale_maps.h5'), 'w')
writer.create_dataset('raw', data=volume, dtype='|u1')
writer.close()