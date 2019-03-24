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

out_root = '/media/data_cifs/connectomics/datasets/third_party/wide_fov'

# # BERSON 384
name = 'berson_w_inf_memb'
volume = np.load('/media/data_cifs/connectomics/datasets/berson_0.npz')['volume']
membrane = np.load('/media/data_cifs/connectomics/datasets/berson_v1_predictions.npy')[:,:,:,0]
membrane*=255

membrane = np.expand_dims(membrane, axis=3)
volume = np.expand_dims(volume, axis=3)
volume_n_membrane = np.concatenate([volume, membrane], axis=3)
print(' vol: ' + str(volume.shape))

write_dir = os.path.join(out_root,name,'train')
if not os.path.isdir(write_dir):
    os.makedirs(write_dir)
writer = h5py.File(os.path.join(out_root,name,'train','grayscale_maps.h5'), 'w')
writer.create_dataset('raw', data=volume_n_membrane, dtype='|u1')
writer.close()

# # BERSON 384 GT (because it needs to be redone)
path = '/media/data_cifs/andreas/connectomics/Berson'
file = 'updated_Berson.h5'
gt_labels = np.array(h5py.File(os.path.join(path, file), 'r')['masks'])
gt_labels, _, _ = skimage.segmentation.relabel_sequential(gt_labels, offset=1)
writer = h5py.File(os.path.join(out_root,name,'train','groundtruth.h5'), 'w')
writer.create_dataset('stack', data=gt_labels, dtype='<i8')
writer.close()

# BERSON 768 384 384
name = 'berson2x_w_inf_memb'
volume = np.load('/media/data_cifs/connectomics/datasets/berson_v2_0.npz')['volume']
membrane = np.load('/media/data_cifs/connectomics/datasets/berson_v2_predictions.npy')[:,:,:,0]
membrane*=255
# # transpose (DONT DO WITH NEW SEGMENT LABELS)
# membrane = np.transpose(membrane, (0,2,1))
# volume = np.transpose(volume, (0,2,1))
membrane = np.expand_dims(membrane, axis=3)
volume = np.expand_dims(volume, axis=3)
volume_n_membrane = np.concatenate([volume, membrane], axis=3)
print(' vol: ' + str(volume.shape))
write_dir = os.path.join(out_root,name,'train')
if not os.path.isdir(write_dir):
    os.makedirs(write_dir)
writer = h5py.File(os.path.join(out_root,name,'train','grayscale_maps.h5'), 'w')
writer.create_dataset('raw', data=volume_n_membrane, dtype='|u1')
writer.close()

# # BERSON 768 GT (because it needs to be redone)
labelpath = '/media/data_cifs/connectomics/datasets/berson2x_labels.npy'
gt_labels = np.load(labelpath)
gt_labels, _, _ = skimage.segmentation.relabel_sequential(gt_labels, offset=1)
writer = h5py.File(os.path.join(out_root,name,'train','groundtruth.h5'), 'w')
writer.create_dataset('stack', data=gt_labels, dtype='<i8')
writer.close()
