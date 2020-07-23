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

# # BERSON 384 384 384
# name = 'berson'
# fullpath = '/media/data_cifs/connectomics/datasets/berson.npz'
# data = np.load(fullpath)
# membrane = np.expand_dims(1 - data['label'], axis=3)
# volume = np.expand_dims(data['volume'], axis=3)
# volume_n_membrane = np.concatenate([volume, membrane], axis=3)
#
# print(' vol: ' + str(volume.shape))
#
# write_dir = os.path.join(out_root,name,'train')
# if not os.path.isdir(write_dir):
#     os.makedirs(write_dir)
# writer = h5py.File(os.path.join(out_root,name,'train','grayscale_maps.h5'), 'w')
# writer.create_dataset('raw', data=volume_n_membrane, dtype='|u1')
# writer.close()

# BERSON 768 384 384
name = 'berson2x_w_memb'
volpath = '/media/data_cifs/connectomics/datasets/berson2x.npy'
labelpath = '/media/data_cifs/connectomics/datasets/berson2x_labels.npy'
volume = np.load(volpath)
volume = np.transpose(volume, (2, 0, 1)) ## FROM (sp, sp, depth) to (depth, sp, sp)
print(volume.shape)
labels = np.load(labelpath)
labels = np.transpose(labels, (0, 2, 1))
print(labels.shape)
labels, _, _ = skimage.segmentation.relabel_sequential(labels, offset=1)

# get membrane
import cv2
from skimage.filters import scharr
membranes = []
for i in range(np.shape(labels)[0]):
    membrane = scharr(labels[i,:,:])
    membrane[membrane > 0] = 1
    membrane = membrane.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    membrane = cv2.dilate(
            membrane,
            kernel,
            iterations=1)
    membrane = np.invert(membrane)  # * 255
    membranes.append(membrane)
membranes = np.array(membranes)
membranes -= 254
plt.subplot(121);plt.imshow(membranes[200,:,:]);plt.subplot(122);plt.imshow(volume[200,:,:],cmap='gray');plt.show()

volume = np.expand_dims(volume, axis=3)
membranes = np.expand_dims(membranes, axis=3)
volume_n_membrane = np.concatenate([volume, membranes], axis=3)

print(' vol: ' + str(volume.shape))

write_dir = os.path.join(out_root,name,'train')
if not os.path.isdir(write_dir):
    os.makedirs(write_dir)
writer = h5py.File(os.path.join(out_root,name,'train','grayscale_maps.h5'), 'w')
writer.create_dataset('raw', data=volume_n_membrane, dtype='|u1')
writer.close()


writer = h5py.File(os.path.join(out_root,name,'train','groundtruth.h5'), 'w')
writer.create_dataset('stack', data=labels, dtype='<i8')
writer.close()