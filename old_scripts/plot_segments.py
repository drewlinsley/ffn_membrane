import matplotlib.pyplot as plt
from skimage import io
import os
import numpy as np 
import h5py
from scipy import ndimage
import skimage.feature
import skimage.measure
import skimage.morphology 

membrane = np.load('/media/data_cifs/connectomics/evaluations/predictions_berson3d_0_seung_unet3d_berson_0_berson_0_2018_08_24_14_28_37_755680.npy')
volume = np.load('/media/data_cifs/connectomics/evaluations/volume_berson3d_0_seung_unet3d_berson_0_berson_0_2018_08_24_14_28_37_755680.npy')
membrane = membrane[:,:,:,0] > 0.5

#### WATERSHED THIS DATASET (2D)
distance = []
local_maxi = []
labels = []
segments = []
for idx in range(membrane.shape[0]):
	distance.append(ndimage.distance_transform_edt(membrane[idx,:,:]))
	local_maxi.append(skimage.feature.peak_local_max(distance[idx], min_distance=0, indices=False, footprint=None, labels=None))
	labels.append(skimage.measure.label(local_maxi[idx], background=0))
	segments.append(skimage.morphology.watershed(~membrane[idx,:,:], labels[idx], mask=None))
distance = np.stack(distance, axis=0)
local_maxi = np.stack(local_maxi, axis=0)
labels = np.stack(labels, axis=0)
segments = np.stack(segments, axis=0)

show_idx=50


plt.subplot(141);plt.imshow(volume[show_idx,:,:], cmap='bone')
plt.subplot(142);plt.imshow(membrane[show_idx,:,:])
plt.subplot(144);plt.imshow(segments[show_idx,:,:])
plt.show()

