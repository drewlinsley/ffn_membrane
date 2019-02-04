import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    script_root = '/home/jk/PycharmProjects/ffn/'
    request_txt_root = os.path.join(script_root, 'configs')
    hdf_root = '/media/data_cifs/connectomics/datasets/third_party/traditional/'
    ckpt_root = '/media/data_cifs/connectomics/ffn_ckpts'
    output_root = '/media/data_cifs/connectomics/ffn_inferred'

    net_name = 'ffn'
    fov_size = [41, 41, 21] # [33,33,33]
    deltas = [10, 10, 5] #[8,8,8]
    dataset_name_list = ['neuroproof','berson','isbi2013','cremi_a','cremi_b','cremi_c']
    dataset_type = 'val' #'train'
    dataset_shape_list = [[520, 520, 520],#[250, 250, 250],
    					  [384, 192, 384],
                          [1024, 512, 100],
                          [1250, 625, 125],
                          [1250, 625, 125],
                          [1250, 625, 125]]
    # dataset_shape_list = [[250, 250, 250],
    #                       [384, 384, 300],
    #                       [1024, 1024, 75], 
    #                       [1250, 1250, 100], 
    #                       [1250, 1250, 100],
    #                       [1250, 1250, 100]]

    num_model_repeats = 1
    image_mean = 128
    image_stddev = 33

    for trian_dataset_name in dataset_name_list:
        for test_dataset_name in dataset_name_list:
            for irep in range(num_model_repeats):
				# cond_name = net_name + '_' + trian_dataset_name + '_r' + str(irep)
				cond_name = net_name + '_' + 'pretrained'

				# LOAD VOLUME
				img_fullpath = os.path.join(hdf_root, test_dataset_name, dataset_type, 'grayscale_maps.h5')
				data = h5py.File(img_fullpath, 'r')
				volume = np.array(data['raw'])

				# LOAD GT
				gt_fullpath = os.path.join(hdf_root, test_dataset_name, dataset_type, 'groundtruth.h5')
				data = h5py.File(gt_fullpath, 'r')
				instances = np.array(data['stack'])

				# LOAD INFERRED MAP
				output_fullpath = os.path.join(output_root, cond_name, test_dataset_name, '0/0/seg-0_0_0.npz')
				inference = np.load(output_fullpath)
				# ['overlaps', 'segmentation', 'request', 'origins', 'counters']
				inference = inference['segmentation']
				
				import ipdb
				ipdb.set_trace()

show_idx=342
import skimage.feature
import skimage.measure
import skimage.morphology 
instances_slice, _, _ = skimage.segmentation.relabel_sequential(instances[show_idx,:,:], offset=1)
inference_slice, _, _ = skimage.segmentation.relabel_sequential(inference[show_idx,:,:], offset=1)
plt.subplot(131);plt.imshow(volume[show_idx,:,:], cmap='gray')
plt.subplot(132);plt.imshow(instances_slice)
plt.subplot(133);plt.imshow(inference_slice)
plt.show()
