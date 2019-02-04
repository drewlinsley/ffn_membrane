import sys
import os
import subprocess
import numpy as np
import metrics
import h5py


if __name__ == '__main__':

    script_root = '/home/jk/PycharmProjects/ffn'
    train_dataset_name = ['neurproof']
    test_dataset_name_list = ['berson',
                         'isbi2013',
                         'cremi_a',
                         'cremi_b',
                         'cremi_c']
    fov_type = 'traditional_fov'
    test_dataset_type = 'val' #'train'


    net_cond_name = 'feedback_hgru_generic_longfb_3l_neuroproof_r0_46959'#'feedback_hgru_generic_longfb_3l_long'#'feedback_hgru_generic_longfb_3l' #'feedback_hgru_3l_dualch' #'feedback_hgru_2l'  # 'convstack_3d'


    hdf_root = os.path.join('/media/data_cifs/connectomics/datasets/third_party/', fov_type)
    inference_root = os.path.join('/media/data_cifs/connectomics/ffn_inferred', fov_type)


    num_model_repeats = 1 # 5

    for test_dataset_name in test_dataset_name_list:
        for irep in range(num_model_repeats):
            gt_fullpath = os.path.join(hdf_root, test_dataset_name, test_dataset_type, 'groundtruth.h5')
            data = h5py.File(gt_fullpath, 'r')
            gt = np.array(data['stack'])
            gt_unique = np.unique(gt)

            inference_fullpath = os.path.join(inference_root, net_cond_name, test_dataset_name, test_dataset_type, 'inferred.npz')
            seg = np.load(inference_fullpath)['segmentation']
            seg_unique = np.unique(seg)

            # import matplotlib.pyplot as plt
            # import skimage.segmentation
            # xy_relabeled,_,_= skimage.segmentation.relabel_sequential(gt[50,:,:], offset=1)
            # zx_relabeled,_,_= skimage.segmentation.relabel_sequential(gt[:,:,50], offset=1)
            # plt.subplot(121);plt.imshow(xy_relabeled);plt.title('YX slice')
            # plt.subplot(122);plt.imshow(zx_relabeled);plt.title('ZY slice')
            # plt.show()

            arand, precision, recall = metrics.adapted_rand(seg, gt, all_stats=True)
            print('>>>>>>>>>>>>>>Tested on: ' + test_dataset_name, ' shape: ' +str(gt.shape))
            print('gt_unique_lbls: ' + str(gt_unique.shape) + ' seg_unique_lbls: ' + str(seg_unique.shape))
            print('arand: '+str(arand)+', precision: '+str(precision)+' recall: '+str(recall))

# .npz
# #
