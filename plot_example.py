import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure

def get_edge(im):
    edge_horizont = ndimage.sobel(im, 0)
    edge_vertical = ndimage.sobel(im, 1)
    edge = 1-np.array(np.hypot(edge_horizont, edge_vertical)>0, dtype=np.float)
    return measure.label(edge, background=0)

if __name__ == '__main__':
    hdf_root = '/media/data_cifs/connectomics/datasets/third_party/'
    output_root = '/media/data_cifs/connectomics/ffn_inferred'
    fov = 'wide_fov'
    test_dataset_name = 'fullberson'
    dataset_type = 'train'
    
    cond_name = 'convstack_3d_allbutberson_r0_topup_206465' #'feedback_hgru_v5_3l_linfb_allbutberson_r0_363'

    # dataset_shape_list = [[250, 250, 250],
    #                       [384, 384, 300],
    #                       [1024, 1024, 75], 
    #                       [1250, 1250, 100], 
    #                       [1250, 1250, 100],
    #                       [1250, 1250, 100]]


    # LOAD VOLUME
    img_fullpath = os.path.join(hdf_root, fov, test_dataset_name, dataset_type, 'grayscale_maps.h5')
    data = h5py.File(img_fullpath, 'r')
    volume = np.array(data['raw'])

    # LOAD GT
    gt_fullpath = os.path.join(hdf_root, fov, test_dataset_name, dataset_type, 'groundtruth.h5')
    data = h5py.File(gt_fullpath, 'r')
    instances = np.array(data['stack'])

    # LOAD INFERRED MAP
    output_fullpath = os.path.join(output_root, fov, cond_name, test_dataset_name, dataset_type, '0/0/seg-0_0_0.npz')
    inference = np.load(output_fullpath)
    # ['overlaps', 'segmentation', 'request', 'origins', 'counters']
    inference = inference['segmentation']

    import skimage.feature
    import skimage.measure
    import skimage.morphology 

    print('mean ='+ str(np.mean(volume.flatten())))
    print('std ='+str(np.std(volume.flatten())))

    # i=100
    # while i<384:
    #     i+=1
    #     instances_slice, _, _ = skimage.segmentation.relabel_sequential(instances[i,:,:], offset=1)
    #     inference_slice, _, _ = skimage.segmentation.relabel_sequential(inference[i,:,:], offset=1)
    #     plt.subplot(1,5,1);plt.imshow(volume[i,:,:], cmap='gray');plt.axis('off')
    #     plt.subplot(1,5,2);plt.imshow(instances_slice, cmap='viridis');plt.axis('off')
    #     plt.subplot(1,5,3);plt.imshow(get_edge(instances_slice), cmap='viridis');plt.axis('off')
    #     plt.subplot(1,5,4);plt.imshow(inference_slice, cmap='viridis');plt.axis('off')
    #     plt.subplot(1,5,5);plt.imshow(get_edge(inference_slice), cmap='viridis');plt.axis('off')
    #     print(i)
    #     plt.show()

    
    instances_slice, _, _ = skimage.segmentation.relabel_sequential(instances[102,:,:], offset=1)
    inference_slice, _, _ = skimage.segmentation.relabel_sequential(inference[102,:,:], offset=1)
    plt.subplot(3,5,1);plt.imshow(volume[102,:,:], cmap='gray');plt.axis('off')
    plt.subplot(3,5,2);plt.imshow(instances_slice, cmap='viridis');plt.axis('off')
    plt.subplot(3,5,3);plt.imshow(get_edge(instances_slice), cmap='viridis');plt.axis('off')
    plt.subplot(3,5,4);plt.imshow(inference_slice, cmap='viridis');plt.axis('off')
    plt.subplot(3,5,5);plt.imshow(get_edge(inference_slice), cmap='viridis');plt.axis('off')

    instances_slice, _, _ = skimage.segmentation.relabel_sequential(instances[280,:,:], offset=1)
    inference_slice, _, _ = skimage.segmentation.relabel_sequential(inference[280,:,:], offset=1)
    plt.subplot(3,5,6);plt.imshow(volume[280,:,:], cmap='gray');plt.axis('off')
    plt.subplot(3,5,7);plt.imshow(instances_slice, cmap='viridis');plt.axis('off')
    plt.subplot(3,5,8);plt.imshow(get_edge(instances_slice), cmap='viridis');plt.axis('off')
    plt.subplot(3,5,9);plt.imshow(inference_slice, cmap='viridis');plt.axis('off')
    plt.subplot(3,5,10);plt.imshow(get_edge(inference_slice), cmap='viridis');plt.axis('off')

    instances_slice, _, _ = skimage.segmentation.relabel_sequential(instances[370,:,:], offset=1)
    inference_slice, _, _ = skimage.segmentation.relabel_sequential(inference[370,:,:], offset=1)
    plt.subplot(3,5,11);plt.imshow(volume[370,:,:], cmap='gray');plt.axis('off')
    plt.subplot(3,5,12);plt.imshow(instances_slice, cmap='viridis');plt.axis('off')
    plt.subplot(3,5,13);plt.imshow(get_edge(instances_slice), cmap='viridis');plt.axis('off')
    plt.subplot(3,5,14);plt.imshow(inference_slice, cmap='viridis');plt.axis('off')
    plt.subplot(3,5,15);plt.imshow(get_edge(inference_slice), cmap='viridis');plt.axis('off')
    plt.show()
