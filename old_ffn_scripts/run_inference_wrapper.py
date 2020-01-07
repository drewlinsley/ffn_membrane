import sys
import os
import subprocess
import numpy as np
import metrics
import h5py

def write_custom_request(request_txt_fullpath, hdf_fullpath, ckpt_fullpath, output_fullpath,
                         net_name,
                         fov_size, deltas,
                         image_mean=128, image_stddev=33):
    # 'validate_request_' + str(validate_jobid) + '.txt'
    directory = os.path.split(request_txt_fullpath)[0]
    if not os.path.isdir(directory):
        os.makedirs(directory)
    file = open(request_txt_fullpath,"w+")
    file.write('image { \n')
    file.write('  hdf5: "' + hdf_fullpath + ':raw" \n')
    file.write('}')
    file.write('image_mean: ' + str(image_mean) + '\n')
    file.write('image_stddev: ' + str(image_stddev) + '\n')
    file.write('seed_policy: "PolicyPeaks" \n')
    file.write('model_checkpoint_path: "' + ckpt_fullpath + '" \n')
    file.write('model_name: "'+net_name+'.ConvStack3DFFNModel" \n')
    file.write('model_args: "{\\"depth\\": 12, \\"fov_size\\": ' + str(fov_size) + ', \\"deltas\\": ' + str(deltas) + '}" \n')
    file.write('segmentation_output_dir: "' + output_fullpath + '" \n')
    file.write('inference_options { \n')
    file.write('  init_activation: 0.95 \n')
    file.write('  pad_value: 0.05 \n')
    file.write('  move_threshold: 0.9 \n')
    file.write('  min_boundary_dist { x: 1 y: 1 z: 1} \n')
    file.write('  segment_threshold: 0.6 \n')
    file.write('  min_segment_size: 1000 \n')
    file.write('} \n')
    file.close()


def find_all_ckpts(ckpt_root, fov_type, net_cond_name):
    raw_items = os.listdir(os.path.join(ckpt_root, fov_type, net_cond_name))
    items = []
    for item in raw_items:
        if (item.split('.')[0]=='model') & (item.split('.')[-1]=='meta'):
            items.append(int(item.split('.')[1].split('-')[1]))
    items.sort()
    return items


def find_checkpoint(checkpoint_num, ckpt_root, fov_type, net_cond_name, factor):
    raw_items = os.listdir(os.path.join(ckpt_root, fov_type, net_cond_name))
    items = []
    for item in raw_items:
        if (item.split('.')[0]=='model') & (item.split('.')[-1]=='meta'):
            items.append(int(item.split('.')[1].split('-')[1]))
    items.sort()
    interval = len(items)/factor

    checkpoint_num = items[checkpoint_num*interval]
    return checkpoint_num


if __name__ == '__main__':

    num_machines = int(sys.argv[1])
    i_machine = int(sys.argv[2])

    script_root = '/home/jk/PycharmProjects/ffn/'
    request_txt_root = os.path.join(script_root, 'configs')
    hdf_root = '/media/data_cifs/connectomics/datasets/third_party/'
    ckpt_root = '/media/data_cifs/connectomics/ffn_ckpts'
    output_root = '/media/data_cifs/connectomics/ffn_inferred'

    fov_type = 'traditional_fov'
    fov_size = [33, 33, 33]
    deltas = [8, 8, 8]
    # fov_type = 'flat_fov'
    # fov_size = [41, 41, 21]
    # deltas = [10, 10, 5]
    # fov_type = 'wide_fov'
    # fov_size = [57, 57, 13]
    # deltas = [8, 8, 3]

    net_name = 'convstack_3d'
    train_dataset_name = 'provided'
    train_dataset_shape = [520, 520, 520]
    train_dataset_type = 'train'

    min_ckpt = None
    max_ckpt = None
    ckpt_steps = 5000  ## <500 for

    finetune_dataset_name = ['berson','isbi2013', 'cremi_a','cremi_b','cremi_c']
    finetune_dataset_shape_list = [[384, 192, 384],
                                   [1024, 512, 100],
                                   [1250, 625, 125],
                                   [1250, 625, 125],
                                   [1250, 625, 125]]
    test_dataset_type = 'val'  # 'train'
    include_self_test = True

    image_mean = 128
    image_stddev = 33

    kth_job = 0
    for finetune_dataset_name, finetune_dataset_shape in zip(finetune_dataset_name, finetune_dataset_shape_list):

        kth_job += 1
        if np.mod(kth_job, num_machines) != i_machine and num_machines != i_machine:
            continue
        elif np.mod(kth_job, num_machines) != 0 and num_machines == i_machine:
            continue

        for test_dataset_name, test_dataset_type, test_dataset_shape in [(train_dataset_name, train_dataset_type, train_dataset_shape),
                                                                         (finetune_dataset_name, test_dataset_type, finetune_dataset_shape)]:
            if test_dataset_name == 'provided':
                test_dataset_name ='neuroproof'

            if (test_dataset_name == train_dataset_name)|((test_dataset_name=='neuroproof')&(train_dataset_name=='provided')):
                if not include_self_test:
                    continue

            ### DEFINE NAMES
            net_cond_name = net_name + '_' + train_dataset_name + '_' + finetune_dataset_name + '_r0' ############ TODO (jk): modified for finetuning-eval
            request_txt_fullpath = os.path.join(request_txt_root, fov_type, net_cond_name + '_teston_' + test_dataset_name + '_' + test_dataset_type + '.pbtxt')
            hdf_fullpath = os.path.join(hdf_root, fov_type, test_dataset_name, test_dataset_type, 'grayscale_maps.h5')
            gt_fullpath = os.path.join(hdf_root, fov_type, test_dataset_name, test_dataset_type, 'groundtruth.h5')

            ### OPEN TEXT
            eval_result_txt_fullpath = os.path.join(ckpt_root, fov_type, net_cond_name, 'eval_on_'+ test_dataset_name + '.txt')
            if os.path.isfile(eval_result_txt_fullpath):
                eval_result_txt = open(eval_result_txt_fullpath, "a")
            else:
                eval_result_txt = open(eval_result_txt_fullpath, "w+")
            eval_result_txt.write('TEST DATASET: ' + test_dataset_name + ', FOV: ' + fov_type)
            eval_result_txt.write("\n")
            eval_result_txt.close()

            current_best_arand = 99.

            ## Get list of ckpts to load
            print('>>>>> TRIMMING CKPS')
            ckpt_list = find_all_ckpts(ckpt_root, fov_type, net_cond_name)
            to_remove = []
            if min_ckpt != None:
                for ckpt in ckpt_list:
                    if ckpt < min_ckpt:
                        to_remove.append(ckpt)
            if max_ckpt != None:
                for ckpt in ckpt_list:
                    if ckpt > max_ckpt:
                        to_remove.append(ckpt)
            if ckpt_steps<500:
                interval = len(ckpt_list)/ckpt_steps
                for i, ckpt in enumerate(ckpt_list):
                    if i % interval != 0:
                        to_remove.append(ckpt)
            else:
                interval = ckpt_steps
                accumulator = -1
                for i, ckpt in enumerate(ckpt_list):
                    if ckpt / interval >= accumulator:
                        accumulator += 1
                    else:
                        to_remove.append(ckpt)
            for ckpt in to_remove:
                if ckpt in ckpt_list:
                    ckpt_list.remove(ckpt)
            print('>>>>> DONE.')
            print('>>>>> CKPTS :: '+ str(ckpt_list))
            for checkpoint_num in ckpt_list:

                ### DEFINE NAMES
                ckpt_fullpath = os.path.join(ckpt_root, fov_type, net_cond_name, 'model.ckpt-' + str(checkpoint_num))
                inference_fullpath = os.path.join(output_root, fov_type, net_cond_name + '_' + str(checkpoint_num), test_dataset_name, test_dataset_type)

                ### RUN INFERENCE
                print('>>>>>>>>>>>>>> Model: ' + net_cond_name)
                print('>>>>>>>>>>>>>> Tested on: ' + test_dataset_name, ' fov: ' + fov_type)
                write_custom_request(request_txt_fullpath, hdf_fullpath, ckpt_fullpath, inference_fullpath,
                                     net_name,
                                     fov_size, deltas,
                                     image_mean, image_stddev)
                command = 'python ' + os.path.join(script_root,'run_inference.py') + \
                          ' --inference_request="$(cat ' + request_txt_fullpath + ')"' +\
                          ' --bounding_box "start { x:0 y:0 z:0 } size { x:' + str(test_dataset_shape[0]) + ' y:' + str(test_dataset_shape[1]) + ' z:' + str(test_dataset_shape[2]) + ' }"'
                subprocess.call(command, shell=True)

                #### EVALUATE INFERRENCE
                data = h5py.File(gt_fullpath, 'r')
                gt = np.array(data['stack'])
                gt_unique = np.unique(gt)
                inference_fullpath = os.path.join(inference_fullpath, '0/0/seg-0_0_0.npz')
                seg = np.load(inference_fullpath)['segmentation']
                seg_unique = np.unique(seg)
                arand, precision, recall = metrics.adapted_rand(seg, gt, all_stats=True)

                print('gt_unique_lbls: ' + str(gt_unique.shape) + ' seg_unique_lbls: ' + str(seg_unique.shape))
                print('arand: ' + str(arand) + ', precision: ' + str(precision) + ' recall: ' + str(recall))
                eval_result_txt = open(eval_result_txt_fullpath, "a")
                eval_result_txt.write('>> CKPT: ' + str(checkpoint_num))
                eval_result_txt.write("\n")
                eval_result_txt.write('>>>> arand: ' + str(arand) + ', precision: ' + str(precision) + ', recall: ' + str(recall))
                eval_result_txt.write("\n")
                eval_result_txt.close()

                if arand < current_best_arand:
                    current_best_arand = arand

# .npz
# #
