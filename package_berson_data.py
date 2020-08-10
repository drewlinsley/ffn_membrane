import os
import numpy as np
from config import Config
from db import db
import pandas as pd
import nibabel as nib
from utils.hybrid_utils import recursive_make_dir
from utils.hybrid_utils import pad_zeros
from utils.hybrid_utils import make_dir
from glob2 import glob
from synapse_test import test
from skimage.segmentation import relabel_sequential as rs


save_segmentations = True
max_h, max_w, max_d = 256, 256, 256
force_write = True
dry_run = False
config = Config()
paths = {
    # os.path.join(config.berson_path, '384_4096_1792_curated_with_labels', 'seg_384_4096_1792.031.npy'): [384 / config.shape[0], 4096 / config.shape[1], 1792 / config.shape[2]], 
    # os.path.join(config.berson_path, 'for_dustbusting', 'x0017_y0035_z0026_seg_v1.004.nii.gz'): [17, 35, 26],
}
paths = pd.read_csv("db/synapse_paths_for_berson.csv")
path_extent = np.asarray([3, 9, 9])
out_dir = "berson_synapse_segs"
make_dir(out_dir)
outpath_seg = os.path.join(out_dir, "{}_seg.nii") 
outpath_vol = os.path.join(out_dir, "{}_vol.nii")
outpath_mem = os.path.join(out_dir, "{}_mem.nii")
outpath_syn_ribbon = os.path.join(out_dir, "{}_syn_ribbon.nii")
outpath_recon_syn_ribbon = os.path.join(out_dir, "{}_syn_recon_ribbon.nii")
outpath_syn_amacrine = os.path.join(out_dir, "{}_syn_amacrine.nii")
search_radius = path_extent // 2
for k, seed in paths.iterrows():
    x, y, z = seed.x // 128, seed.y // 128, seed.z // 128
    seed = np.asarray([x, y, z])
    """
    qx, qy, qz = x // config.shape[0], y // config.shape[0], z // config.shape[0]
    candidates_p0, candidates_p1 = [], []
    for sz in range(qz - search_radius[2], qz + search_radius[2]):
        for sy in range(qy - search_radius[1], qy + search_radius[1]):
            for sx in range(qx - search_radius[0], qx + search_radius[0]):
                p0 = "/media/data_cifs/connectomics/ding_segmentations/{}/{}/{}/v0/0/0/seg-0_0_0.npz".format(
                    pad_zeros(sx, 4),
                    pad_zeros(sy, 4),
                    pad_zeros(sz, 4))
                print("Checking {}".format((sx, sy, sz)))
                if os.path.exists(p0):
                    candidates_p0.append((sx, sy, sz))
                else:
                    p1 = "/media/data_cifs/connectomics/ding_segmentations_merge/{}/{}/{}/v0/0/0/seg-0_0_0.npz".format(
                        pad_zeros(sx, 4),
                        pad_zeros(sy, 4),
                        pad_zeros(sz, 4))
                    if os.path.exists(p1):
                        candidates_p1.append((sx, sy, sz))
    if len(candidates_p0):
        seg_path =  "/media/data_cifs/connectomics/ding_segmentations/{}/{}/{}/v0/0/0/seg-0_0_0.npz".format(
            pad_zeros(candidates_p0[0], 4),
            pad_zeros(candidates_p0[1], 4),
            pad_zeros(candidates_p0[2], 4))
    elif len(candidates_p1):
        seg_path =  "/media/data_cifs/connectomics/ding_segmentations_merge/{}/{}/{}/v0/0/0/seg-0_0_0.npz".format(
            pad_zeros(candidates_p1[0], 4),
            pad_zeros(candidates_p1[1], 4),
            pad_zeros(candidates_p1[2], 4))
    else:
        seg_path = None
    """
    if save_segmentations:
        # segments = np.load(seg_path)["segmentation"]
        # vol_shape = np.array(segments.shape)
        # path_extent = vol_shape / np.array(config.shape)
        vol = np.zeros((np.array(config.shape) * path_extent), dtype=np.float32)
        segments = np.zeros_like(vol)
        for z in range(path_extent[0]):
            for y in range(path_extent[1]):
                for x in range(path_extent[2]):
                    path = config.path_str % (
                        pad_zeros(seed[0] + x, 4),
                        pad_zeros(seed[1] + y, 4),
                        pad_zeros(seed[2] + z, 4),
                        pad_zeros(seed[0] + x, 4),
                        pad_zeros(seed[1] + y, 4),
                        pad_zeros(seed[2] + z, 4))
                    v = np.fromfile(
                        path, dtype='uint8').reshape(config.shape)
                    vol[
                        z * config.shape[0]: z * config.shape[0] + config.shape[0],  # nopep8
                        y * config.shape[1]: y * config.shape[1] + config.shape[1],  # nopep8
                        x * config.shape[2]: x * config.shape[2] + config.shape[2]] = v  # nopep8
                    path = config.read_nii_path_str % (
                        pad_zeros(seed[0] + x, 4),
                        pad_zeros(seed[1] + y, 4),
                        pad_zeros(seed[2] + z, 4),
                        pad_zeros(seed[0] + x, 4),
                        pad_zeros(seed[1] + y, 4),
                        pad_zeros(seed[2] + z, 4))
                    h = nib.load(path)
                    v = h.get_fdata()
                    h.uncache()
                    del h
                    segments[
                        z * config.shape[0]: z * config.shape[0] + config.shape[0],  # nopep8
                        y * config.shape[1]: y * config.shape[1] + config.shape[1],  # nopep8
                        x * config.shape[2]: x * config.shape[2] + config.shape[2]] = v  # nopep8
        segments = rs(segments)[0].astype(np.float32)
        img = nib.Nifti1Image(vol[:max_h, :max_w, :max_d], np.eye(4))
        nib.save(img, outpath_vol.format(seed))
        seg = nib.Nifti1Image(segments[:max_h, :max_w, :max_d], np.eye(4))
        nib.save(seg, outpath_seg.format(seed))

    # Run synapse detections for this volume
    vol_mem, syn_preds, dot_preds = test(
        output_dir='synapse_predictions_v0',
        # ckpt_path='new_synapse_checkpoints_new_dataloader_smaller_weight/-85000.ckpt',  # noqa
        ckpt_path="/media/data_cifs_lrs/projects/prj_connectomics/ffn_membrane_v2/synapse_fgru_ckpts/synapse_fgru_ckpts-165000",
        paths='/media/data_cifs/connectomics/membrane_paths.npy',
        pull_from_db=False,
        keep_processing=True,
        path_extent=','.join([str(x) for x in path_extent]),
        save_preds=False,
        divs=[3, 6, 6],
        debug=True,
        segmentation_path=','.join([str(x) for x in seed]),
        out_dir=out_dir,
        finish_membranes=False,
        device="/gpu:0",
        rotate=False)
    mem = vol_mem[..., 1]
    mem = mem[:max_h, :max_w, :max_d]
    mem = nib.Nifti1Image(mem, np.eye(4))
    nib.save(mem, outpath_mem.format(seed))
    # ribbon = (syn_preds[..., 0] > 0.95).astype(np.float32)
    # ama = (syn_preds[..., 1] > 0.51).astype(np.float32)
    # ribbon = ribbon[:max_h, :max_w, :max_d]
    ribbon = syn_preds[:max_h, :max_w, :max_d]
    # ama = ama[:max_h, :max_w, :max_d]
    syn_ribbon = nib.Nifti1Image(ribbon, np.eye(4))
    # syn_ama = nib.Nifti1Image(ama, np.eye(4))
    nib.save(syn_ribbon, outpath_syn_ribbon.format(seed))
    # nib.save(syn_ama, outpath_syn_ribbon.format(seed))
    recon_ribbon = nib.Nifti1Image(dot_preds[:max_h, :max_w, :max_d], np.eye(4))
    nib.save(recon_ribbon, outpath_recon_syn_ribbon.format(seed))
