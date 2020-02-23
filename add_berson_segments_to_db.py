import os
import numpy as np
from config import Config
from db import db
import nibabel as nib
from utils.hybrid_utils import recursive_make_dir
from utils.hybrid_utils import pad_zeros


force_write = True
dry_run = False
config = Config()
paths = {
    os.path.join(config.berson_path, '384_4096_1792_curated_with_labels', 'seg_384_4096_1792.031.npy'): [384 / config.shape[0], 4096 / config.shape[1], 1792 / config.shape[2]], 
    os.path.join(config.berson_path, 'for_dustbusting', 'x0017_y0035_z0026_seg_v1.004.nii.gz'): [17, 35, 26],
}

for k, seed in paths.iteritems():
    if '.nii' in k:
        f = nib.load(k)
        segments = f.get_data()
        f.uncache()
    elif '.npy' in k:
        segments = np.load(k)
    else:
        raise NotImplementedError('Cant load {}'.format(k))
    vol_shape = np.array(segments.shape)
    path_extent = vol_shape / np.array(config.shape)

    # Get max seg ID from DB
    if not dry_run:
        segments = db.adjust_max_id(segments)

    # Save niis and update the db
    for z in range(path_extent[0]):
        for y in range(path_extent[1]):
            for x in range(path_extent[2]):
                path = config.nii_path_str % (
                    pad_zeros(seed[0] + x, 4),
                    pad_zeros(seed[1] + y, 4),
                    pad_zeros(seed[2] + z, 4),
                    pad_zeros(seed[0] + x, 4),
                    pad_zeros(seed[1] + y, 4),
                    pad_zeros(seed[2] + z, 4))
                coor_check = os.path.exists(path)
                if not coor_check or force_write:
                    print('Updating {}'.format(path))
                    seg = segments[
                        z * config.shape[0]: z * config.shape[0] + config.shape[0],
                        y * config.shape[1]: y * config.shape[1] + config.shape[1],
                        x * config.shape[2]: x * config.shape[2] + config.shape[2]]
                    if not dry_run:
                        # Update DB with coordinate info
                        db.finish_coordinate_segmentation(
                            x=x,
                            y=y,
                            z=z)

                        # Save everything
                        recursive_make_dir(path)
                        img = nib.Nifti1Image(seg, np.eye(4))
                        nib.save(img, path)
                else:
                    print('Skipping {}'.format(path))

