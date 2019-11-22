import logging
# import sys
# import os
import numpy as np
from db import db
from hybrid_inference import get_segmentation
from skimage import measure
# import argparse
import pandas as pd
# from config import Config


def get_new_coors(x, y, z, next_direction, stride):
    """Consolidate this ugly conditional for iterating coordinates."""
    if next_direction == 0:
        x -= stride[0]
    elif next_direction == 1:
        y -= stride[1]
    elif next_direction == 2:
        z -= stride[2]
    elif next_direction == 3:
        x += stride[0]
    elif next_direction == 4:
        y += stride[1]
    elif next_direction == 5:
        z += stride[2]
    return x, y, z


def main(
        move_threshold=0.7,
        segment_threshold=0.5,
        deltas='[15, 15, 3]',
        path_extent=[5, 5, 5],  # [3, 3, 2],  # x/y/z 128 voxel cube extent
        seed_policy='PolicyMembrane',
        seg_ordering=[2, 1, 0],  # transpose to z/y/x for segmentation
        offset=[32, 32, 8],  # Should match FOV in FFN
        stride=[3, 3, 3]):  # [2, 1, 1# x/y/z
    """Run a worker by pulling volume info from the DB."""
    # config = Config()
    path_extent = np.array(path_extent)
    stride = np.array(stride)
    next_coordinate = db.get_next_coordinate(
        path_extent=path_extent,
        stride=stride)
    if next_coordinate is None:
        # No need to process this point
        return
    (
        x,
        y,
        z,
        chain_id,
        prev_chain_idx,
        is_priority,
        prev_coordinate) = next_coordinate
    # logging.basicConfig(
    #     # stream=sys.stdout,
    #     level=logging.INFO,
    #     filename=os.path.join(config.errors, '{}_{}_{}'.format(x,y,z)))
    # logging.getLogger().addHandler(logging.StreamHandler())

    # Run segmentation
    try:
        success, segments, probabilities = get_segmentation(
            idx=0,  # Force membrane detection
            seed=None,
            move_threshold=move_threshold,
            segment_threshold=segment_threshold,
            x=x,
            y=y,
            z=z,
            prev_coordinate=prev_coordinate,
            deltas=deltas,
            path_extent=path_extent[[seg_ordering]],
            seed_policy=seed_policy)
    except Exception as e:
        logging.exception('Failed segmentation: {}'.format(e))
        print('Failed segmentation: {}'.format(e))
        success = False

    # Update DB with results
    if success:
        seg_props = measure.regionprops(segments.astype(np.uint64))
        segs_ids = np.array([rx.label for rx in seg_props])
        segs_areas = np.array([rx.area for rx in seg_props])
        id_areas = np.stack((segs_ids, segs_areas), 1)
        id_areas = id_areas[id_areas[:, 0] != 0]
        id_area_dict = []
        for seg_id, seg_area in id_areas:
            # Allows for segments to grow under new _ids
            id_area_dict += [{
                'segment_id': seg_id,
                'size': seg_area,
                'x': x,
                'y': y,
                'z': z,
                'chain_id': chain_id
            }]
        db.insert_segments(id_area_dict)
        db.update_config_segments_chain(len(id_area_dict))
        db.finish_coordinate(x=x, y=y, z=z)

        # Chains start here.
        # Check if any face has probability > 0.5 (255 / 2 = 128 = 0.5).
        # If so add this face to priority list and continue chain.
        probabilities = probabilities.transpose(seg_ordering)
        probability_faces = np.stack([
            probabilities[offset[0], :, :].max(),  # -x
            probabilities[:, offset[1], :].max(),  # -y
            probabilities[:, :, offset[2]].max(),  # -z
            probabilities[-offset[0], :, :].max(),  # +x
            probabilities[:, -offset[1], :].max(),  # +y
            probabilities[:, :, -offset[2]].max(),  # +z
        ], 0)
        supra_threshold_faces = probability_faces > 128
        stride_check = np.concatenate((stride, stride)) > 0
        supra_threshold_faces = np.logical_and(
            supra_threshold_faces,
            stride_check)
        if np.any(supra_threshold_faces):
            next_direction = np.argmax(
                probability_faces * supra_threshold_faces.astype(float))
            new_x, new_y, new_z = get_new_coors(
                x=x,
                y=y,
                z=z,
                next_direction=next_direction,
                stride=stride)
            columns = [
                'x',
                'y',
                'z',
                'quality',
                'location',
                'force',
                'prev_chain_idx',
                'chain_id',
            ]
            # If this is a new random coordinate, we need to first add it
            # to the priority table before the new one.
            if not is_priority:
                priority = pd.DataFrame(
                    np.array([
                        x,
                        y,
                        z,
                        'auto',
                        'origin',
                        True,
                        0,
                        chain_id]).reshape(1, -1),
                    columns=columns)
                db.add_priorities(priority)
            priority = pd.DataFrame(
                np.array([
                    new_x,
                    new_y,
                    new_z,
                    'auto',
                    None,
                    True,
                    prev_chain_idx + 1,
                    chain_id]).reshape(1, -1),
                columns=columns)
            db.add_priorities(priority)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    # args = parser.parse_args()
    # main(**vars(args))
    main()
