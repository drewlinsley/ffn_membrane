import numpy as np
from db import db
from hybrid_inference import get_segmentation
from skimage import measure
import argparse
import pandas as pd


def get_new_coors(x, y, z, next_direction, stride):
    """Consolidate this ugly conditional for iterating coordinates."""
    if next_direction == 0:
        x += stride[0]
    elif next_direction == 1:
        y += stride[1]
    elif next_direction == 2:
        z += stride[2]
    elif next_direction == 0:
        x -= stride[0]
    elif next_direction == 1:
        y -= stride[1]
    elif next_direction == 2:
        z -= stride[2]
    return x, y, z


def main(
        move_threshold=0.7,
        segment_threshold=0.5,
        deltas='[15, 15, 3]',
        path_extent=[3, 1, 1],  # x/y/z 128 voxel cube extent
        seed_policy='PolicyMembrane',
        seg_ordering=[2, 1, 0],  # transpose to z/y/x for segmentation
        stride=[2, 1, 1]):  # x/y/z
    """Run a worker by pulling volume info from the DB."""
    path_extent = np.array(path_extent)
    stride = np.array(stride)
    next_coordinate = db.get_next_coordinate(
        path_extent=path_extent,
        stride=stride)
    if next_coordinate is None:
        # No need to process this point
        return
    x, y, z, chain_id, prev_chain_idx, prev_coordinate = next_coordinate

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
        print('Failed segmentation: %s' % e)
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
        probability_faces = np.stack([
            probabilities[:, :, 0].max(),  # x <- +z
            probabilities[:, 0, :].max(),  # y <- +y
            probabilities[0, :, ].max(),  # z <- +x
            probabilities[:, :, -1].max(),  # -z
            probabilities[:, -1, :].max(),  # -y
            probabilities[-1, :, :].max(),  # -x
        ], 0)
        # probability_faces = probability_faces.max(-1)
        supra_threshold_faces = probability_faces > 128
        if np.any(supra_threshold_faces):
            next_direction = np.argmax(probability_faces)
            new_x, new_y, new_z = get_new_coors(
                x=x,
                y=y,
                z=z,
                next_direction=next_direction,
                stride=stride)
            priority = pd.DataFrame(np.array([
                new_x,
                new_y,
                new_z,
                'auto',
                None,
                True,
                prev_chain_idx + 1,
                chain_id]).reshape(1, -1),
                columns=[
                    'x',
                    'y',
                    'z',
                    'quality',
                    'location',
                    'force',
                    'prev_chain_idx',
                    'chain_id',
            ])
            db.add_priorities(priority)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    # args = parser.parse_args()
    # main(**vars(args))
    main()
