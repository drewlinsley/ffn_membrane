from db import db
from hybrid_inference import get_segmentation
from skimage import measure
import argparse
import pandas as pd


def main(
        move_threshold=0.7,
        segment_threshold=0.5,
        deltas='[15, 15, 3]',
        path_extent=[1, 1, 1],  # [3, 3, 3],  # 384 voxel cube extent
        seed_policy='PolicyMembrane',
        stride=[2, 2, 2]):
    """Run a worker by pulling volume info from the DB."""
    next_coordinate = db.get_next_coordinate(
        path_extent=path_extent,
        stride=stride)
    if next_coordinate is None:
        # No need to process this point
        return
    x, y, z, chain_id, prev_coordinate = next_coordinate

    # Run segmentation
    chain_id = db.get_max_chain_id() + 1
    db.update_max_chain_id(chain_id)
    try:
        success, segments, probabilities = get_segmentation(
            idx=0,  # Force membrane detection
            move_threshold=move_threshold,
            segment_threshold=segment_threshold,
            x=x,
            y=y,
            z=z,
            prev_coordinate=prev_coordinate,
            deltas=deltas,
            seed_policy=seed_policy)
    except Exception as e:
        print('Failed segmentation: %s' % e)
        success = False

    import ipdb;ipdb.set_trace()
    # Update DB with results
    if success:
        seg_props = measure.regionprops(segments.astype(np.uint64))
        segs_ids = np.array([x.label for x in seg_props])
        segs_areas = np.array([x.area for x in seg_props])
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
        db.update_config_segments_chain(segments=len(id_areas))
        db.finish_coordinate(x=x, y=y, z=z)

        # Chains start here.
        # Check if any face has probability > 0.5 (255 / 2 = 128 = 0.5).
        # If so add this face to priority list and continue chain. 
        probability_faces = np.array([
            probabilities[0, :, :],
            probabilities[:, 0, :],
            probabilities[:, :, 0],
            probabilities[-1, :, :],
            probabilities[:, -1, :],
            probabilities[:, :, -1],
        ])
        supra_threshold_faces = probability_faces > 128
        if np.any(supra_threshold_faces):
            next_direction = np.argmax(probability_faces)
            priority = pd.DataFrame.from_dict({
                'x': new_x,
                'y': new_y,
                'z': new_z,
                'quality': 'auto',
                'location': None,
                'force': True,
                'prev_chain_idx': prev_chain_idx + 1,
                'chain_id': chain_id})
            db.add_priorities(priority)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    # args = parser.parse_args()
    # main(**vars(args))
    main()

