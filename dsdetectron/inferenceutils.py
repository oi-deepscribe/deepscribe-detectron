from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.modeling.matcher import Matcher
import numpy as np
from typing import List


def extract_boxes(
    image: np.ndarray,
    boxes: Boxes,
) -> List[np.ndarray]:
    # extracts invididual hotspot instances from predicted regions.
    # optionally, aligns them to existing gt and produce groundtruth classes
    # from the full class set.
    # TODO: determine best way to deal with false positives.

    # extract hotspot for each bbox in instances

    hotspots = []
    for i in range(len(boxes)):
        bbox = [int(v) for v in boxes.tensor[i, :].tolist()]
        hotspot = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        hotspots.append(hotspot)

    return hotspots


# def get_gt_classes(
#     boxes: Boxes,
#     matcher: Matcher,
#     gt_boxes: Boxes,
#     gt_classes: np.ndarray,
# ):

#     match_quality_matrix = pairwise_iou(gt_boxes, boxes)
#     matched_idxs, matched_labels = matcher(match_quality_matrix)

#     # compute ground-truth classes for every box
#     aligned_classes = gt_classes[matched_idxs]

#     # handle edge case where only one aligned box shows up
#     if not isinstance(aligned_classes, np.ndarray):
#         aligned_classes = np.ndarray([aligned_classes])

#     # handle background classes:
#     aligned_classes[matched_labels == 0] = -1
#     aligned_classes[matched_labels == -1] = -1

#     return aligned_classes