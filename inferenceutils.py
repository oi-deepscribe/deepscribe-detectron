from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.modeling.matcher import Matcher
import numpy as np


def get_hotspots(
    image: np.ndarray,
    boxes: Boxes,
    matcher: Matcher = None,
    gt_boxes: Boxes = None,
    gt_classes: np.ndarray = None,
):
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

    if gt_boxes is None:
        return hotspots

    match_quality_matrix = pairwise_iou(gt_boxes, boxes)
    matched_idxs, matched_labels = matcher(match_quality_matrix)

    # compute ground-truth classes for every box
    aligned_classes = gt_classes[matched_idxs]
    # handle background classes
    aligned_classes[matched_labels == 0] = -1

    return hotspots, aligned_classes