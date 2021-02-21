# Matched Signs

Each .npz archive in this folder contains the following keys:
- hotspots: a list of 2D image arrays. These are the predicted hotspots by the model initialized from `configs/retinanet_18.yaml`
- scores: the confidence score of each hotspot
- gt_classes: the ground-truth class ID as determined by the Matcher in `dsdetectron/match_hotspots.py`. Signs with IoU < 0.5 from any ground-truth signs are marked as "-1", meaning a false positive. 
