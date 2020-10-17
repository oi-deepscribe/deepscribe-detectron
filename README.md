# Deepscribe - Sign Detection 

Repo containing config files and training scripts for a sign detector based on a Faster-RCNN object detector. 

# data

Unpack the data archive into this directory. The archive contains: 

data/
+-- hotspots
|   +-- train
|   |   +-- UUID_ind_cls.jpg
|   +-- test
|   +-- val
+-- images_annotated
|   +-- UUID.jpg
+-- images_cropped
|   +-- UUID.jpg
+-- hotspots_all.json
+-- hotspots_test_coco.json
+-- hotspots_test.json
+-- hotspots_train_coco.json
+-- hotspots_train.json
+-- hotspots_val_coco.json
+-- hotspots_val.json
+-- signs.txt


# config files

The config files in `configs` are standard Detectron2 train/test configuration files. More information here: https://detectron2.readthedocs.io/tutorials/configs.html

Of particular interest are the test NMS and score threshold parameters - feel free to play with those to change results at inference time.

# training

The script `train.sh` contains an example using the script `dsdetectron/train.py`, a modification of the default Detectron2 training script. 

# example inference scripts

The scripts `dsdetectron/visualize.py` and `dsdetectron/groundtruth.py` contain examples of using trained models in inference. `dsdetectron/visualize.py` takes a tablet as input and returns an annotated image with predicted boxes drawn and a `.npz` archive containing confidence scores and individual hotspot images. 

`dsdetectron/groundtruth.py` contains an example of producing predicted hotspots as well as approximate ground-truth labels using the complete sign list. This script loads a single example from the complete validation data, runs inference from a trained model, then extracts hotspot images as well as predicted ground-truth labels from the ground-truth hotspots. This script uses functions defined in `dsdetectron/inferenceutils.py` to align the predicted boxes with ground-truth boxes. If a box cannot be aligned, its predicted label will be "-1". 