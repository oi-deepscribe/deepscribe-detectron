from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
    PascalVOCDetectionEvaluator,
)
from detectron2.data import build_detection_test_loader
import os
import cv2
import json
from copy import deepcopy
import random
from pfa import get_test_data
from tqdm import trange

# register datasets

cfg = get_cfg()
# using full network - may just be a way to just use RPN
cfg.merge_from_file("singlescale.yml")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

predictor = DefaultPredictor(cfg)

test_dataset = get_test_data()

with open("signs.txt", "r") as sgns:
    signs = [sign.strip() for sign in sgns.readlines()]

infdir = os.path.join(cfg.OUTPUT_DIR, "test_inference")

# make output directory
os.mkdir(infdir)

for d in trange(0, 100):
    im = cv2.imread(test_dataset[d]["file_name"])
    outputs = predictor(
        im
    )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(
        im[:, :, ::-1],
        metadata={"thing_classes": signs},
        scale=0.5,
    )

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2.imwrite(
        f"{infdir}/test{d}.jpg",
        out.get_image()[:, :, ::-1],
    )
