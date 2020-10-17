from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling.matcher import Matcher
from detectron2.structures import Boxes
import torch
from pathlib import Path
import cv2
from argparse import ArgumentParser
from sys import argv

import numpy as np

# register datasets
import pfa
from inferenceutils import get_hotspots, get_gt_classes


def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--cfg", help="Detectron2 config file.")
    parser.add_argument("--outpath", help="Output path. ")
    return parser.parse_args(args)


def main(args):
    # retrieve configuration file and update the weights
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    # update the model so that it uses the final output weights.
    cfg.MODEL.WEIGHTS = str(Path(cfg.OUTPUT_DIR) / Path("model_final.pth"))

    predictor = DefaultPredictor(cfg)

    # load image.

    # get data from validation data
    # neeg to get data from the signs dataset, not the hotspots dataset.
    example_data = DatasetCatalog.get("signs_val")[0]

    img = cv2.imread(example_data["file_name"])
    # # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    outputs = predictor(img)

    # gets individual hotspot images, save to npz array

    hotspots = get_hotspots(img[:, :, ::-1], outputs["instances"].to("cpu").pred_boxes)
    # get scores
    scores = outputs["instances"].to("cpu").scores

    # get groundtruth classes

    # make Matcher object
    # these parameters can be customized.
    matcher = Matcher([0.3, 0.7], [0, -1, 1], allow_low_quality_matches=False)

    # convert the groundtruth annotations into a detectron Boxes object
    gt_boxes = Boxes(
        torch.tensor(
            np.vstack(
                [annotation["bbox"] for annotation in example_data["annotations"]]
            )
        )
    )

    gt_classes = np.array(
        [annotation["category_id"] for annotation in example_data["annotations"]]
    )

    aligned_classes = get_gt_classes(
        outputs["instances"].to("cpu").pred_boxes, matcher, gt_boxes, gt_classes
    )

    np.savez(
        Path(args.outpath).with_suffix(".npz"),
        hotspots=hotspots,
        scores=scores,
        gt_classes=aligned_classes,
    )


if __name__ == "__main__":
    main(parse(argv[1:]))