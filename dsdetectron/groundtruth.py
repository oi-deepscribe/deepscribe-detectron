from argparse import ArgumentParser
from pathlib import Path
from sys import argv

import cv2
import numpy as np
import torch
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.modeling.matcher import Matcher
from detectron2.structures import Boxes
from detectron2.utils.visualizer import Visualizer

# register datasets
import pfa
from inferenceutils import get_gt_classes, get_hotspots


def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--cfg", help="Detectron2 config file.")
    parser.add_argument("--dataset", help="dataset on which to perform inference")
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
    # need to get data from the signs dataset, not the hotspots dataset.
    dset = DatasetCatalog.get(args.dataset)

    all_hotspots = []
    all_gt_aligned = []
    all_scores = []

    for example in tqdm(dset):

        img = cv2.imread(example["file_name"])
        # # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = predictor(img)

        # gets individual hotspot images, save to npz array

        hotspots = get_hotspots(
            img[:, :, ::-1], outputs["instances"].to("cpu").pred_boxes
        )
        all_hotspots.extend(hotspots)

        # get scores
        scores = outputs["instances"].to("cpu").scores

        all_scores.extend(scores.numpy())

        # get groundtruth classes

        # make Matcher object
        # these parameters can be customized.
        matcher = Matcher([0.3, 0.7], [0, -1, 1], allow_low_quality_matches=False)

        # convert the groundtruth annotations into a detectron Boxes object
        gt_boxes = Boxes(
            torch.tensor(
                np.vstack([annotation["bbox"] for annotation in example["annotations"]])
            )
        )

        gt_classes = np.array(
            [annotation["category_id"] for annotation in example["annotations"]]
        )

        aligned_classes = get_gt_classes(
            outputs["instances"].to("cpu").pred_boxes, matcher, gt_boxes, gt_classes
        )

        all_gt_aligned.extend(aligned_classes)

    np.savez(
        Path(args.outpath).with_suffix(".npz"),
        hotspots=np.array(all_hotspots, dtype=object),
        scores=all_scores,
        gt_classes=all_gt_aligned,
    )


if __name__ == "__main__":
    main(parse(argv[1:]))
