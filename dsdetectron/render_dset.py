from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from pathlib import Path
import cv2
from argparse import ArgumentParser
from sys import argv
import os
from tqdm import tqdm

import numpy as np

# register datasets
import pfa


def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--cfg", help="Detectron2 config file.")
    parser.add_argument("--dataset", help="dataset on which to perform inference")
    parser.add_argument("--outfolder", help="Output folder. ")
    return parser.parse_args(args)


def main(args):
    # retrieve configuration file and update the weights
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    # update the model so that it uses the final output weights.
    cfg.MODEL.WEIGHTS = str(Path(cfg.OUTPUT_DIR) / Path("model_final.pth"))

    predictor = DefaultPredictor(cfg)

    os.makedirs(args.outfolder, exist_ok=True)

    # load image.

    # get data from validation data
    # need to get data from the signs dataset, not the hotspots dataset.
    dset = DatasetCatalog.get(args.dataset)

    for example in tqdm(dset):
        img = cv2.imread(example["file_name"])
        # # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = predictor(img)

        outputs = predictor(img)
        # the -1 at the end reverses the color order from BGR (openCV standard) to RGB (normal standard)
        v = Visualizer(
            img[:, :, ::-1],
            metadata={"thing_classes": ["hotspot"]},
            scale=0.5,
        )

        predictions = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        cv2.imwrite(
            f"{args.outfolder}/{Path(example['file_name']).stem}.jpg",
            predictions.get_image()[:, :, ::-1],
        )


if __name__ == "__main__":
    main(parse(argv[1:]))