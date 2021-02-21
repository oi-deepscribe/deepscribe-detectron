from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from pathlib import Path
import cv2
from argparse import ArgumentParser
from sys import argv

import numpy as np

# register datasets
import pfa

# from inferenceutils import get_hotspots


def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--cfg", help="Detectron2 config file.")
    parser.add_argument("--img", help="input image")
    parser.add_argument("--outpath", help="root path for outputs. ")
    return parser.parse_args(args)


def main(args):
    # retrieve configuration file and update the weights
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    # update the model so that it uses the final output weights.
    cfg.MODEL.WEIGHTS = str(Path(cfg.OUTPUT_DIR) / Path("model_final.pth"))

    predictor = DefaultPredictor(cfg)

    # load image.

    img = cv2.imread(args.img)
    # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    outputs = predictor(img)
    # the -1 at the end reverses the color order from BGR (openCV standard) to RGB (normal standard)
    v = Visualizer(
        img[:, :, ::-1],
        metadata={"thing_classes": ["hotspot"]},
        scale=0.5,
    )

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2.imwrite(
        str(Path(args.outpath).with_suffix(".jpg")),
        out.get_image()[:, :, ::-1],
    )

    # # gets individual hotspot images, save to npz array

    # hotspots = get_hotspots(img[:, :, ::-1], outputs["instances"].to("cpu").pred_boxes)
    # # get scores
    # scores = outputs["instances"].to("cpu").scores

    # np.savez(Path(args.outpath).with_suffix(".npz"), hotspots=hotspots, scores=scores)


if __name__ == "__main__":
    main(parse(argv[1:]))