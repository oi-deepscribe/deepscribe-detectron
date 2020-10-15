import json
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from pathlib import Path
import functools

# data loading functions. These require the data archive to be unpacked in a
# folder called "data" in the current working directory.

# data JSON files are in the Detectron2 dataset format described
# here: https://detectron2.readthedocs.io/tutorials/datasets.html


def get_data(datapath: str, img_folder: str, hotspots_only: bool = True) -> dict:
    """
    Loads a data JSON file from the provided path.

    Updates the dictionary with the full path to the image folder,
    and optionally removes class information to provide a "hotspot-only"
    detection dataset.

    """

    with open(datapath, "r") as inf:
        dat = json.load(inf)

        for example in dat:
            # update the image filepath with the correct folder location.

            example["file_name"] = Path(img_folder) / Path(example["file_name"])

            # correctly annotate the bounding box format
            for hotspot in example["annotations"]:
                hotspot["bbox_mode"] = BoxMode.XYXY_ABS
                if hotspots_only:
                    # change the category ID. Ignore sign data.
                    hotspot["category_id"] = 0

        return dat


# load sign list.
with open("data/signs.txt", "r") as sgns:
    signs = [sign.strip() for sign in sgns.readlines()]


# register everything in the Detectron2 DatasetCatalog and MetadataCatalog.
DatasetCatalog.register(
    "signs_train",
    functools.partial(
        get_data, "data/hotspots_train.json", "data/images_cropped", hotspots_only=False
    ),
)
DatasetCatalog.register(
    "signs_val",
    functools.partial(
        get_data, "data/hotspots_val.json", "data/images_cropped", hotspots_only=False
    ),
)
DatasetCatalog.register(
    "signs_test",
    functools.partial(
        get_data, "data/hotspots_test.json", "data/images_cropped", hotspots_only=False
    ),
)
MetadataCatalog.get("signs_train").set(thing_classes=signs)
MetadataCatalog.get("signs_test").set(thing_classes=signs)
MetadataCatalog.get("signs_val").set(thing_classes=signs)


# register hotspot-only datasets.

DatasetCatalog.register(
    "hotspots_train",
    functools.partial(
        get_data, "data/hotspots_train.json", "data/images_cropped", hotspots_only=True
    ),
)
DatasetCatalog.register(
    "hotspots_val",
    functools.partial(
        get_data, "data/hotspots_val.json", "data/images_cropped", hotspots_only=True
    ),
)
DatasetCatalog.register(
    "hotspots_time",
    functools.partial(
        get_data, "data/hotspots_test.json", "data/images_cropped", hotspots_only=True
    ),
)
MetadataCatalog.get("hotspots_train").set(thing_classes=["hotspot"])
MetadataCatalog.get("hotspots_test").set(thing_classes=["hotspot"])
MetadataCatalog.get("hotspots_val").set(thing_classes=["hotspot"])
