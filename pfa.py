import json
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog


def get_train_data():
    with open("hotspots_labeled_train.json", "r") as inf:
        dat = json.load(inf)

    for example in dat:
        for hotspot in example["annotations"]:
            hotspot["bbox_mode"] = BoxMode.XYXY_ABS

    return dat


def hotspots_only_train():
    with open("hotspots_labeled_train.json", "r") as inf:
        dat = json.load(inf)

    for example in dat:
        for hotspot in example["annotations"]:
            hotspot["bbox_mode"] = BoxMode.XYXY_ABS
            hotspot["category_id"] = 0

    return dat


def get_test_data():
    with open("hotspots_labeled_test.json", "r") as inf:
        dat = json.load(inf)

    for example in dat:
        for hotspot in example["annotations"]:
            hotspot["bbox_mode"] = BoxMode.XYXY_ABS

    return dat


def hotspots_only_test():
    with open("hotspots_labeled_test.json", "r") as inf:
        dat = json.load(inf)

    for example in dat:
        for hotspot in example["annotations"]:
            hotspot["bbox_mode"] = BoxMode.XYXY_ABS
            hotspot["category_id"] = 0

    return dat


with open("signs.txt", "r") as sgns:
    signs = [sign.strip() for sign in sgns.readlines()]

DatasetCatalog.register("signs_train", get_train_data)
DatasetCatalog.register("signs_test", get_test_data)
MetadataCatalog.get("signs_train").set(thing_classes=signs)
MetadataCatalog.get("signs_test").set(thing_classes=signs)

DatasetCatalog.register("hotspots_train", hotspots_only_train)
DatasetCatalog.register("hotspots_test", hotspots_only_test)
MetadataCatalog.get("hotspots_train").set(thing_classes=["hotspot"])
MetadataCatalog.get("hotspots_test").set(thing_classes=["hotspot"])
