#!/bin/bash

python dsdetectron/visualize.py --cfg configs/retinanet_18.yaml \
                                --img data/images_cropped/0a0cfbd1-ccf5-9e83-0819-716406aa14cf.jpg \
                                --outpath img_annotated_highnms