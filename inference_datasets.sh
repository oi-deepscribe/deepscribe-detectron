#!/bin/bash


python dsdetectron/render_dset.py --cfg configs/retinanet_18.yaml \
                                    --dataset hotspots_new_train \
                                    --outfolder predicted_nov_2020/train

python dsdetectron/render_dset.py --cfg configs/retinanet_18.yaml \
                                    --dataset hotspots_new_val \
                                    --outfolder predicted_nov_2020/val

python dsdetectron/render_dset.py --cfg configs/retinanet_18.yaml \
                                    --dataset hotspots_new_test \
                                    --outfolder predicted_nov_2020/test