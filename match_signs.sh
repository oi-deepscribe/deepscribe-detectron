#!/bin/bash


python dsdetectron/match_hotspots.py --cfg configs/retinanet_18.yaml \
                                    --dataset signs_new_train \
                                    --outpath signs_matched/train_matched

python dsdetectron/match_hotspots.py --cfg configs/retinanet_18.yaml \
                                    --dataset signs_new_val \
                                    --outpath signs_matched/val_matched


python dsdetectron/match_hotspots.py --cfg configs/retinanet_18.yaml \
                                    --dataset signs_new_test \
                                    --outpath signs_matched/test_matched