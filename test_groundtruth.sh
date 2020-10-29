#!/bin/bash

python dsdetectron/groundtruth.py --cfg configs/sign_detector.yml \
                                    --dataset signs_test \
                                    --outpath gt_test