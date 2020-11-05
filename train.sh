#!/bin/bash

export LRU_CACHE_CAPACITY=1

python dsdetectron/train_loop.py --config-file configs/sign_detector_shallow_newdata.yml 