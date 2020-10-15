#!/bin/bash

export LRU_CACHE_CAPACITY=1

python dsdetectron/train.py --config-file configs/sign_detector.yml 