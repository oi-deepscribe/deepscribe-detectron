#!/bin/bash

export LRU_CACHE_CAPACITY=1

python train.py --config-file train_hotspots.yml 

# python train.py --config-file train_signs.yml 

# python train.py --config-file train.yml  --resume --eval-only