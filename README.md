# Deepscribe - Detectron2 

Early experiments using the FAIR Detectron2 library for object detection. The library works very well with minimal configuration, but adding modified metrics (top-k precision, recall, and accuracy in our case) proved to be difficult due to the internal detection API. Currently experimenting with a RetinaNet based SSD here: https://github.com/oi-deepscribe/deepscribe-detect. 

In the meantime, this may be useful for performing sign-only detection.

# Setup

Install conda env with `conda env create -f environment.yml`. 


# Preprocessing 

The data was preprocessed from the raw hotspots using the notebooks `reformat.ipynb` and `crop.ipynb`. The former converts the hotspots to the Detectron2 dataset format, and the latter crops images such that unlabeled regions of the image are removed. 

# Datasets

`pfa.py` defines two sets of data: one where each sign is given its own category, and another where they're all lumped into one "sign" category. 

# Training

Pretty much just the default detectron2 train API - the train.py script was copied from the official detectron2 repo. 