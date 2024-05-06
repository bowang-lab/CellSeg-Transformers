"""
The code was modified from 
https://github.com/MouseLand/cellpose/blob/main/paper/neurips/analysis.py
Thanks Dr. Carsen Stringer and Dr. Marius Pachitariu for sharing the code.

Usage:
1. Download the data
https://neurips22-cellseg.grand-challenge.org/dataset/
2. Set the root path to the testing data folder 

"""

import os
join = os.path.join
import numpy as np
from cellpose import io, transforms, models
from natsort import natsorted 
from pathlib import Path
from glob import glob
from tqdm import tqdm
from cellpose.io import logger_setup
import tifffile as tiff


root = Path('data/test_images') # path to Testing image folder
save_path_cp = 'data/public-models/seg_public-cp-noTTA'
os.makedirs(save_path_cp, exist_ok=True)


logger_setup()
# path to images
fall = natsorted(glob((root /  "*").as_posix()))
img_files = sorted([f for f in fall if "_masks" not in f and "_flows" not in f])

for f in tqdm(img_files):
    # load images
    imgs = [io.imread(f)]
    nimg = len(imgs)

    # for 3 channel model, normalize images and convert to 3 channels if needed
    imgs_norm = []
    for img in tqdm(imgs):
        if img.ndim==2:
            img = np.tile(img[:,:,np.newaxis], (1,1,3))
        img = transforms.normalize_img(img, axis=-1)
        imgs_norm.append(img.transpose(2,0,1))

    model = models.Cellpose(gpu=True, nchan=3, model_type="neurips_cellpose_default")
    channels = None
    normalize = False
    diams = None # Cellpose will estimate diameter

    out = model.eval(imgs_norm, diameter=diams,
                    channels=channels, normalize=normalize, 
                    tile_overlap=0.6, augment=False)
    # predicted masks
    seg_mask = out[0][0]
    seg_name = os.path.basename(f).split('.')[0] + '_label.tiff'
    tiff.imwrite(join(save_path_cp, seg_name), seg_mask.astype(np.uint16), compression='zlib')
