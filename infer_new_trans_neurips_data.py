"""
The code was modified from 
https://github.com/MouseLand/cellpose/blob/main/paper/neurips/analysis.py
Thanks Dr. Carsen Stringer and Dr. Marius Pachitariu for sharing the code.
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
from collections import OrderedDict
import pandas as pd


root = Path('data/test_images') # path to Testing image folder
save_path = './data'
save_path_cp = join(save_path, 'seg_new-cp')
save_path_cp_trans_repair_lr = join(save_path, 'seg_new-trans-repair-lr')
os.makedirs(save_path_cp, exist_ok=True)
os.makedirs(save_path_cp_trans_repair_lr, exist_ok=True)

diameter_dict = OrderedDict()
diameter_dict['names'] = []
diameter_dict['diams'] = []

logger_setup()
# path to images
files = natsorted(glob((root /  "*").as_posix()))
img_files = [f for f in files if "_masks" not in f and "_flows" not in f]

for f in tqdm(img_files):
    # load images
    imgs = [io.imread(f)]

    # for 3 channel model, normalize images and convert to 3 channels if needed
    imgs_norm = []
    for img in tqdm(imgs):
        if img.ndim==2:
            img = np.tile(img[:,:,np.newaxis], (1,1,3))
        img = transforms.normalize_img(img, axis=-1)
        imgs_norm.append(img.transpose(2,0,1))

    dat = {}
    for mtype in ["default", "transformer_repair_lr"]:
        if mtype=="default":
            model = models.Cellpose(gpu=True, nchan=3, model_type="neurips_cellpose_default")
            channels = None
            normalize = False
            diams = None # Cellpose will estimate diameter
        elif mtype=="transformer_repair_lr":
            model_ckpt = 'cp-trans224-5en5'
            model = model = models.Cellpose(gpu=True, nchan=3, model_type=model_ckpt, backbone="transformer")
            channels = None 
            normalize = False
            diams = dat["diams_pred"] # to fairly compare the network backbone, we also used the predicted diameter from cellpose

        out = model.eval(imgs_norm, diameter=diams,
                        channels=channels, normalize=normalize, 
                        tile_overlap=0.6, augment=True)
        
        # predicted masks
        seg_mask = out[0][0]
        if mtype=="default":
            diams = out[-1]
            dat["diams_pred"] = diams
            diameter_dict['names'].append(os.path.basename(f))
            diameter_dict['diams'].append(out[-1])

        seg_name = os.path.basename(f).split('.')[0] + '_label.tiff'
        if mtype=="default":
            tiff.imwrite(join(save_path_cp, seg_name), seg_mask.astype(np.uint16), compression='zlib')
        elif mtype=="transformer_repair_lr":
            tiff.imwrite(join(save_path_cp_trans_repair_lr, seg_name), seg_mask.astype(np.uint16), compression='zlib')
        
diameter_df = pd.DataFrame(diameter_dict)
diameter_df.to_csv(join(save_path_cp, 'diameters.csv'), index=False)