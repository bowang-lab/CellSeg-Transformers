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
import pandas as pd
from collections import OrderedDict
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

root = Path('data/wsi') # path to Testing image folder
save_path_cp = 'data/seg_public-cp-wsi'
os.makedirs(save_path_cp, exist_ok=True)

logger_setup()
# path to images
fall = natsorted(glob((root /  "*").as_posix()))
img_files = sorted([f for f in fall if "_masks" not in f and "_flows" not in f])
print('num files:', len(img_files))
print(os.system('nvidia-smi'))

running_time = OrderedDict()
running_time['names'] = []
running_time['time'] = []

model = models.Cellpose(gpu=True, nchan=3, model_type="neurips_cellpose_default")
channels = None
normalize = False
# diams = None # Cellpose will estimate diameter
diams = 30 # manually set diameter to bypass diameter estimation model


for f in tqdm(img_files):
    start_time = time.time()
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
    print(os.path.basename(f), 'img size:', img.shape)
    out = model.eval(imgs_norm, diameter=diams,
                    channels=channels, normalize=normalize, 
                    tile_overlap=0.6, augment=True)
   # predicted masks
    seg_mask = out[0][0]
    seg_name = os.path.basename(f).split('.')[0] + '_label.tiff'
    tiff.imwrite(join(save_path_cp, seg_name), seg_mask.astype(np.uint16), compression='zlib')

    end_time = time.time()
    print(os.path.basename(f), 'img size:', img.shape, 'time:', end_time-start_time)
    
    running_time['names'].append(os.path.basename(f))
    running_time['time'].append(end_time-start_time)

running_time_df = pd.DataFrame(running_time)
running_time_df.to_csv(join(save_path_cp, 'cp_running_time.csv'), index=False)
