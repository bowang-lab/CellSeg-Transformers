"""
Thanks Dr. Carsen Stringer and Dr. Marius Pachitariu for sharing the script.
"""


import torch
from pathlib import Path
from cellpose import io, dynamics 
from natsort import natsorted
from glob import glob
io.logger_setup()

root = Path("path/to/your/data/")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
iall = natsorted(glob((root / "*.tif").as_posix()))
img_files = [img for img in iall if "_masks" not in img]
print(f'{len(img_files)}')
mask_files = [img for img in iall if "_masks" in img]
masks_all = [io.imread(mask) for mask in mask_files]
dynamics.labels_to_flows(masks_all, files=img_files, return_flows=False, device=torch.device(device))
