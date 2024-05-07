import os
import shutil
import time
import torch
join = os.path.join
import tifffile as tif
import pandas as pd
from collections import OrderedDict

test_img_path = 'data/wsi' 
temp_in = 'inputs/'
temp_out = 'outputs/'
os.makedirs(temp_in, exist_ok=True)
os.makedirs(temp_out, exist_ok=True)
os.system("chmod -R 777 outputs/")

test_cases = sorted(os.listdir(test_img_path))
test_cases = [case for case in test_cases if case.endswith('.tif') or case.endswith('.tiff')]
print('num of test cases:', len(test_cases))

running_time = OrderedDict()
running_time['names'] = []
running_time['time'] = []

for case in test_cases:
    shutil.copy(join(test_img_path, case), temp_in)
    img_data = tif.imread(join(temp_in, case))
    start_time = time.time()
    print(f"{case} img size {img_data.shape}")
    os.system('docker container run --gpus "device=0" --name mediar --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ osilab:latest /bin/bash -c "sh predict.sh"')
    end_time = time.time()
    print(f"{case} img size {img_data.shape} cost time: {end_time - start_time}")
    running_time['names'].append(case)
    running_time['time'].append(end_time - start_time)

    os.remove(join('./inputs', case))

torch.cuda.empty_cache()
shutil.rmtree(temp_in)

running_time_df = pd.DataFrame(running_time)
running_time_df.to_csv('mediar_running_time.csv', index=False)