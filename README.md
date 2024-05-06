# CellSeg-Transformers

Scripts to reproduce the results in the response to "Transformers do not outperform Cellpose"


## Installation 
```
# create virtual environment
conda create --name cp3 python=3.10 -y
conda activate cp3

# install the latest cellpose
git clone https://github.com/mouseland/cellpose.git
cd cellpose
pip install -e .

# install GPU version of torch
pip uninstall torch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# install segmentation_models_pytorch for Transformers
pip install segmentation_models_pytorch
pip install six pandas
```

## Fig1c: comparison between w/TTA and w/o TTA

```bash
python infer_cp_noTTA.py
```


## Fig2a: Transformer model with learning rate `0.00005`

1. Generate the flows: `python gen_flow`
2. Run the training command: `cp-trans224-5e-5.sh`
3. Download the pre-trained model [here](https://drive.google.com/file/d/13jzt2Ebil6H32heioF1RbYi-XDsAJrir/view?usp=sharing) and run the inference script: `python infer_new_trans_neurips_data.py`
4. Submit the results to the [challenge platform](https://neurips22-cellseg.grand-challenge.org/)


## Fig2b-m: New experiments on [CTC cell segmentation dataset](https://celltrackingchallenge.net/2d-datasets/)

0. Download the organized dataset [here](https://drive.google.com/file/d/1OYTxoJX_XtRwK2lNhoptFw0i_5u-HDmz/view?usp=sharing)
1. Infer CTC dataset with `Cellpose` and `Cellpose-Transformerd` trained by Dr. Carsen Stringer and Dr. Marius Pachitariu: 
`
python infer_ctc492.py
`
2. Infer CTC dataset with `Mediar`, which was the winning solution in the NeurIPS 2022 segmentation challenge
   - Download the docker [here](https://drive.google.com/file/d/1i40GEr6dRIOfkVysDz7hjNMlLMQCdUcv/view?usp=sharing)
   - Run the inference `docker container run --gpus="device=0" -m 28G --name mediar --rm -v $PWD/CTC-Data/imagesTr_GT492/:/workspace/inputs/ -v $PWD/CTC-Data/seg_mediar:/workspace/outputs/ osilab:latest /bin/bash -c "sh predict.sh"`

3. Compute Metrics: `python compute_metrics -g path_to_gt -s path_to_seg -o save_path -n save_name`
