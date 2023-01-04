#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J pipeline
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu16gb]"
#BSUB -o logs/Output_arcface%J.out

#BSUB -e logs/Error_arcface%J.err

set -o errexit
set -o nounset

#mxnet-cu115
#mxnet-cu102==1.9.1

module load ninja/1.10.1
module load gcc/4.9.1
module load python3/3.7.11
# module load cuda/11.5
# module load cudnn/v8.3.0.98-prod-cuda-11.5
module load cuda/10.2
module load cudnn/v8.0.4.30-prod-cuda-10.2  
module load cmake/3.20.1
module load nccl/2.7.3-1-cuda-10.2          

DATASET_NAME='4k_full_pipeline'
# DATASET_NAME='200_full_pipeline'
BASE_PATH='/work3/s210521'

echo "Running script..."
echo "Extraction features from database ........"

# # Make and activate venv
python3 -m venv /work3/s210521/venvs/arcface_venv
source /work3/s210521/venvs/arcface_venv/bin/activate


# pip3 uninstall mxnet -y
# pip3 uninstall mxnet-cu102 -y 

# pip3 uninstall tensorflow
# pip3 uninstall retina-face -y
# pip3 install tensorflow-gpu==2.5.0
# pip3 install retina-face
# pip3 install -r recognition/arcface_torch/requirement.txt

# cd detection/retinaface/
# make 
# cd ../..
cd recognition/arcface_torch
python3 detect_recognize.py \
    --detector_path "/zhome/ab/6/160488/github/insightface-torch-thesis/recognition/arcface_torch/models/retinaface-R50/R50" \
    --recognizer_path "/zhome/ab/6/160488/github/insightface-torch-thesis/recognition/arcface_torch/models/ms1mv3_arcface_r100_fp16/backbone.pth" \
    --db_root "$BASE_PATH/$DATASET_NAME" \
    --folders "accepted,age_group_4,age_group_3,age_group_2,age_group_1,age_group_0" \
    --subfolders "references,probes"

echo "Finished"