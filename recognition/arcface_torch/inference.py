import argparse
import sys

import cv2
import numpy as np
import torch

from backbones import get_model

sys.path.append("../../detection/retinaface")
from retinaface import RetinaFace

@torch.no_grad()
def inference(weight, name, img):
    img = cv2.imread(img)
    img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight, map_location=torch.device('cpu')))
    net.eval()
    feat = net(img).numpy()
    print(feat)
    print(feat.shape)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    # parser.add_argument('--network', type=str, default='r50', help='backbone network')
    # parser.add_argument('--weight', type=str, default='')
    # parser.add_argument('--img', type=str, default=None)
    # args = parser.parse_args()
    # inference(args.weight, args.network, args.img)

    weight = '/Users/andersbensen/Documents/github/insightface/recognition/arcface_torch/models/ms1mv3_arcface_r100_fp16/backbone.pth'
    network = 'r100'
    img = '/Users/andersbensen/Documents/university/dtu/4sem/master_thesis/database/200_full_pipeline/accepted/references/images/114485_40_M_middle eastern.png'


    inference(weight, network, img)
