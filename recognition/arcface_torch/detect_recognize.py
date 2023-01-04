"""
Code is based on the script from https://github.com/deepinsight/insightface/blob/cdc3d4ed5de14712378f3d5a14249661e54a03ec/python-package/insightface/utils/face_align.py
"""
# from retinaface import RetinaFace

import argparse
import sys

import cv2
import numpy as np
import torch
import glob 
import time 
from pathlib import Path
# import mxnet as mx 
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf

from backbones import get_model

from skimage import transform as trans

# sys.path.append("../../detection/retinaface")
# from retinaface import RetinaFace
# import retinaface.RetinaFace as RetinaFace

import matplotlib.pyplot as plt


# I did not delete these different alignment configurations, but you can remove them if u want
src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
#<--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

#---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

#-->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

#-->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)

def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        if image_size == 112:
            src = arcface_src
        else:
            src = float(image_size) / 112 * arcface_src
    else:
        src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

def detect_deepface(net, img, seed_name):
    print("--- img shape ---")
    print(img.shape)
    print("-----------------")
    resp = net.detect_faces(img)
    nr_keys = len(resp.keys()) # check how many faces was detected
    if nr_keys == 1:
        detected_face = resp["face_1"]
    else:
        # It should not happen that more than a single face is detected per image; however if it happens you need to support it
        # What I usually do is to take the one with the highest confidence score.
        # Hence if it happens for your images iterate through the faces and make the detected face be the one with the highest scores
        # Note: Some implementations make sure face_1 is always the one with the highest confidence

        # raise Exception("-- Several faces detected in recognizing, handle this --")
        print(f"-- Several faces detected in recognizing seed {seed_name}, handling this. --")
        min_score = -1 
        detected_face = None
        for face in resp.keys():
            print(f"Score: {resp[face]['score']}")
            if resp[face]['score'] > min_score:
                detected_face = resp[face]
        print("-- Handled multiple faces by taking the one with the largest score. --")       
    landmarks = np.array(list(detected_face['landmarks'].values()))  # represented as a 2d numpy array
    aligned_image = norm_crop(img, landmarks, image_size=112, mode='arcface')
    return aligned_image

def detect(net, img, seed_name):
    im_shape = img.shape

    thresh = 0.8
    scales = [1024, 1980]

    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)

    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    scales = [im_scale]

    faces, landmarks = net.detect(img, thresh, scales=scales, do_flip=False)

    # print(faces.shape, landmarks.shape)
    if (faces.shape[0] > 1):
        print(f" -- DETECTED MULTIPLE FACES IN DETECTION OF SEED {seed_name}, TAKING FIRST FACE -- ")

    aligned_image = norm_crop(img, landmarks[0], image_size=112, mode='arcface')
    return aligned_image


@torch.no_grad()
def recognize(net, img):
    if ((img.shape[1] != 112) | (img.shape[2] != 112)):
        img = cv2.resize(img, (112, 112))
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    feat = net(img).numpy()
    # print(feat)
    print(feat.shape)
    return feat

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector_path', type=str)
    parser.add_argument('--recognizer_path', type=str)
    parser.add_argument('--db_root', type=str)
    parser.add_argument('--folders', type=str)
    parser.add_argument('--subfolders', type=str)
    return parser.parse_args()

def save_features(base_path, detector_path, recognizer, subfolders, use_deepface=True):
    print("-"*50)
    print(f"Analyzing database {base_path}")
    if (use_deepface):
        print("## Using deepface ##")
        import retinaface.RetinaFace as RetinaFace
        detector = RetinaFace
    else: 
        print("## Using pure retinaface ##")
        sys.path.append("../../detection/retinaface")
        from retinaface import RetinaFace
        gpu_id = -1
        detector = RetinaFace(detector_path, 0, gpu_id, 'net3')

    for fold in subfolders:
        db_path = f"{base_path}/{fold}"
        print(f" - Analyzing folder: {db_path}")
        Path(f"{db_path}/features/").mkdir(parents=True, exist_ok=True)
        imgs = glob.glob(f"{db_path}/images/*")
        total_imgs = len(imgs)
        start_time = time.time()
        for i, img_path in enumerate(imgs):
            seed_name = img_path.split("/")[-1].split(".")[0]
            img = cv2.imread(img_path)  
            if (use_deepface):
                aligned_img = detect_deepface(detector, img, seed_name)
            else:
                aligned_img = detect(detector, img, seed_name)
            features = recognize(recognizer, aligned_img)
            features_path = f"{db_path}/features/{seed_name}.npy"
            np.save(features_path, features)
            if (i % 20 == 0):
                percent_finished  = round((i / total_imgs)*100, 3)
                time_taken = round(time.time() - start_time, 3)
                print(f"Finished {percent_finished}% in {time_taken} seconds.")

if __name__ == "__main__":
    print("RUNNING SCRIPT")
    args = parse_args()

    print("---- CHECKING FOR GPU ----")
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    print("--------------------------")

    detector_path = args.detector_path
    # print("yello")
    # print(mx.context.num_gpus())
    # gpus = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(',')]
    # print("gpus",gpus)
    # print("hello")
    # gpu_id = -1
    # detector = RetinaFace(detector_path, 0, gpu_id, 'net3')

    recognizer_path = args.recognizer_path
    network = 'r100'
    recognizer = get_model(network, fp16=False)
    recognizer.load_state_dict(torch.load(recognizer_path, map_location=torch.device('cuda')))
    recognizer.eval()

    db_root = args.db_root
    folders = args.folders.split(',')
    subfolders = args.subfolders.split(',')
    for folder in folders:
        path = f"{db_root}/{folder}"
        save_features(path, detector_path, recognizer, subfolders, use_deepface=True)