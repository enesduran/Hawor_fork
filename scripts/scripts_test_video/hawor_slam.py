import os
import sys
import cv2
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from glob import glob
import imageio.v3 as iio
from natsort import natsorted

sys.path.insert(0, os.path.dirname(__file__) + '/../..')

from lib.pipeline.est_scale import *
from pycocotools import mask as masktool
from lib.pipeline.masked_droid_slam import *
from hawor.utils.process import block_print, enable_print

sys.path.insert(0, os.path.dirname(__file__) + '/../../thirdparty/Metric3D')

from metric import Metric3D


def get_all_mp4_files(folder_path):
    # Ensure the folder path is absolute
    folder_path = os.path.abspath(folder_path)
    
    # Recursively search for all .mp4 files in the folder and its subfolders
    mp4_files = glob(os.path.join(folder_path, '**', '*.mp4'), recursive=True)
    
    return mp4_files

def split_list_by_interval(lst, interval=1000):
    start_indices = []
    end_indices = []
    split_lists = []
    
    for i in range(0, len(lst), interval):
        start_indices.append(i)
        end_indices.append(min(i + interval, len(lst)))
        split_lists.append(lst[i:i + interval])
    
    return start_indices, end_indices, split_lists

def hawor_slam(args, start_idx, end_idx, seq_folder):
    # File and folders
    os.makedirs(seq_folder, exist_ok=True)
    video_folder = seq_folder

    img_folder = f'{video_folder}/extracted_images'
    imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))

    first_img = cv2.imread(imgfiles[0])
    height, width, _ = first_img.shape
    
    print(f'Running slam on {video_folder} ...')

    ##### Run SLAM #####
    # Use Masking
    masks = np.load(f'{video_folder}/tracks_{start_idx}_{end_idx}/model_masks.npy', allow_pickle=True)
    masks = torch.from_numpy(masks)
    print(masks.shape)

    # Camera calibration (intrinsics) for SLAM
    focal = args.img_focal
    if focal is None:
        try:
            with open(os.path.join(video_folder, 'est_focal.txt'), 'r') as file:
                focal = file.read()
                focal = float(focal)
        except:
            
            print('No focal length provided')
            focal = 600
            with open(os.path.join(video_folder, 'est_focal.txt'), 'w') as file:
                file.write(str(focal))
    calib = np.array(est_calib(imgfiles)) # [focal, focal, cx, cy]
    cx = getattr(args, 'img_cx', None)
    cy = getattr(args, 'img_cy', None)
    if cx is not None and cy is not None:
        calib[2], calib[3] = cx, cy
    center = calib[2:]
    calib[:2] = focal

    import ipdb; ipdb.set_trace()
    
    # Droid-slam with masking
    droid, traj = run_slam(imgfiles, masks=masks, calib=calib)
    n = droid.video.counter.value
    tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
    disps = droid.video.disps_up.cpu().numpy()[:n]
    print('DBA errors:', droid.backend.errors)

    del droid
    torch.cuda.empty_cache()

    # Estimate scale  
    block_print()  
    metric = Metric3D('thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth') 
    enable_print() 
    min_threshold = 0.4
    max_threshold = 0.7

    print('Predicting Metric Depth ...')
    pred_depths = []
    H, W = get_dimention(imgfiles)
    for t in tqdm(tstamp):
        pred_depth = metric(imgfiles[t], calib)
        pred_depth = cv2.resize(pred_depth, (W, H))
        pred_depths.append(pred_depth)

    pred_depths = np.stack(pred_depths, axis=0)

    ##### Estimate Metric Scale #####
    print('Estimating Metric Scale ...')
    scales_ = []
    n = len(tstamp)   # for each keyframe
    for i in tqdm(range(n)):
        t = tstamp[i]
        disp = disps[i]
        pred_depth = pred_depths[i]
        slam_depth = 1/disp
        
        # Estimate scene scale
        msk = masks[t].numpy().astype(np.uint8)
        scale = est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=msk, near_thresh=min_threshold, far_thresh=max_threshold)  
        while math.isnan(scale):
            min_threshold -= 0.1
            max_threshold += 0.1
            scale = est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=msk, near_thresh=min_threshold, far_thresh=max_threshold)                    
        scales_.append(scale)

  
    

    #  normalize for visualization
    # pred_depths_vis = (pred_depths - pred_depths.min()) / (pred_depths.max() - pred_depths.min()) * 255 

    # iio.imwrite(f"deneme.mp4",
    #         pred_depths_vis.astype(np.uint8), 
    #         fps=30, codec="libx264", quality=8,
    #         macro_block_size=1)

    median_s = np.median(scales_)
    print(f"estimated scale: {median_s}")

    # Save results
    os.makedirs(f"{seq_folder}/SLAM", exist_ok=True)
    save_path = f'{seq_folder}/SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz'
    np.savez(save_path, 
            tstamp=tstamp, disps=disps, traj=traj, 
            img_focal=focal, img_center=calib[-2:],
            scale=median_s)    







