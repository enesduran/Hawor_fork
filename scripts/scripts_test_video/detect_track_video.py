import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import argparse
import numpy as np
from glob import glob
import cv2
from lib.pipeline.tools import detect_track
from natsort import natsorted
import subprocess


def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    command = [
        'ffmpeg',               
        '-i', video_path,       
        '-vf', 'fps=30',         
        '-start_number', '0',
        os.path.join(output_folder, '%04d.jpg')  
    ]

    subprocess.run(command, check=True)


def _list_image_files(folder_path):
    valid_exts = {'.jpg', '.jpeg', '.png'}
    image_files = []
    for name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, name)
        if not os.path.isfile(file_path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in valid_exts:
            image_files.append(file_path)
    return natsorted(image_files)


def _normalize_image_folder_to_jpg(input_folder, output_folder):
    src_images = _list_image_files(input_folder)
    if len(src_images) == 0:
        raise RuntimeError(f"No input images found in folder: {input_folder}")

    existing = natsorted(glob(f'{output_folder}/*.jpg'))
    if len(existing) > 0:
        print("Skip preparing frames from image folder")
        return

    print(f'Preparing {len(src_images)} images from folder input ...')
    for idx, src_path in enumerate(src_images):
        image = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read input image: {src_path}")
        dst_path = os.path.join(output_folder, f'{idx:06d}.jpg')
        ok = cv2.imwrite(dst_path, image)
        if not ok:
            raise RuntimeError(f"Failed to write normalized frame: {dst_path}")


def build_tracks_from_gt_bboxes(gt_bboxes_path, n_frames):
    """Build a tracks dict from GT bounding boxes, bypassing YOLO detection.

    Args:
        gt_bboxes_path: Path to a .npy file containing an array of shape
            (N_frames, 2, 4) with [x1, y1, x2, y2] per hand per frame.
            Axis 1: index 0 = left hand, index 1 = right hand.
            Use np.nan to mark a hand as invisible in a given frame.
        n_frames: Total number of frames in the sequence.

    Returns:
        (boxes_, tracks_) tuple matching the output format of detect_track(),
        where tracks_ is a 0-d numpy object array wrapping a dict
        {0: [left-hand subj list], 1: [right-hand subj list]}.
    """
    gt = np.load(gt_bboxes_path, allow_pickle=True)
    if gt.ndim == 0:
        gt = gt.item()
    # gt shape: (N_frames, 2, 4)
    n_gt = len(gt)

    tracks = {}
    for hand_idx, handedness in enumerate([0, 1]):  # 0=left, 1=right
        trk = []
        for t in range(min(n_frames, n_gt)):
            bbox = gt[t, hand_idx]  # [x1, y1, x2, y2]
            has_det = not np.any(np.isnan(bbox))
            subj = {
                'frame': t,
                'det': has_det,
                'det_box': np.array([[*bbox, 1.0]]) if has_det else np.zeros((1, 5)),
                'det_handedness': np.array([handedness]),
            }
            trk.append(subj)
        tracks[hand_idx] = trk

    boxes_ = np.array([], dtype=object)
    tracks_ = np.array(tracks, dtype=object)
    return boxes_, tracks_


def detect_track_video(args):
    file = args.video_path
    norm_file = os.path.normpath(file)
    if os.path.isdir(norm_file):
        # For image-folder input, save outputs in the parent sequence folder.
        seq_folder = os.path.dirname(norm_file)
        img_folder = os.path.join(seq_folder, 'extracted_images')
    else:
        root = os.path.dirname(norm_file)
        seq = os.path.splitext(os.path.basename(norm_file))[0]
        seq_folder = f'{root}/{seq}'
        img_folder = f'{seq_folder}/extracted_images'
    os.makedirs(seq_folder, exist_ok=True)
    os.makedirs(img_folder, exist_ok=True)
    print(f'Running detect_track on {file} ...')

    ##### Extract Frames #####
    imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))
    # print(imgfiles[:10])
    if len(imgfiles) > 0:
        print("Skip extracting frames")
    else:
        if os.path.isdir(norm_file):
            _normalize_image_folder_to_jpg(norm_file, img_folder)
        else:
            _ = extract_frames(file, img_folder)
    imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))

    ##### Detection + Track #####
    print('Detect and Track ...')

    start_idx = 0
    end_idx = len(imgfiles)

    if os.path.exists(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy'):
        print(f"skip track for {start_idx}_{end_idx}")
        return start_idx, end_idx, seq_folder, imgfiles
    os.makedirs(f"{seq_folder}/tracks_{start_idx}_{end_idx}", exist_ok=True)

    gt_bboxes_path = getattr(args, 'gt_bboxes', None)
    if gt_bboxes_path is not None and os.path.isfile(gt_bboxes_path):
        print(f'Using GT bounding boxes from {gt_bboxes_path}')
        boxes_, tracks_ = build_tracks_from_gt_bboxes(gt_bboxes_path, len(imgfiles))
    else:
        boxes_, tracks_ = detect_track(imgfiles, thresh=0.2)

    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy', boxes_)
    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_tracks.npy', tracks_)

    return start_idx, end_idx, seq_folder, imgfiles

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default='')
    parser.add_argument("--input_type", type=str, default='file')
    args = parser.parse_args()

    detect_track_video(args)