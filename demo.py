import argparse
import sys
import os
import cv2
from natsort import natsorted

if os.environ.get('DISPLAY', '') == '' and os.environ.get('QT_QPA_PLATFORM') is None:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import torch
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import joblib
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam
from lib.vis.run_vis2 import run_vis2_on_video, run_vis2_on_video_cam


def _list_input_images(folder_path):
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


def get_video_resolution(video_path):
    if os.path.isdir(video_path):
        image_files = _list_input_images(video_path)
        if len(image_files) == 0:
            raise RuntimeError(f"No input images found in folder: {video_path}")
        first_img = cv2.imread(image_files[0])
        if first_img is None:
            raise RuntimeError(f"Cannot read first image from folder: {image_files[0]}")
        height, width = first_img.shape[:2]
        return width, height

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video resolution from: {video_path}")
    return width, height


def ensure_video_resolution(video_path, target_width, target_height):
    cur_width, cur_height = get_video_resolution(video_path)
    if cur_width == target_width and cur_height == target_height:
        return
    temp_video_path = video_path.replace('.mp4', '_resized.mp4')
    os.system(
        f"ffmpeg -y -i '{video_path}' "
        f"-vf 'scale={target_width}:{target_height}:flags=lanczos,setsar=1' "
        f"'{temp_video_path}'"
    )
    os.replace(temp_video_path, video_path)


def build_video_from_frames(frame_folder, output_video_path, fps=30):
    os.system(
        f"ffmpeg -y -framerate {fps} -i '{frame_folder}/%06d.jpg' "
        f"-c:v libx264 -pix_fmt yuv420p '{output_video_path}'"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--img_cx", type=float, default=None,
                        help='Principal point x (pixels). If omitted, estimated from image centre.')
    parser.add_argument("--img_cy", type=float, default=None,
                        help='Principal point y (pixels). If omitted, estimated from image centre.')
    parser.add_argument("--video_path", type=str, default='example/video_0.mp4')
    parser.add_argument("--input_type", type=str, default='file')
    parser.add_argument("--checkpoint",  type=str, default='./weights/hawor/checkpoints/hawor.ckpt')
    parser.add_argument("--infiller_weight",  type=str, default='./weights/hawor/checkpoints/infiller.pt')
    parser.add_argument("--vis_mode",  type=str, default='cam', help='cam | world')
    parser.add_argument("--gt_bboxes", type=str, default=None,
                        help='Path to a .npy file of shape (N_frames, 2, 4) with GT hand bboxes '
                             '[x1,y1,x2,y2]. Axis-1 index 0=left hand, 1=right hand. '
                             'Use np.nan to mark a hand as absent. Skips YOLO detection.')
    parser.add_argument("--static_camera", action="store_true",
                        help='Set when the camera is static (exocentric view). Skips SLAM and '
                             'uses identity camera pose for every frame.')
    args = parser.parse_args()
    input_width, input_height = get_video_resolution(args.video_path)

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)

    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

    if args.static_camera:
        # Exocentric / static-camera setup: no camera motion, skip SLAM entirely.
        N = end_idx - start_idx
        R_c2w_sla_all = torch.eye(3).unsqueeze(0).repeat(N, 1, 1)
        t_c2w_sla_all = torch.zeros(N, 3)
        R_w2c_sla_all = R_c2w_sla_all.clone()   # identity^T == identity
        t_w2c_sla_all = torch.zeros(N, 3)
    else:
        slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        if not os.path.exists(slam_path):
            hawor_slam(args, start_idx, end_idx, seq_folder)
        R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(args, start_idx, end_idx, frame_chunks_all, seq_folder)

    # vis sequence for this video
    hand2idx = {"right": 1, "left": 0}
    vis_start = 0
    vis_end = pred_trans.shape[1] 
            
    # get faces
    faces = get_mano_faces()
    faces_new = np.array([[92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79]])
    faces_right = np.concatenate([faces, faces_new], axis=0)

    # get right hand vertices
    hand = 'right'
    hand_idx = hand2idx[hand]
    pred_glob_r = run_mano(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
    right_verts = pred_glob_r['vertices'][0]
    right_dict = {
            'vertices': right_verts.unsqueeze(0),
            'faces': faces_right,
        }

    # get left hand vertices
    faces_left = faces_right[:,[0,2,1]]
    hand = 'left'
    hand_idx = hand2idx[hand]
    pred_glob_l = run_mano_left(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
    left_verts = pred_glob_l['vertices'][0]
    left_dict = {
            'vertices': left_verts.unsqueeze(0),
            'faces': faces_left,
        }

    R_x = torch.tensor([[1,  0,  0],
                        [0, -1,  0],
                        [0,  0, -1]]).float()
    R_c2w_sla_all = torch.einsum('ij,njk->nik', R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum('ij,nj->ni', R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)

    left_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, left_dict['vertices'].cpu())
    right_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, right_dict['vertices'].cpu())
    
    # Here we use aitviewer(https://github.com/eth-ait/aitviewer) for simple visualization.
    if args.vis_mode == 'world': 
        output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
        if not os.path.exists(output_pth):
            os.makedirs(output_pth)
        image_names = imgfiles[vis_start:vis_end]
        print(f"vis {vis_start} to {vis_end}")
        run_vis2_on_video(left_dict, right_dict, output_pth, img_focal, image_names, R_c2w=R_c2w_sla_all[vis_start:vis_end], t_c2w=t_c2w_sla_all[vis_start:vis_end], interactive=False, target_size=(input_width, input_height))
    elif args.vis_mode == 'cam':
        output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
        if not os.path.exists(output_pth):
            os.makedirs(output_pth)
        image_names = imgfiles[vis_start:vis_end]
        print(f"vis {vis_start} to {vis_end}")
        run_vis2_on_video_cam(left_dict, right_dict, output_pth, img_focal, image_names, R_w2c=R_w2c_sla_all[vis_start:vis_end], t_w2c=t_w2c_sla_all[vis_start:vis_end], interactive=False, target_size=(input_width, input_height))


    video_pth = os.path.join(output_pth, "aitviewer/video_0.mp4")
    ensure_video_resolution(video_pth, input_width, input_height)
    final_video_pth = os.path.join(output_pth, "aitviewer/final_vis.mp4")

    orig_video_for_concat = args.video_path
    if os.path.isdir(args.video_path):
        orig_video_for_concat = os.path.join(seq_folder, "input_from_frames.mp4")
        if not os.path.exists(orig_video_for_concat):
            build_video_from_frames(os.path.join(seq_folder, "extracted_images"), orig_video_for_concat)
 
  
    # concatenate with the original video side by side (match both stream resolutions)
    os.system(
        f"ffmpeg -y -i '{orig_video_for_concat}' -i '{video_pth}' "
        f"-filter_complex \"[0:v]scale={input_width}:{input_height}:flags=lanczos,setsar=1[orig];"
        f"[1:v]scale={input_width}:{input_height}:flags=lanczos,setsar=1[vis];"
        f"[orig][vis]hstack=inputs=2[v]\" "
        f"-map '[v]' '{final_video_pth}'")

    hand_dict = {'left': left_dict, 
                'right': right_dict, 
                'R_c2w_sla_all': R_c2w_sla_all, 't_c2w_sla_all': t_c2w_sla_all, 
                'R_w2c_sla_all': R_w2c_sla_all, 't_w2c_sla_all': t_w2c_sla_all}
    joblib.dump(hand_dict, os.path.join(seq_folder, f"hand_dict.pkl"))



# TMPDIR=/tmp/$USER singularity shell --nv hawor.sif

#  pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. 
# The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this packa
# ge or pin to Setuptools<81.   