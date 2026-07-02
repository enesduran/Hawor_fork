import os

if os.environ.get('DISPLAY', '') == '' and os.environ.get('QT_QPA_PLATFORM') is None:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import sys
import cv2
import torch
import joblib
import argparse
import numpy as np
from natsort import natsorted
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam
from lib.vis.run_vis2 import run_vis2_on_video, run_vis2_on_video_cam
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


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
    parser.add_argument("--img_focal", type=float, default=None,
                        help='Camera focal length in pixels. If omitted, HAWOR estimates/loads it.')
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
    parser.add_argument("--cam_traj", type=str, default=None,
                        help='Path to a .npy file of shape (N_video, 4, 4) holding GT '
                             'camera-to-world poses. When provided, DROID-SLAM is bypassed '
                             'and the provided trajectory (sliced to [start_idx, end_idx)) '
                             'is written into the SLAM npz that hawor_infiller consumes.')
    args = parser.parse_args()
    input_width, input_height = get_video_resolution(args.video_path)

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)

    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

    if args.static_camera:
        # Exocentric / static-camera setup: no camera motion, skip SLAM entirely.
        # hawor_infiller still reads the SLAM npz from disk, so persist an
        # identity trajectory there in the same format the cam_traj branch uses.
        N = end_idx - start_idx
        R_c2w_sla_all = torch.eye(3).unsqueeze(0).repeat(N, 1, 1)
        t_c2w_sla_all = torch.zeros(N, 3)
        R_w2c_sla_all = R_c2w_sla_all.clone()   # identity^T == identity
        t_w2c_sla_all = torch.zeros(N, 3)

        slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        if not os.path.exists(slam_path):
            quat_xyzw = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (N, 1))
            trans = np.zeros((N, 3), dtype=np.float32)
            traj = np.concatenate([trans, quat_xyzw], axis=1).astype(np.float32)
            os.makedirs(os.path.dirname(slam_path), exist_ok=True)
            np.savez(slam_path, traj=traj, scale=np.float32(1.0),
                     img_focal=np.float32(img_focal if img_focal else 0.0),
                     img_center=np.array([0.0, 0.0], dtype=np.float32))
            print(f"--static_camera: wrote identity SLAM trajectory to {slam_path}")
    else:
        slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        
        if args.cam_traj is not None:
            # GT trajectory provided — bypass DROID-SLAM and write the npz that
            # hawor_infiller (and load_slam_cam below) will read.
            
            c2w_full = np.load(args.cam_traj)
            
            if c2w_full.ndim != 3 or c2w_full.shape[1:] != (4, 4):
                raise ValueError(f"--cam_traj must be (N,4,4); got {c2w_full.shape}")
            
            if c2w_full.shape[0] < end_idx:
                raise ValueError(f"--cam_traj has {c2w_full.shape[0]} poses, need at least {end_idx}")
            
            c2w = c2w_full[start_idx:end_idx].astype(np.float64)
            quat_xyzw = Rotation.from_matrix(c2w[:, :3, :3]).as_quat()  # (N,4) xyzw
            traj = np.concatenate([c2w[:, :3, 3], quat_xyzw], axis=1).astype(np.float32)
            os.makedirs(os.path.dirname(slam_path), exist_ok=True)
            np.savez(slam_path, traj=traj, scale=np.float32(1.0),
                     img_focal=np.float32(img_focal if img_focal else 0.0),
                     img_center=np.array([0.0, 0.0], dtype=np.float32))
            print(f"Wrote GT trajectory ({traj.shape[0]} frames) to {slam_path}, DROID-SLAM bypassed")
            
        elif not os.path.exists(slam_path):
            hawor_slam(args, start_idx, end_idx, seq_folder)
  
        R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

    # MANO parameters in the SLAM *world* frame (DROID-SLAM's world, or the c2w
    # trajectory passed via --cam_traj). NOT camera space: hawor_infiller calls
    # cam2world_convert(R_c2w, t_c2w, ...) on the per-frame network outputs
    # before the temporal infill stage, then also dumps the result to
    # seq_folder/world_space_res.pth.
    #   pred_trans      (2, T, 3)   root translation [m], world-frame
    #   pred_rot        (2, T, 3)   root orientation, axis-angle, world-frame
    #   pred_hand_pose  (2, T, 45)  finger pose, axis-angle (root-relative,
    #                               frame-invariant: 15 joints x 3)
    #   pred_betas      (2, T, 10)  shape params (frame-invariant)
    #   pred_valid      (2, T)      bool per-frame validity mask
    # Hand index along dim 0: 0 = left, 1 = right (matches hand2idx below).
    # SLAM's world axes are not HOT3D's -- the R_x flip further down maps
    # the visualization dicts (only) into HOT3D convention.
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(args, start_idx, end_idx, frame_chunks_all, seq_folder)

    # vis sequence for this video
    hand2idx = {"right": 1, "left": 0}
    vis_start, vis_end = 0, pred_trans.shape[1] 
            
    # get faces
    faces_right = np.concatenate([get_mano_faces(), 
                                  np.array([[92, 38, 234],
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
                                            [78, 108, 79]])], axis=0)
    faces_left = faces_right[:,[0,2,1]]

    # get right hand vertices
    hand_idx = hand2idx['right']
    right_trans = pred_trans[hand_idx:hand_idx+1, vis_start:vis_end]
    right_rot = pred_rot[hand_idx:hand_idx+1, vis_start:vis_end]
    right_hand_pose = pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end]
    right_betas = pred_betas[hand_idx:hand_idx+1, vis_start:vis_end]
    pred_glob_r = run_mano(right_trans, right_rot, right_hand_pose, betas=right_betas)
    right_verts = pred_glob_r['vertices'][0]
    right_dict = {'vertices': right_verts.unsqueeze(0), 'faces': faces_right,
                  'trans': right_trans, 'rot': right_rot,
                  'hand_pose': right_hand_pose, 'betas': right_betas}

    # get left hand vertices
    hand_idx = hand2idx['left']
    left_trans = pred_trans[hand_idx:hand_idx+1, vis_start:vis_end]
    left_rot = pred_rot[hand_idx:hand_idx+1, vis_start:vis_end]
    left_hand_pose = pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end]
    left_betas = pred_betas[hand_idx:hand_idx+1, vis_start:vis_end]
    pred_glob_l = run_mano_left(left_trans, left_rot, left_hand_pose, betas=left_betas)
    left_verts = pred_glob_l['vertices'][0]
    
    
    left_dict = {'vertices': left_verts.unsqueeze(0), 'faces': faces_left,
                 'trans': left_trans, 'rot': left_rot,
                 'hand_pose': left_hand_pose, 'betas': left_betas}

 
    # R_x = torch.tensor([[1,  0,  0],
    #                     [0, -1,  0],
    #                     [0,  0, -1]]).float()
    R_x = torch.tensor([[1,  0,  0],
                        [0, 1,  0],
                        [0,  0, 1]]).float()
    R_c2w_sla_all = torch.einsum('ij,njk->nik', R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum('ij,nj->ni', R_x, t_c2w_sla_all)

    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)

  
    R_x_minus_I = R_x - torch.eye(3)

    def _to_hot3d_dict(side_dict, joints_world):
        trans_slam = side_dict['trans'].cpu()
        rot_slam = side_dict['rot'].cpu()
        verts_slam = side_dict['vertices'].cpu()
        J0_can = joints_world.detach().cpu()[:, :, 0, :] - trans_slam
        R_root = axis_angle_to_matrix(rot_slam)
        R_root_new = torch.einsum('ij,btjk->btik', R_x, R_root)
        side_dict['rot'] = matrix_to_axis_angle(R_root_new)
        side_dict['trans'] = (torch.einsum('ij,btj->bti', R_x, trans_slam)
                              + torch.einsum('ij,btj->bti', R_x_minus_I, J0_can))
        side_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, verts_slam)

    _to_hot3d_dict(right_dict, pred_glob_r['joints'])
    _to_hot3d_dict(left_dict, pred_glob_l['joints'])
    
    output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
    
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)
    image_names = imgfiles[vis_start:vis_end]
    
    cx = args.img_cx if args.img_cx is not None else input_width / 2
    cy = args.img_cy if args.img_cy is not None else input_height / 2
    K = np.array([[img_focal, 0, cx], [0, img_focal, cy], [0, 0, 1]])
     
    run_vis2_on_video(left_dict, right_dict, output_pth, K, image_names, 
                        R_c2w=R_c2w_sla_all[vis_start:vis_end], 
                        t_c2w=t_c2w_sla_all[vis_start:vis_end], 
                        interactive=False, 
                        target_size=(input_width, input_height))
 
    produced_video_path = os.path.join(output_pth, "video_0.mp4")
    target_video_path = os.path.join(output_pth, "world_space.mp4")
    if os.path.exists(produced_video_path):
        os.rename(produced_video_path, target_video_path)


    from aitviewer.shaders import clear_shader_cache
    clear_shader_cache()
 
    run_vis2_on_video_cam(left_dict, right_dict, output_pth, K, image_names, 
                            R_w2c=R_w2c_sla_all[vis_start:vis_end], 
                            t_w2c=t_w2c_sla_all[vis_start:vis_end], 
                            interactive=False, 
                            target_size=(input_width, input_height))
    
    produced_video_path = os.path.join(output_pth, "video_0.mp4")
    target_video_path = os.path.join(output_pth, "camera_space.mp4")
    if os.path.exists(produced_video_path):
        os.rename(produced_video_path, target_video_path)
 
    ensure_video_resolution(target_video_path, input_width, input_height)
    final_video_pth = os.path.join(output_pth, "final_vis.mp4")

    orig_video_for_concat = args.video_path
    if os.path.isdir(args.video_path):
        orig_video_for_concat = os.path.join(seq_folder, "input_from_frames.mp4")
        if not os.path.exists(orig_video_for_concat):
            build_video_from_frames(os.path.join(seq_folder, "extracted_images"), orig_video_for_concat)
 
  
    # concatenate with the original video side by side (match both stream resolutions)
    os.system(f"ffmpeg -y -i '{orig_video_for_concat}' -i '{target_video_path}' "
            f"-filter_complex \"[0:v]scale={input_width}:{input_height}:flags=lanczos,setsar=1[orig];"
            f"[1:v]scale={input_width}:{input_height}:flags=lanczos,setsar=1[vis];"
            f"[orig][vis]hstack=inputs=2[v]\" "
            f"-map '[v]' '{final_video_pth}'")

    # Suppress hands HAWOR never genuinely observed. pred_valid (2, T) is the
    # infiller's per-frame validity — now honest: a hand with no detections
    # stays all-False (see hawor_infiller). NaN out the vertices of invalid
    # frames so every downstream consumer skips them via its existing
    # finite-vertex check, and stash the mask for any consumer that wants it
    # explicitly. Applied here, after HAWOR's own debug videos are rendered, so
    # only the dumped hand_dict.pkl carries the masking. hand2idx: 1=right, 0=left.
    valid_right = torch.from_numpy(pred_valid[1, vis_start:vis_end]).bool()
    valid_left = torch.from_numpy(pred_valid[0, vis_start:vis_end]).bool()
    right_dict['vertices'][0, ~valid_right] = float('nan')
    left_dict['vertices'][0, ~valid_left] = float('nan')
    right_dict['valid'] = valid_right
    left_dict['valid'] = valid_left

    hand_dict = {'left': left_dict,
                'right': right_dict,
                'R_c2w_sla_all': R_c2w_sla_all, 
                't_c2w_sla_all': t_c2w_sla_all, 
                'R_w2c_sla_all': R_w2c_sla_all, 
                't_w2c_sla_all': t_w2c_sla_all}
    
    joblib.dump(hand_dict, os.path.join(seq_folder, f"hand_dict.pkl"))