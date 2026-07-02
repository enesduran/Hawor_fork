import os
import os.path as op
import re
from abc import abstractmethod

import matplotlib.cm as cm
import numpy as np
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.scene.material import Material
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.viewer import Viewer
from easydict import EasyDict as edict
from loguru import logger
from PIL import Image
from tqdm import tqdm
import random

OBJ_ID = 100
SMPLX_ID = 150
LEFT_ID = 200
RIGHT_ID = 250
SEGM_IDS = {"object": OBJ_ID, "smplx": SMPLX_ID, "left": LEFT_ID, "right": RIGHT_ID}

cmap = cm.get_cmap("plasma")
materials = {
    "white": Material(color=(1.0, 1.0, 1.0, 1.0), ambient=0.2),
    "green": Material(color=(0.0, 1.0, 0.0, 1.0), ambient=0.2),
    "blue": Material(color=(0.0, 0.0, 1.0, 1.0), ambient=0.2),
    "red": Material(color=(0.969, 0.106, 0.059, 1.0), ambient=0.2),
    "cyan": Material(color=(0.051, 0.659, 0.051, 1.0), ambient=0.2),
    "light-blue": Material(color=(0.588, 0.5647, 0.9725, 1.0), ambient=0.2),
    "cyan-light": Material(color=(0.051, 0.659, 0.051, 1.0), ambient=0.2),
    "dark-light": Material(color=(0.404, 0.278, 0.278, 1.0), ambient=0.2),
    "rice": Material(color=(0.922, 0.922, 0.102, 1.0), ambient=0.2),
    "whac-whac": Material(color=(167/255, 193/255, 203/255, 1.0), ambient=0.2),
    "whac-wham": Material(color=(165/255, 153/255, 174/255, 1.0), ambient=0.2),
    "pace-blue": Material(color=(0.584, 0.902, 0.976, 1.0), ambient=0.2),
    "pace-green": Material(color=(0.631, 1.0, 0.753, 1.0), ambient=0.2),
    "director-purple": Material(color=(0.804, 0.6, 0.820, 1.0), ambient=0.2),
    "director-blue": Material(color=(0.207, 0.596, 0.792, 1.0), ambient=0.2),
    "none": None,
}
color_list = list(materials.keys())

def random_material():
    return Material(color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1), ambient=0.2)


class ViewerData(edict):
    """
    Interface to standardize viewer data.
    """

    def __init__(self, Rt, K, cols, rows, imgnames=None):
        self.imgnames = imgnames
        self.Rt = Rt
        self.K = K
        self.num_frames = Rt.shape[0]
        self.cols = cols
        self.rows = rows
        self.validate_format()

    def validate_format(self):
        assert len(self.Rt.shape) == 3
        assert self.Rt.shape[0] == self.num_frames
        assert self.Rt.shape[1] == 3
        assert self.Rt.shape[2] == 4

        assert len(self.K.shape) == 2
        assert self.K.shape[0] == 3
        assert self.K.shape[1] == 3
        if self.imgnames is not None:
            assert self.num_frames == len(self.imgnames)
            assert self.num_frames > 0
            im_p = self.imgnames[0]
            assert op.exists(im_p), f"Image path {im_p} does not exist"


class ARCTICViewer:
    def __init__(
        self,
        render_types=["rgb", "depth", "mask"],
        interactive=True,
        size=(2024, 2024),
        fps=30,
    ):
        self.fps = fps
        if not interactive:
            # NOTE: aitviewer HeadlessRenderer in this environment does not pass
            # backend kwargs down to moderngl-window, so we override glcontext's
            # default backend to EGL before creating the renderer.
            headless_backend = os.environ.get("AITVIEWER_GL_BACKEND", "egl").lower()
            if headless_backend != "egl":
                logger.warning(f"Unsupported AITVIEWER_GL_BACKEND={headless_backend}, falling back to egl")
                headless_backend = "egl"
            # On systems that ship both libOpenGL.so.0 (vendor-neutral) and
            # libGL.so.1 (legacy), glcontext's EGL backend can end up with a
            # libGL handle whose GL functions are unresolved against the EGL-
            # bound context, producing a context that reports version 0 and
            # ValueError("Requested OpenGL version 450, got version 0").
            # Pinning GLCONTEXT_LINUX_LIBGL=libGL.so.1 forces the path that
            # exposes a working GL dispatch under EGL on NVIDIA drivers.
            os.environ.setdefault("GLCONTEXT_LINUX_LIBGL", "libGL.so.1")
            # Conda envs ship libglvnd's libEGL.so.1 but not libEGL_nvidia.so.0.
            # libEGL.so.1 reads /usr/share/glvnd/egl_vendor.d/10_nvidia.json
            # which references libEGL_nvidia.so.0 by name; if it lives under a
            # system path that isn't on LD_LIBRARY_PATH, the EGL context comes
            # back as version 0. Locate the NVIDIA backend on standard system
            # paths and prepend its directory so dlopen() can resolve it.
            _nvidia_egl_search = ["/usr/lib64", "/usr/lib/x86_64-linux-gnu"]
            for _p in _nvidia_egl_search:
                if op.exists(op.join(_p, "libEGL_nvidia.so.0")):
                    _ld = os.environ.get("LD_LIBRARY_PATH", "")
                    if _p not in _ld.split(os.pathsep):
                        os.environ["LD_LIBRARY_PATH"] = (
                            _p + (os.pathsep + _ld if _ld else "")
                        )
                    break
            try:
                import glcontext

                glcontext.default_backend = lambda: glcontext.get_backend_by_name(headless_backend)
                logger.info(f"Using headless OpenGL backend={headless_backend}")
            except Exception as exc:
                logger.warning(f"Failed to set glcontext backend={headless_backend}: {exc}")

            try:
                v = HeadlessRenderer(size=size)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize headless OpenGL context with EGL. "
                    "Please run inside an EGL-enabled environment/container."
                ) from exc
        else:
            v = Viewer(size=size)

        self.v = v
        self.interactive = interactive
        # self.layers = layers
        self.render_types = render_types

    def view_interactive(self):
        self.v.run()

    def view_fn_headless(self, num_iter, out_folder):
        v = self.v

        # _init_scene calls Scene.make_renderable on every node, which dumps a
        # lot of aitviewer print() chatter ("camera config does not exist",
        # lines.py shape spam, etc.). None of it is actionable for headless
        # batch renders, so redirect stdout/stderr at the fd level for the
        # duration of the call only.
        with open(os.devnull, "w") as _devnull:
            _o, _e = os.dup(1), os.dup(2)
            try:
                os.dup2(_devnull.fileno(), 1)
                os.dup2(_devnull.fileno(), 2)
                v._init_scene()
            finally:
                os.dup2(_o, 1); os.close(_o)
                os.dup2(_e, 2); os.close(_e)

 
        if "video" in self.render_types:
            vid_p = op.join(out_folder, "video.mp4")
            v.save_video(video_dir=vid_p, output_fps=int(v.playback_fps))
            return

        pbar = tqdm(range(num_iter))
        for fidx in pbar:
            out_rgb = op.join(out_folder, "images", f"rgb/{fidx:04d}.png")
            out_mask = op.join(out_folder, "images", f"mask/{fidx:04d}.png")
            out_depth = op.join(out_folder, "images", f"depth/{fidx:04d}.npy")

            # render RGB, depth, segmentation masks
            if "rgb" in self.render_types:
                v.export_frame(out_rgb)
            if "depth" in self.render_types:
                os.makedirs(op.dirname(out_depth), exist_ok=True)
                render_depth(v, out_depth)
            if "mask" in self.render_types:
                os.makedirs(op.dirname(out_mask), exist_ok=True)
                render_mask(v, out_mask)
            v.scene.next_frame()
        logger.info(f"Exported to {out_folder}")

    @abstractmethod
    def load_data(self):
        pass

    def check_format(self, batch):
        meshes_all, data = batch
        assert isinstance(meshes_all, dict)
        assert len(meshes_all) > 0
        for mesh in meshes_all.values():
            assert isinstance(mesh, Meshes)
        assert isinstance(data, ViewerData)

    def render_seq(self, batch, out_folder="./render_out", floor_y=0):
         
        meshes_all, data = batch
        self.setup_viewer(data, floor_y)
        for mesh in meshes_all.values():
            self.v.scene.add(mesh)
        if self.interactive:
            self.view_interactive()
        else:
            num_iter = data["num_frames"]      
            self.view_fn_headless(num_iter, out_folder)

    def setup_viewer(self, data, floor_y):
        v = self.v
        fps = self.fps

        if "imgnames" in data:
            setup_billboard(data, v)

        v.run_animations = True  # autoplay
        v.run_animations = False  # autoplay
        v.playback_fps = fps
        v.scene.fps = fps
        v.scene.origin.enabled = False
        v.scene.floor.enabled = False
        v.auto_set_floor = False
        self.v = v


def dist2vc(dist_ro, dist_lo, dist_o, _cmap, tf_fn=None):
    if tf_fn is not None:
        exp_map = tf_fn
    else:
        exp_map = small_exp_map
    dist_ro = exp_map(dist_ro)
    dist_lo = exp_map(dist_lo)
    dist_o = exp_map(dist_o)

    vc_ro = _cmap(dist_ro)
    vc_lo = _cmap(dist_lo)
    vc_o = _cmap(dist_o)
    return vc_ro, vc_lo, vc_o


def small_exp_map(_dist):
    dist = np.copy(_dist)
    # dist = 1.0 - np.clip(dist, 0, 0.1) / 0.1
    dist = np.exp(-20.0 * dist)
    return dist


def construct_viewer_meshes(data, draw_edges=False, flat_shading=True):
    meshes = {}
    for key, val in data.items():
        if 'single' in key:
            draw_edges = True
        else:
            draw_edges = False
        if "object" in key:
            flat_shading = False
        else:
            flat_shading = False
        v3d = val["v3d"]
        if not isinstance(val["color"], str):
            val["color"] = color_list[val["color"]]
        if val["color"] == "random":
            mesh_material = random_material()
        else:
            mesh_material = materials[val["color"]]
        meshes[key] = Meshes(
            v3d,
            val["f3d"],
            vertex_colors=val["vc"],
            face_colors=val["fc"] if "fc" in val else None,
            name=val["name"],
            flat_shading=flat_shading,
            draw_edges=draw_edges,
            material=mesh_material,
        )
    return meshes

  

def render_depth(v, depth_p):
    depth = np.array(v.get_depth()).astype(np.float16)
    np.save(depth_p, depth)


def render_mask(v, mask_p):
    nodes_uid = {node.name: node.uid for node in v.scene.collect_nodes()}
    my_cmap = {
        uid: [SEGM_IDS[name], SEGM_IDS[name], SEGM_IDS[name]]
        for name, uid in nodes_uid.items()
        if name in SEGM_IDS.keys()
    }
    mask = np.array(v.get_mask(color_map=my_cmap)).astype(np.uint8)
    mask = Image.fromarray(mask)
    mask.save(mask_p)


def setup_billboard(data, v):
    images_paths = data.imgnames
    K = data.K
    Rt = data.Rt
    rows = data.rows
    cols = data.cols
    camera = OpenCVCamera(K, Rt, cols, rows, viewer=v)
     
    if images_paths is not None:
        billboard = Billboard.from_camera_and_distance(
            camera, 10.0, cols, rows, images_paths)
        v.scene.add(billboard)
    v.scene.add(camera)
    v.scene.camera.load_cam()
    v.set_temp_camera(camera)
