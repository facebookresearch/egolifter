import time
import threading
import traceback
import numpy as np
import torch
import viser
import viser.transforms as vtf


# from internal.cameras.cameras import Cameras
from scene.cameras import Camera
from utils.graphics_utils import fov2focal

def get_sizes(
    max_res: int, 
    canvas_aspect_ratio: float,
    lock_aspect_one: bool,
):
    if lock_aspect_one:
        render_aspect_ratio = 1
        if canvas_aspect_ratio <= 1: # width <= height
            canvas_height = max_res
            canvas_width = int(canvas_height * canvas_aspect_ratio)
        else: # width > height
            canvas_width = max_res
            canvas_height = int(canvas_width / canvas_aspect_ratio)
        # Image size is always smaller than canvas size
        image_width = min(canvas_width, canvas_height)
        image_height = min(canvas_width, canvas_height)
    else:
        render_aspect_ratio = canvas_aspect_ratio
        if render_aspect_ratio <= 1: # width <= height
            image_height = max_res
            image_width = int(image_height * render_aspect_ratio)
        else: # width > height
            image_width = max_res
            image_height = int(image_width / render_aspect_ratio)
        canvas_width = image_width
        canvas_height = image_height
    
    return canvas_width, canvas_height, image_width, image_height

class ClientThread(threading.Thread):
    def __init__(self, viewer, model, client: viser.ClientHandle):
        super().__init__()
        self.viewer = viewer
        self.model = model
        self.client = client

        self.render_trigger = threading.Event()

        self.last_move_time = 0

        self.last_camera = None  # store camera information

        self.state = "low"  # low or high render resolution

        self.stop_client = False  # whether stop this thread

        client.camera.up_direction = viewer.up_direction
        
        # Set the camera position to be close to look_at
        # Such that the camera motion is more natural
        look_at = client.camera.look_at
        position = client.camera.position
        direction = position - look_at
        direction = direction / np.linalg.norm(direction) * 0.11 # 10cm is the closest distance possible. 
        
        with client.atomic():
            client.camera.position = look_at + direction
            client.camera.look_at = look_at
            
        @client.camera.on_update
        def _(cam: viser.CameraHandle) -> None:
            with self.client.atomic():
                self.last_camera = cam
                self.state = "low"  # switch to low resolution mode when a new camera received
                self.render_trigger.set()

    def render_and_send(self):
        with self.client.atomic():
            cam = self.last_camera
            
            self.last_move_time = time.time()

            # get camera pose
            R = vtf.SO3(wxyz=self.client.camera.wxyz)
            R = R @ vtf.SO3.from_x_radians(np.pi)
            R = torch.tensor(R.as_matrix())
            pos = torch.tensor(self.client.camera.position, dtype=torch.float64)
            c2w = torch.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = pos

            c2w = torch.matmul(self.viewer.camera_transform, c2w)

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = torch.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, 3]

            # calculate resolution
            max_res, jpeg_quality = self.get_render_options()
            canvas_aspect_ratio = cam.aspect
            
            canvas_width, canvas_height, image_width, image_height = get_sizes(
                max_res, 
                canvas_aspect_ratio,
                self.viewer.lock_aspect_one,
            )
            
            # construct camera
            appearance_id = self.viewer.get_appearance_id_value()

            # The FoV of the point cloud is constant w.r.t. the image height. 
            # Thus change the fx computation based on the image height, to ensure alignment. 
            fx = torch.tensor([fov2focal(cam.fov, canvas_height)], dtype=torch.float)
            
            # This is needed when we use the Camera transformation from the original 3DGS codebase
            R = R.T
            
            camera = Camera(
                colmap_id=0,
                uid=0,
                R=R.cpu().numpy(),
                T=T.cpu().numpy(),
                FoVx=cam.fov,
                FoVy=cam.fov,
                # cx=torch.tensor([(image_width // 2)], dtype=torch.int),
                # cy=torch.tensor([(image_height // 2)], dtype=torch.int),
                image_width=image_width,
                image_height=image_height,
                image_name="",
                image_path="",
                fid=self.viewer.time_slider.value,
                # appearance_id=torch.tensor([appearance_id[0]], dtype=torch.int),
                # normalized_appearance_id=torch.tensor([appearance_id[1]], dtype=torch.float),
                # time=torch.tensor([self.viewer.time_slider.value], dtype=torch.float),
                # distortion_params=None,
                # camera_type=torch.tensor([0], dtype=torch.int),
            )

            with torch.no_grad():
                image = self.model(camera, scaling_modifier=self.viewer.scaling_modifier.value)['render']
                image = torch.clamp(image, max=1.)
                image = torch.permute(image, (1, 2, 0))

                if not self.viewer.lock_aspect_one:
                    canvas = image
                else:
                    canvas = torch.zeros((canvas_height, canvas_width, 3), dtype=image.dtype, device=image.device)
                    # Place the image at the center of the canvas
                    left = (canvas_width - image_width) // 2
                    right = left + image_width
                    top = (canvas_height - image_height) // 2
                    bottom = top + image_height
                    canvas[top:bottom, left:right, :] = image
                
                self.client.set_background_image(
                    canvas.cpu().numpy(),
                    format=self.viewer.image_format,
                    jpeg_quality=jpeg_quality,
                )

    def run(self):
        while True:
            trigger_wait_return = self.render_trigger.wait(0.2)  # TODO: avoid wasting CPU
            # stop client thread?
            if self.stop_client is True:
                break
            if not trigger_wait_return:
                # skip if camera is none
                if self.last_camera is None:
                    continue

                # if we haven't received a trigger in a while, switch to high resolution
                if self.state == "low":
                    self.state = "high"  # switch to high resolution mode
                else:
                    continue  # skip if already in high resolution mode

            self.render_trigger.clear()

            try:
                self.render_and_send()
            except Exception as err:
                print("error occurred when rendering for client")
                traceback.print_exc()
                break

        self._destroy()

    def get_render_options(self):
        if self.state == "low":
            return self.viewer.max_res_when_moving.value, int(self.viewer.jpeg_quality_when_moving.value)
        return self.viewer.max_res_when_static.value, int(self.viewer.jpeg_quality_when_static.value)

    def stop(self):
        self.stop_client = True
        # self.render_trigger.set()  # TODO: potential thread leakage?

    def _destroy(self):
        print("client thread #{} destroyed".format(self.client.client_id))
        self.viewer = None
        self.model = None
        self.client = None
        self.last_camera = None
