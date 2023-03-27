import numpy as np

from cv2 import circle
from zod.constants import Camera
from zod.utils.geometry import get_points_in_camera_fov, project_3d_to_2d_kannala, transform_points

def car_points_to_camera(zod_frames, frame_id, car_points):
    camera=Camera.FRONT
    frame = zod_frames[frame_id]
    calibs = frame.calibration
    # transform point to camera coordinate system
    T_inv = np.linalg.pinv(calibs.get_extrinsics(camera).transform)
    camera_points = transform_points(car_points[:, :3], T_inv)
    return camera_points

def camera_points_to_2d(zod_frames, frame_id, camera_points):
    camera=Camera.FRONT
    frame = zod_frames[frame_id]
    calibs = frame.calibration
    # filter points that are not in the camera field of view
    points_in_fov = get_points_in_camera_fov(calibs.cameras[camera].field_of_view, camera_points)
    
    # project points to image plane
    image_points = project_3d_to_2d_kannala(
        points_in_fov,
        calibs.cameras[camera].intrinsics[..., :3],
        calibs.cameras[camera].distortion,
    )
    return image_points

def get_image(zod_frames, frame_id):
    return zod_frames[frame_id].get_image()

def draw_circles(image, image_points, colour=(255, 0, 0)):
    for i in range(image_points.shape[0]):
        x, y = int(image_points[i, 0]), int(image_points[i, 1])
        circle(image, (x,y), 20, colour, -1)
    return image

def visualize_car_points(zod_frames, frame_id, car_points, colour=(255, 0, 0), image=None):
    camera_points = car_points_to_camera(zod_frames, frame_id, car_points)
    image_points = camera_points_to_2d(zod_frames, frame_id, camera_points)
    if image is None:
        image = get_image(zod_frames, frame_id)
    image = draw_circles(image, image_points, colour)
    return image
