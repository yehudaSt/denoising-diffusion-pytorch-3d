import cv2
import torch
from PIL import Image
import numpy as np
from torchvision.transforms.functional import pil_to_tensor

max_possible_depth_mm = 1200
max_pixel_value = 1
min_pixel_value = 800
depth_range = max_possible_depth_mm - min_pixel_value

crop_corner_u1 = 290
crop_corner_v1 = 250
crop_corner_u2 = 1560
crop_corner_v2 = 930

depth_min = 0
depth_max = 2000

def rescale_depth_to_pixel(depth: np.ndarray):
    """Rescale depth image to be between 0 and 255"""

    # set depth less than zero to zero:
    depth[depth < min_pixel_value] = min_pixel_value
    depth[depth > max_possible_depth_mm] = max_possible_depth_mm
    depth = (depth - min_pixel_value) / depth_range * max_pixel_value
    return depth.astype('uint8')

# def rescale_pixel_to_depth(pixel: np.ndarray):
#     """Rescale pixel image to be between 0 and 2000 mm"""
#
#     return pixel / max_pixel_value * depth_range + min_pixel_value

def rescale_pixel_to_depth(pixel: np.ndarray):
    """Rescale pixel image to be between 0 and 2000 mm"""
    d_range = depth_max - depth_min
    return pixel * d_range + depth_min

def get_item(path: str):
    u1 = 290
    v1 = 250
    u2 = 1560
    v2 = 930
    # img = Image.open(path).convert("RGB")
    # normalize to [0, 1] range
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGBA) / 255.0
    depth = np.load(path.replace('color', 'depth').replace('png', 'npy'))
    depth = rescale_depth_to_pixel(depth)
    img[:, :, 3] = depth
    img = img[v1:v2, u1:u2]
    img = cv2.resize(img,(128, 128))
    img_rehsaped = np.reshape(img, (4, 128, 128))
    img = torch.from_numpy(img_rehsaped.astype(np.float16))
    return img