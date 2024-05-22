
import math
import torch
import numpy as np
from PIL import Image
from .upscaler import perform_upscale

MANIFEST = {
    "name": "🍣 Toona Nodes",
}


def up255(x, t=0):
    y = np.zeros_like(x).astype(np.uint8)
    y[x > t] = 255
    return y


def get_shape_ceil(h, w):
    return math.ceil(((h * w) ** 0.5) / 64.0) * 64.0

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

def get_image_shape_ceil(im):
    H, W = im.shape[:2]
    return get_shape_ceil(H, W)

def resample_image(im, width, height):
    im = Image.fromarray(im)
    im = im.resize((int(width), int(height)), resample=LANCZOS)
    return np.array(im)

def set_image_shape_ceil(im, shape_ceil):
    shape_ceil = float(shape_ceil)

    H_origin, W_origin, _ = im.shape
    H, W = H_origin, W_origin
    
    for _ in range(256):
        current_shape_ceil = get_shape_ceil(H, W)
        if abs(current_shape_ceil - shape_ceil) < 0.1:
            break
        k = shape_ceil / current_shape_ceil
        H = int(round(float(H) * k / 64.0) * 64)
        W = int(round(float(W) * k / 64.0) * 64)

    if H == H_origin and W == W_origin:
        return im

    return resample_image(im, width=W, height=H)

class ToonaUpscale:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("Image", "Mask")

    FUNCTION = "run"

    CATEGORY = "Toona Nodes"

    def run(self, image, mask):
        print ("Toona Upscale:") 
        print (image.dtype)
        print (mask.dtype)
        interested_image = image[0, :, : ,:].detach().cpu().numpy()
        interested_mask = mask[0, :, :].detach().cpu().numpy()

        interested_image = (interested_image * 255).astype(np.uint8)
        interested_mask = (interested_mask * 255).astype(np.uint8)

        # super resolution 
        if get_image_shape_ceil(interested_image) < 1024:
            interested_image = perform_upscale(interested_image)
            print("post upscale shape", interested_image.shape)

        # resize to make images ready for diffusion
        interested_image = set_image_shape_ceil(interested_image, 1024)
        H, W, C = interested_image.shape

        # process mask
        interested_mask = up255(resample_image(interested_mask, W, H), t=127)

        print("Image: ", interested_image.shape)
        # print data type of interested_image
        print("Image Data Type: ", interested_image.dtype)
        print("Mask: ", interested_mask.shape)
        # print data type of interested_mask
        print("Mask Data Type: ", interested_mask.dtype)

        print("image min max" , interested_image.min(), interested_image.max())
        print("mask min max" , interested_mask.min(), interested_mask.max())
        interested_image = (interested_image / 255.0).astype(np.float32)

        # Convert the interested_mask_uint8 back to float32
        interested_mask = (interested_mask / 255.0).astype(np.float32)

        result_mask = torch.from_numpy(interested_mask).unsqueeze(0)
        result_image = torch.from_numpy(interested_image).unsqueeze(0)

        return result_image, result_mask
    
NODE_DISPLAY_NAME_MAPPINGS = {
    "ToonaUpscale": "Toona Upscale"
}