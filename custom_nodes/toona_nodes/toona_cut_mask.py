
import math
import torch
import numpy as np

MANIFEST = {
    "name": "🍣 Toona Nodes",
}

def compute_initial_abcd(x):
    indices = np.where(x)
    a = np.min(indices[0])
    b = np.max(indices[0])
    c = np.min(indices[1])
    d = np.max(indices[1])
    abp = (b + a) // 2
    abm = (b - a) // 2
    cdp = (d + c) // 2
    cdm = (d - c) // 2
    l = int(max(abm, cdm) * 1.15)
    a = abp - l
    b = abp + l + 1
    c = cdp - l
    d = cdp + l + 1
    a, b, c, d = regulate_abcd(x, a, b, c, d)
    return a, b, c, d



def regulate_abcd(x, a, b, c, d):
    H, W = x.shape[:2]
    if a < 0:
        a = 0
    if a > H:
        a = H
    if b < 0:
        b = 0
    if b > H:
        b = H
    if c < 0:
        c = 0
    if c > W:
        c = W
    if d < 0:
        d = 0
    if d > W:
        d = W
    return int(a), int(b), int(c), int(d)

def solve_abcd(x, a, b, c, d, k):
    k = float(k)
    assert 0.0 <= k <= 1.0

    H, W = x.shape[:2]
    if k == 1.0:
        return 0, H, 0, W
    while True:
        if b - a >= H * k and d - c >= W * k:
            break

        add_h = (b - a) < (d - c)
        add_w = not add_h

        if b - a == H:
            add_w = True

        if d - c == W:
            add_h = True

        if add_h:
            a -= 1
            b += 1

        if add_w:
            c -= 1
            d += 1

        a, b, c, d = regulate_abcd(x, a, b, c, d)
    return a, b, c, d

class ToonaCutMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK", ),
                "inpaint_respective_field": ("FLOAT",{
                    "default": 0.618,
                    "step":0.001,
                    "min": 0.0,
                    "max": 1.0,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("Image", "Mask")

    FUNCTION = "run"

    CATEGORY = "Toona Nodes"

    def run(self, image, mask, inpaint_respective_field):
        print ("Toona Cut Mask:") 
        print ("Inpaint Respective Field: ", inpaint_respective_field)
        image_np = image[0, :, : ,:].detach().cpu().numpy()
        mask_np = mask[0, :, :].detach().cpu().numpy()
        print("Image: ", image_np.shape)
        print("Mask: ", mask_np.shape)


        a, b, c, d = compute_initial_abcd(mask_np > 0)
        a, b, c, d = solve_abcd(mask_np, a, b, c, d, k=inpaint_respective_field)

        # interested area
        interested_area = (a, b, c, d)
        interested_mask = mask_np[a:b, c:d]
        interested_image = image_np[a:b, c:d]

        result_mask = torch.from_numpy(interested_mask).unsqueeze(0)
        result_image = torch.from_numpy(interested_image).unsqueeze(0)

        return result_image, result_mask
    

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToonaCutMask": "Toona Cut Mask"
}