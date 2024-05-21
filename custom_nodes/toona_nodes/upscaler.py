import os
import torch
import numpy as np

from toona_nodes.ldm_patched.pfn.architecture.RRDB import RRDBNet as ESRGAN
from toona_nodes.ldm_patched.contrib.external_upscale_model import ImageUpscaleWithModel
from collections import OrderedDict

model_filename = 'fooocus_upscaler_s409985e5.bin'
opImageUpscaleWithModel = ImageUpscaleWithModel()
model = None



@torch.no_grad()
@torch.inference_mode()
def numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y

@torch.no_grad()
@torch.inference_mode()
def pytorch_to_numpy(x):
    return [np.clip(255. * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]

def perform_upscale(img):
    global model

    print(f'Upscaling image with shape {str(img.shape)} ...')

    if model is None:
        sd = torch.load(model_filename)
        sdo = OrderedDict()
        for k, v in sd.items():
            sdo[k.replace('residual_block_', 'RDB')] = v
        del sd
        model = ESRGAN(sdo)
        model.cpu()
        model.eval()

    img = numpy_to_pytorch(img)
    img = opImageUpscaleWithModel.upscale(model, img)[0]
    img = pytorch_to_numpy(img)[0]

    return img
