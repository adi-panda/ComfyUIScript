from .toona_cut_mask import ToonaCutMask
from .toona_upscale import ToonaUpscale

NODE_CLASS_MAPPINGS = {
    "ToonaCutMask": ToonaCutMask,
    "ToonaUpscale": ToonaUpscale
}

__all__ = ['NODE_CLASS_MAPPINGS']