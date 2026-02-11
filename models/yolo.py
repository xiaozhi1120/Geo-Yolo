import contextlib
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
import math

# Import local modules
from .common import *
from .modules import *

# Attempt to import Ultralytics base classes if available in the environment
try:
    from ultralytics.nn.tasks import BaseModel, DetectionModel as BaseDetectionModel
    from ultralytics.utils import yaml_load, LOGGER, colorstr
    from ultralytics.nn.modules import Detect, Segment, Pose, OBB
except ImportError:
    # Simple mock classes to prevent errors and ensure the script can run independently
    class BaseModel(nn.Module):
        def forward(self, x): pass

    class BaseDetectionModel(BaseModel):
        pass

    def yaml_load(x):
        import yaml; return yaml.safe_load(open(x))

    LOGGER = print

    def colorstr(x):
        return x

    class Detect(nn.Module):
        pass


def make_divisible(x, divisor):
    """Ensure the number of channels is divisible by the specified divisor."""
    return math.ceil(x / divisor) * divisor


class Model(BaseDetectionModel):
    """
    Geo-YOLO Custom Model.
    Supports parsing GAMS-Block, GPDA, and DPS-Conv modules.
    """

    def __init__(self, cfg='configs/geo-yolo.yaml', ch=3, nc=None, verbose=True):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_load(cfg)

        # Override the number of classes if provided, otherwise use YAML settings
        if nc and nc != self.yaml['nc']:
            self.yaml['nc'] = nc

        # Parse the model structure
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch], verbose=verbose)

        # Build strides
        self.stride = torch.tensor([8., 16., 32.])

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Apply default PyTorch or Kaiming initialization
                pass


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """
    Parse the YOLO model dictionary and build the PyTorch model.
    Includes support for the GPDA module.
    """
    import ast

    # Parameter extraction
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))

    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
        depth, width, max_channels = scales[scale]

    if verbose:
        print(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

    layers, save, c2 = [], [], ch[-1]  # Layer list, save list, output channels

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        # Get module class
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]

        for j, a in enumerate(args):
            if isinstance(a, str):
                try:
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                except:
                    pass

        n = max(round(n * depth), 1) if n > 1 else n  # Apply depth gain

        # Update module recognition for Geo-YOLO components
        if m in {Conv, Bottleneck, SPPF, C2f, DPSConv, GAMSBlock, GPDA}:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in {Bottleneck, C2f, GAMSBlock, GPDA}:
                args.insert(2, n)  # Insert the number of repeats
                n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])

        # Instantiate module
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)

        t = str(m)[8:-2].replace("__main__.", "")  # Module type string
        np = sum(x.numel() for x in m_.parameters())  # Parameter count calculation
        m_.i, m_.f, m_.type = i, f, t

        if verbose:
            print(f"{i:>3}{str(f):>20}{n:>3}{np:10.0f}  {t:<45}{str(args):<30}")

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0: ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)
