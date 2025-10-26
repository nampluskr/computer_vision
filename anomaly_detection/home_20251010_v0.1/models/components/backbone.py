"""
Backbone Weight Path Management

Global Variable:
- BACKBONE_DIR: Directory containing pretrained backbone weights

Dictionary:
- BACKBONE_WEIGHT_FILES: Mapping of backbone names to weight filenames
  * CNN backbones: resnet18, resnet50, wide_resnet50_2, efficientnet_b5
  * Transformer backbones: dinov2_vit_*, cait_*, deit_*
  
Functions:
- set_backbone_dir(backbone_dir): Update global BACKBONE_DIR
- get_backbone_path(backbone): Get full path to backbone weight file
  * Automatically handles different backbone types (CNN vs Transformer)
  * Checks file existence and prints status
  * Returns path for safetensors or .pth files depending on backbone type

Example:
    >>> from models.components.backbone import set_backbone_dir, get_backbone_path
    >>> set_backbone_dir("/mnt/d/backbones")
    >>> path = get_backbone_path("resnet50")
    >>> # Returns: "/mnt/d/backbones/resnet50-0676ba61.pth"
    >>>
    >>> path = get_backbone_path("dinov2_vit_base_14")
    >>> # Returns: "/mnt/d/backbones/dinov2_vitb14_pretrain.pth"
"""

import os

BACKBONE_WEIGHT_FILES = {
    "resnet18": "resnet18-f37072fd.pth",
    "resnet34": "resnet34-b627a593.pth",
    "resnet50": "resnet50-0676ba61.pth",
    "wide_resnet50_2": "wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "wide_resnet50_2-32ee1156.pth",
    "efficientnet_b5": "efficientnet_b5_lukemelas-1a07897c.pth",
    
    # https://huggingface.co/timm/wide_resnet50_2.tv_in1k/tree/main
    "wide_resnet50_2.tv_in1k": "wide_resnet50_2.tv_in1k",

    # https://huggingface.co/zgcr654321/pretrained_models/tree/main/dinov2_pretrain_official_pytorch_weights
    "dinov2_vit_small_14": "dinov2_vits14_pretrain.pth",
    "dinov2_vit_base_14": "dinov2_vitb14_pretrain.pth",
    "dinov2_vit_large_14": "dinov2_vitl14_pretrain.pth",
    "dinov2reg_vit_small_14": "dinov2_vits14_reg4_pretrain.pth",
    "dinov2reg_vit_base_14": "dinov2_vitb14_reg4_pretrain.pth",
    "dinov2reg_vit_large_14": "dinov2_vitl14_reg4_pretrain.pth",

    # https://huggingface.co/timm/cait_m48_448.fb_dist_in1k/tree/main
    # https://huggingface.co/timm/cait_s24_224.fb_dist_in1k/tree/main
    # https://huggingface.co/timm/deit_base_distilled_patch16_224.fb_in1k/tree/main
    # https://huggingface.co/timm/deit_base_distilled_patch16_384.fb_in1k/tree/main
    "cait_s24_224": "cait_s24_224.fb_dist_in1k",
    "cait_m48_448": "cait_m48_448.fb_dist_in1k",
    "deit_base_distilled_patch16_224": "deit_base_distilled_patch16_224.fb_in1k",
    "deit_base_distilled_patch16_384": "deit_base_distilled_patch16_384.fb_in1k",
}

BACKBONE_DIR = ""   #"/mnt/d/backbones"

def get_backbone_dir():
    return BACKBONE_DIR

def set_backbone_dir(backbone_dir):
    global BACKBONE_DIR
    BACKBONE_DIR = backbone_dir
    print(f" > Backbone directory set to: {BACKBONE_DIR}")


def get_backbone_path(backbone: str):
    if backbone.startswith("cait"):
        dirname = BACKBONE_WEIGHT_FILES.get(backbone, f"{backbone}.fb_dist_in1k")
        weight_path = os.path.join(BACKBONE_DIR, dirname, "model.safetensors")
    elif backbone.startswith("deit"):
        dirname = BACKBONE_WEIGHT_FILES.get(backbone, f"{backbone}.fb_in1k")
        weight_path = os.path.join(BACKBONE_DIR, dirname, "model.safetensors")
    elif backbone == "wide_resnet50_2.tv_in1k":
        dirname = BACKBONE_WEIGHT_FILES.get(backbone, f"{backbone}.tv_in1k")
        weight_path = os.path.join(BACKBONE_DIR, dirname, "model.safetensors")
    else:
        filename = BACKBONE_WEIGHT_FILES.get(backbone, f"{backbone}.pth")
        weight_path = os.path.join(BACKBONE_DIR, filename)

    if os.path.isfile(weight_path):
        print(f" > {backbone} weight is loaded from {weight_path}.")
    else:
        print(f" > {backbone} weight not found in {weight_path}. ")
    return weight_path