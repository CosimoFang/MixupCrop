from .simsiam import SimSiam
from .simsiam_mixup import SimSiam_MixUp
from .byol import BYOL
from .simclr import SimCLR
from .simclr_mixup import SimCLR_MixUp
from .moco import MoCo
from torchvision.models import resnet50, resnet18
import torch
from .backbones import resnet18_cifar_variant1, resnet18_cifar_variant2, resnet18_cifar100_variant1

def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}()")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(model_cfg):    

    if model_cfg.name == 'simsiam':
        model = SimSiam(get_backbone(model_cfg.backbone))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.name == 'simsiam_mixup':
        model = SimSiam_MixUp(get_backbone(model_cfg.backbone))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.name == 'byol':
        model = BYOL(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'simclr':
        model = SimCLR(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'simclr_mixup':
        model = SimCLR_MixUp(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'moco_mixup':
        model = MoCo(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'swav':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model






