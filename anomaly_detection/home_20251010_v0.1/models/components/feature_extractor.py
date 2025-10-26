import logging
from collections.abc import Sequence
import os

import timm
import torch
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor
from torch.fx.graph_module import GraphModule

from .backbone import get_backbone_path


logger = logging.getLogger(__name__)


#####################################################################
# anomalib/src/anomalib/models/components/feature_extractors/utils.py
#####################################################################

def dryrun_find_featuremap_dims(
    feature_extractor: GraphModule,
    input_size: tuple[int, int],
    layers: list[str],
) -> dict[str, dict[str, int | tuple[int, int]]]:
    device = next(feature_extractor.parameters()).device
    dryrun_input = torch.empty(1, 3, *input_size).to(device)
    was_training = feature_extractor.training
    feature_extractor.eval()
    with torch.no_grad():
        dryrun_features = feature_extractor(dryrun_input)
    if was_training:
        feature_extractor.train()
    return {
        layer: {
            "num_features": dryrun_features[layer].shape[1],
            "resolution": dryrun_features[layer].shape[2:],
        }
        for layer in layers
    }


#####################################################################
# anomalib/src/anomalib/models/components/feature_extractors/timm.py
#####################################################################

class TimmFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: str | nn.Module,
        layers: Sequence[str],
        pre_trained: bool = True,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.layers = list(layers)
        self.requires_grad = requires_grad

        if isinstance(backbone, nn.Module):
            self.feature_extractor = create_feature_extractor(
                backbone,
                return_nodes={layer: layer for layer in self.layers},
            )
            layer_metadata = dryrun_find_featuremap_dims(self.feature_extractor, (256, 256), layers=self.layers)
            self.out_dims = [feature_info["num_features"] for layer_name, feature_info in layer_metadata.items()]

        elif isinstance(backbone, str):
            self.idx = self._map_layer_to_idx()
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=False,
                pretrained_cfg=None,
                features_only=True,
                exportable=True,
                out_indices=self.idx,
            )
            self.out_dims = self.feature_extractor.feature_info.channels()
            if pre_trained:
                weight_path = get_backbone_path(backbone)
                state_dict = torch.load(weight_path, map_location="cpu")
                self.feature_extractor.load_state_dict(state_dict, strict=False)
        else:
            msg = f"Backbone of type {type(backbone)} must be of type str or nn.Module."
            raise TypeError(msg)

        self._features = {layer: torch.empty(0) for layer in self.layers}

    def _map_layer_to_idx(self) -> list[int]:
        idx = []
        model = timm.create_model(
            self.backbone,
            pretrained=False,
            features_only=True,
            exportable=True,
        )
        # model.feature_info.info returns list of dicts containing info,
        # inside which "module" contains layer name
        layer_names = [info["module"] for info in model.feature_info.info]
        for layer in self.layers:
            try:
                idx.append(layer_names.index(layer))
            except ValueError:  # noqa: PERF203
                msg = f"Layer {layer} not found in model {self.backbone}. Available layers: {layer_names}"
                logger.warning(msg)
                # Remove unfound key from layer dict
                self.layers.remove(layer)

        return idx

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.requires_grad:
            features = self.feature_extractor(inputs)
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                features = self.feature_extractor(inputs)
        if not isinstance(features, dict):
            features = dict(zip(self.layers, features, strict=True))
        return features