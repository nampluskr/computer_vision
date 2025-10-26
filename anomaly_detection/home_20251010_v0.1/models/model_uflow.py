from collections.abc import Iterable
import math
import scipy.stats as st
from omegaconf import ListConfig

import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torch.distributions import Normal


from FrEIA import framework as ff
from FrEIA import modules as fm

from .components.feature_extractor import TimmFeatureExtractor
from .components.all_in_one_block import AllInOneBlock


#####################################################################
# anomalib/src/anomalib/models/image/uflow/anomaly_map.py
#####################################################################

class AnomalyMapGenerator(nn.Module):
    def __init__(self, input_size: ListConfig | tuple) -> None:
        super().__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else tuple(input_size)

    def forward(self, latent_variables: list[Tensor]) -> Tensor:
        return self.compute_anomaly_map(latent_variables)

    def compute_anomaly_map(self, latent_variables: list[Tensor]) -> Tensor:
        likelihoods = []
        for z in latent_variables:
            # Mean prob by scale. Likelihood is actually with sum instead of mean. Using mean to avoid numerical issues.
            # Also, this way all scales have the same weight, and it does not depend on the number of channels
            log_prob_i = -torch.mean(z**2, dim=1, keepdim=True) * 0.5
            prob_i = torch.exp(log_prob_i)
            likelihoods.append(
                F.interpolate(
                    prob_i,
                    size=self.input_size,
                    mode="bilinear",
                    align_corners=False,
                ),
            )
        return 1 - torch.mean(torch.stack(likelihoods, dim=-1), dim=-1)

    def compute_anomaly_mask(
        self,
        z: list[torch.Tensor],
        window_size: int = 7,
        binomial_probability_thr: float = 0.5,
        high_precision: bool = False,
    ) -> torch.Tensor:

        log_prob_l = [
            self.binomial_test(zi, window_size / (2**scale), binomial_probability_thr, high_precision)
            for scale, zi in enumerate(z)
        ]

        log_prob_l_up = torch.cat(
            [F.interpolate(lpl, size=self.input_size, mode="bicubic", align_corners=True) for lpl in log_prob_l],
            dim=1,
        )

        log_prob = torch.sum(log_prob_l_up, dim=1, keepdim=True)

        log_number_of_tests = torch.log10(torch.sum(torch.tensor([zi.shape[-2] * zi.shape[-1] for zi in z])))
        log_nfa = log_number_of_tests + log_prob

        anomaly_score = -log_nfa

        return anomaly_score < 0

    @staticmethod
    def binomial_test(
        z: torch.Tensor,
        window_size: int,
        probability_thr: float,
        high_precision: bool = False,
    ) -> torch.Tensor:

        # Calculate tau using pure PyTorch
        normal_dist = Normal(0, 1)
        p_adjusted = (probability_thr + 1) / 2
        tau = normal_dist.icdf(torch.tensor(p_adjusted)) ** 2
        half_win = max(int(window_size // 2), 1)

        n_chann = z.shape[1]

        # Use float64 for high precision mode
        dtype = torch.float64 if high_precision else torch.float32
        z = z.to(dtype)
        tau = tau.to(dtype)

        # Candidates
        z2 = F.pad(z**2, tuple(4 * [half_win]), "reflect")
        z2_unfold_h = z2.unfold(-2, 2 * half_win + 1, 1)
        z2_unfold_hw = z2_unfold_h.unfold(-2, 2 * half_win + 1, 1)
        observed_candidates_k = torch.sum(z2_unfold_hw >= tau, dim=(-2, -1))

        # All volume together
        observed_candidates = torch.sum(observed_candidates_k, dim=1, keepdim=True)
        x = observed_candidates / n_chann
        n = int((2 * half_win + 1) ** 2)

        # Use scipy for the binomial test as PyTorch does not have a stable/direct equivalent.
        # nosemgrep: trailofbits.python.numpy-in-pytorch-modules.numpy-in-pytorch-modules
        x_np = x.detach().cpu().numpy()
        log_prob_np = st.binom.logsf(x_np, n, 1 - probability_thr) / math.log(10)

        return torch.from_numpy(log_prob_np).to(z.device)

#####################################################################
# anomalib/src/anomalib/models/image/uflow/feature_extraction.py
#####################################################################

AVAILABLE_EXTRACTORS = ["mcait", "resnet18", "wide_resnet50_2"]

def get_feature_extractor(backbone: str, input_size: tuple[int, int] = (256, 256)) -> nn.Module:
    if backbone not in AVAILABLE_EXTRACTORS:
        msg = f"Feature extractor must be one of {AVAILABLE_EXTRACTORS}."
        raise ValueError(msg)

    feature_extractor: nn.Module
    if backbone in {"resnet18", "wide_resnet50_2"}:
        feature_extractor = LayerNormFeatureExtractor(
            backbone,
            input_size,
            layers=("layer1", "layer2", "layer3"),
        ).eval()
    if backbone == "mcait":
        feature_extractor = CaitFeatureExtractor().eval()

    return feature_extractor


class LayerNormFeatureExtractor(TimmFeatureExtractor):
    def __init__(
        self,
        backbone: str,
        input_size: tuple[int, int],
        layers: tuple[str, ...] = ("layer1", "layer2", "layer3"),
        **kwargs,  # noqa: ARG002 | unused argument
    ) -> None:
        super().__init__(backbone, layers, pre_trained=True, requires_grad=False)
        self.channels = self.feature_extractor.feature_info.channels()
        self.scale_factors = self.feature_extractor.feature_info.reduction()
        self.scales = range(len(self.scale_factors))

        self.feature_normalizations = nn.ModuleList()
        for in_channels, scale in zip(self.channels, self.scale_factors, strict=True):
            self.feature_normalizations.append(
                nn.LayerNorm(
                    [in_channels, int(input_size[0] / scale), int(input_size[1] / scale)],
                    elementwise_affine=True,
                ),
            )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(img)
        return self.normalize_features(features)

    def extract_features(self, img: torch.Tensor) -> torch.Tensor:
        self.feature_extractor.eval()
        return self.feature_extractor(img)

    def normalize_features(self, features: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        return [self.feature_normalizations[i](feature) for i, feature in enumerate(features)]


class CaitFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_size = 448
        self.extractor1 = timm.create_model("cait_m48_448", pretrained=False)
        self.extractor2 = timm.create_model("cait_s24_224", pretrained=False)
        self._load_pretrained_weights()
        
        self.channels = [768, 384]
        self.scale_factors = [16, 32]
        self.scales = range(len(self.scale_factors))

        for param in self.extractor1.parameters():
            param.requires_grad = False
        for param in self.extractor2.parameters():
            param.requires_grad = False

    def _load_pretrained_weights(self):
        import os
        from safetensors.torch import load_file
        
        weight_path1 = get_transformer_weight_path("cait_m48_448", "cait_m48_448.fb_dist_in1k")
        if weight_path1 and os.path.isfile(weight_path1):
            state_dict = load_file(weight_path1)
            self.extractor1.load_state_dict(state_dict)
            print(f" > [Info] Loaded cait_m48_448 from {weight_path1}")
        else:
            print(f" > [Warning] cait_m48_448 weights not found")
        
        weight_path2 = get_transformer_weight_path("cait_s24_224", "cait_s24_224.fb_dist_in1k")
        if weight_path2 and os.path.isfile(weight_path2):
            state_dict = load_file(weight_path2)
            self.extractor2.load_state_dict(state_dict)
            print(f" > [Info] Loaded cait_s24_224 from {weight_path2}")
        else:
            print(f" > [Warning] cait_s24_224 weights not found")

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(img)
        return self.normalize_features(features)

    def extract_features(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.extractor1.eval()
        self.extractor2.eval()

        # Scale 1 --> Extractor 1
        x1 = self.extractor1.patch_embed(img)
        x1 = x1 + self.extractor1.pos_embed
        x1 = self.extractor1.pos_drop(x1)
        for i in range(41):  # paper Table 6. Block Index = 40
            x1 = self.extractor1.blocks[i](x1)

        # Scale 2 --> Extractor 2
        img_sub = F.interpolate(img, size=(224, 224), mode="bicubic", align_corners=True)
        x2 = self.extractor2.patch_embed(img_sub)
        x2 = x2 + self.extractor2.pos_embed
        x2 = self.extractor2.pos_drop(x2)
        for i in range(21):
            x2 = self.extractor2.blocks[i](x2)

        return (x1, x2)

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        normalized_features = []
        for i, extractor in enumerate([self.extractor1, self.extractor2]):
            batch, _, channels = features[i].shape
            scale_factor = self.scale_factors[i]

            x = extractor.norm(features[i].contiguous())
            x = x.permute(0, 2, 1)
            x = x.reshape(batch, channels, self.input_size // scale_factor, self.input_size // scale_factor)
            normalized_features.append(x)

        return normalized_features

#####################################################################
# anomalib/src/anomalib/models/image/uflow/loss.py
#####################################################################

class UFlowLoss(nn.Module):
    @staticmethod
    def forward(hidden_variables: list[Tensor], jacobians: list[Tensor]) -> Tensor:
        lpz = torch.sum(torch.stack([0.5 * torch.sum(z_i**2, dim=(1, 2, 3)) for z_i in hidden_variables], dim=0))
        return torch.mean(lpz - jacobians)


#####################################################################
# anomalib/src/anomalib/models/image/uflow/torch_model.py
#####################################################################

class AffineCouplingSubnet:
    def __init__(self, kernel_size: int, subnet_channels_ratio: float) -> None:
        self.kernel_size = kernel_size
        self.subnet_channels_ratio = subnet_channels_ratio

    def __call__(self, in_channels: int, out_channels: int) -> nn.Sequential:
        mid_channels = int(in_channels * self.subnet_channels_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, self.kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, self.kernel_size, padding="same"),
        )


class UflowModel(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int] = (448, 448),
        flow_steps: int = 4,
        backbone: str = "mcait",
        affine_clamp: float = 2.0,
        affine_subnet_channels_ratio: float = 1.0,
        permute_soft: bool = False,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.affine_clamp = affine_clamp
        self.affine_subnet_channels_ratio = affine_subnet_channels_ratio
        self.permute_soft = permute_soft

        self.feature_extractor = get_feature_extractor(backbone, input_size)
        self.flow = self.build_flow(flow_steps)
        self.anomaly_map_generator = AnomalyMapGenerator(input_size)

    def build_flow(self, flow_steps: int) -> ff.GraphINN:
        input_nodes = []
        for channel, s_factor in zip(
            self.feature_extractor.channels,
            self.feature_extractor.scale_factors,
            strict=True,
        ):
            input_nodes.append(
                ff.InputNode(
                    channel,
                    self.input_size[0] // s_factor,
                    self.input_size[1] // s_factor,
                    name=f"cond_{channel}",
                ),
            )

        nodes, output_nodes = [], []
        last_node = input_nodes[-1]
        for i in reversed(range(1, len(input_nodes))):
            flows = self.build_flow_stage(last_node, flow_steps)
            volume_size = flows[-1].output_dims[0][0]
            split = ff.Node(
                flows[-1],
                fm.Split,
                {"section_sizes": (volume_size // 8 * 4, volume_size - volume_size // 8 * 4), "dim": 0},
                name=f"split_{i + 1}",
            )
            output = ff.OutputNode(split.out1, name=f"output_scale_{i + 1}")
            up = ff.Node(split.out0, fm.IRevNetUpsampling, {}, name=f"up_{i + 1}")
            last_node = ff.Node([input_nodes[i - 1].out0, up.out0], fm.Concat, {"dim": 0}, name=f"cat_{i}")

            output_nodes.append(output)
            nodes.extend([*flows, split, up, last_node])

        flows = self.build_flow_stage(last_node, flow_steps)
        output = ff.OutputNode(flows[-1], name="output_scale_1")

        output_nodes.append(output)
        nodes.extend(flows)

        return ff.GraphINN(input_nodes + nodes + output_nodes[::-1])

    def build_flow_stage(self, in_node: ff.Node, flow_steps: int, condition_node: ff.Node = None) -> list[ff.Node]:
        flow_size = in_node.output_dims[0][-1]
        nodes = []
        for step in range(flow_steps):
            nodes.append(
                ff.Node(
                    in_node,
                    AllInOneBlock,
                    module_args={
                        "subnet_constructor": AffineCouplingSubnet(
                            3 if step % 2 == 0 else 1,
                            self.affine_subnet_channels_ratio,
                        ),
                        "affine_clamping": self.affine_clamp,
                        "permute_soft": self.permute_soft,
                    },
                    conditions=condition_node,
                    name=f"flow{flow_size}_step{step}",
                ),
            )
            in_node = nodes[-1]
        return nodes

    def forward(self, image: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        features = self.feature_extractor(image)
        z, ljd = self.encode(features)

        if self.training:
            return z, ljd

        anomaly_map = self.anomaly_map_generator(z)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, ljd = self.flow(features, rev=False)
        if len(self.feature_extractor.scales) == 1:
            z = [z]
        return z, ljd
    

#####################################################################
# Trainer for FastFlow Model
#####################################################################
from .components.trainer import BaseTrainer, EarlyStopper

class UflowTrainer(BaseTrainer):
    def __init__(self, model=None, optimizer=None, loss_fn=None, metrics=None, device=None,
                 scheduler=None, early_stopper_loss=None, early_stopper_auroc=None,
                 backbone="wide_resnet50_2", input_size=(256, 256)):

        if model is None:
            model = UflowModel(backbone=backbone, input_size=input_size)
        if optimizer is None:
            optimizer = torch.optim.Adam([{"params": model.parameters(), "initial_lr": 1e-3}], 
                lr=1e-3, weight_decay=1e-5)
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.4, total_iters=25000)
        # if early_stopper_loss is None:
        #     early_stopper_loss = EarlyStopper(patience=10, min_delta=0.01, mode='min', target_value=0.1)
        # if early_stopper_auroc is None:
        #     early_stopper_auroc = EarlyStopper(patience=10, min_delta=0.001, mode='max', target_value=0.995)
        if loss_fn is None:
            loss_fn = UFlowLoss()

        super().__init__(model, optimizer, loss_fn, metrics, device,
                         scheduler, early_stopper_loss, early_stopper_auroc)
        self.eval_period = 5

        self.model.feature_extractor.eval()

    @torch.enable_grad()
    def train_step(self, batch):
        images = batch["image"].to(self.device)
        
        self.optimizer.zero_grad()
        z, ljd = self.model(images)
        loss = self.loss_fn(z, ljd)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}