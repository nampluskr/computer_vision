"""
- UniNet (2025): Unified Contrastive Learning Framework for Anomaly Detection
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/uninet
  - https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/uninet.html
  - https://github.com/pangdatangtt/UniNet
  - https://pangdatangtt.github.io/#poster (2025)
"""
from collections.abc import Callable

import os
import einops
import torch
import torchvision
from torch import nn
from torch.fx import GraphModule
from torch.nn import functional as F  # noqa: N812
from torchvision.models.feature_extraction import create_feature_extractor

# from anomalib.data import InferenceBatch
# from anomalib.models.components.backbone import get_decoder
# from .components import AttentionBottleneck, BottleneckLayer, DomainRelatedFeatureSelection, weighted_decision_mechanism

from .components.blur import GaussianBlur2d
from .components.resnet_decoder import get_decoder
from .components.backbone import get_backbone_path


#####################################################################
# anomalib/src/anomalib/models/image/uninet/components/anomaly_map.py
#####################################################################

def weighted_decision_mechanism(
    batch_size: int,
    output_list: list[torch.Tensor],
    alpha: float,
    beta: float,
    output_size: tuple[int, int] = (256, 256),
) -> tuple[torch.Tensor, torch.Tensor]:

    # Convert to tensor operations to avoid SequenceConstruct/ConcatFromSequence
    device = output_list[0].device
    num_outputs = len(output_list)

    # Pre-allocate tensors instead of using lists
    total_weights = torch.zeros(batch_size, device=device)
    gaussian_blur = GaussianBlur2d(sigma=4.0, kernel_size=(5, 5), channels=1).to(device)

    # Process each batch item individually
    for i in range(batch_size):
        # Get max value from each output for this batch item
        # Create tensor directly from max values to avoid list operations
        max_values = torch.zeros(num_outputs, device=device)
        for j, output_tensor in enumerate(output_list):
            max_values[j] = torch.max(output_tensor[i])

        probs = F.softmax(max_values, dim=0)

        # Use tensor operations instead of list filtering
        prob_mean = torch.mean(probs)
        mask = probs > prob_mean

        if mask.any():
            weight_tensor = max_values[mask]
            weight = torch.max(torch.stack([torch.mean(weight_tensor) * alpha, torch.tensor(beta, device=device)]))
        else:
            weight = torch.tensor(beta, device=device)

        total_weights[i] = weight

    # Process anomaly maps using tensor operations
    # Pre-allocate the processed anomaly maps tensor
    processed_anomaly_maps = torch.zeros(batch_size, *output_size, device=device)

    # Process each output tensor separately due to different spatial dimensions
    for output_tensor in output_list:
        # Interpolate current output to target size
        # Add channel dimension for interpolation: [batch_size, H, W] -> [batch_size, 1, H, W]
        output_resized = F.interpolate(
            output_tensor.unsqueeze(1),
            output_size,
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)  # [batch_size, H_out, W_out]

        # Add to accumulated anomaly maps
        processed_anomaly_maps += output_resized

    # Pre-allocate anomaly scores tensor instead of using list
    anomaly_scores = torch.zeros(batch_size, device=device)

    for idx in range(batch_size):
        top_k = int(output_size[0] * output_size[1] * total_weights[idx])
        top_k = max(top_k, 1)  # Ensure at least 1 element

        single_anomaly_score_exp = processed_anomaly_maps[idx]
        single_anomaly_score_exp = gaussian_blur(einops.rearrange(single_anomaly_score_exp, "h w -> 1 1 h w"))
        single_anomaly_score_exp = single_anomaly_score_exp.squeeze()

        # Flatten and get top-k values
        single_map_flat = single_anomaly_score_exp.view(-1)
        top_k_values = torch.topk(single_map_flat, top_k).values
        single_anomaly_score = top_k_values[0] if len(top_k_values) > 0 else torch.tensor(0.0, device=device)
        anomaly_scores[idx] = single_anomaly_score.detach()

    return anomaly_scores.unsqueeze(1), processed_anomaly_maps.detach()

#####################################################################
# anomalib/src/anomalib/models/image/uninet/components/dfs.py
#####################################################################

class DomainRelatedFeatureSelection(nn.Module):
    def __init__(self, num_channels: int = 256, learnable: bool = True) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.theta1 = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.theta2 = nn.Parameter(torch.zeros(1, num_channels * 2, 1, 1))
        self.theta3 = nn.Parameter(torch.zeros(1, num_channels * 4, 1, 1))
        self.learnable = learnable

    def _get_theta(self, idx: int) -> torch.Tensor:
        match idx:
            case 1:
                return self.theta1
            case 2:
                return self.theta2
            case 3:
                return self.theta3
            case _:
                msg = f"Invalid index: {idx}"
                raise ValueError(msg)

    def forward(
        self,
        source_features: list[torch.Tensor],
        target_features: list[torch.Tensor],
        conv: bool = False,
        maximize: bool = True,
    ) -> list[torch.Tensor]:

        features = []
        for idx, (source_feature, target_feature) in enumerate(zip(source_features, target_features, strict=True)):
            theta = 1
            if self.learnable:
                #  to avoid losing local weight, theta should be as non-zero value as possible
                if idx < 3:
                    theta = torch.clamp(torch.sigmoid(self._get_theta(idx + 1)) * 1.0 + 0.5, max=1)
                else:
                    theta = torch.clamp(torch.sigmoid(self._get_theta(idx - 2)) * 1.0 + 0.5, max=1)

            b, c, h, w = source_feature.shape
            if not conv:
                prior_flat = target_feature.view(b, c, -1)
                if maximize:
                    prior_flat_ = prior_flat.max(dim=-1, keepdim=True)[0]
                    prior_flat = prior_flat - prior_flat_
                weights = F.softmax(prior_flat, dim=-1)
                weights = weights.view(b, c, h, w)

                global_inf = target_feature.mean(dim=(-2, -1), keepdim=True)

                inter_weights = weights * (theta + global_inf)

                x_ = source_feature * inter_weights
                features.append(x_)

        return features


#####################################################################
# anomalib/src/anomalib/models/image/uninet/components/attention_bottleneck.py
#####################################################################

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def fuse_bn(conv: nn.Module, bn: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class AttentionBottleneck(nn.Module):
    channel_expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Callable[..., nn.Module] | None = None,
        attention: bool = True,
        halve: int = 1,
    ) -> None:
        super().__init__()
        self.attention = attention
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups  # 512
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.halve = halve
        k = 7
        p = 3

        self.bn2 = norm_layer(width // halve)
        self.conv3 = conv1x1(width, planes * self.channel_expansion)
        self.bn3 = norm_layer(planes * self.channel_expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

        self.bn4 = norm_layer(width // 2)
        self.bn5 = norm_layer(width // 2)
        self.bn6 = norm_layer(width // 2)
        self.bn7 = norm_layer(width)
        self.conv3x3 = nn.Conv2d(inplanes // 2, width // 2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3x3_ = nn.Conv2d(width // 2, width // 2, 3, 1, 1, bias=False)
        self.conv7x7 = nn.Conv2d(inplanes // 2, width // 2, kernel_size=k, stride=stride, padding=p, bias=False)
        self.conv7x7_ = nn.Conv2d(width // 2, width // 2, k, 1, p, bias=False)

    def get_same_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get same kernel and bias of the bottleneck."""
        k1, b1 = fuse_bn(self.conv3x3, self.bn2)
        k2, b2 = fuse_bn(self.conv3x3_, self.bn6)

        return k1, b1, k2, b2

    def merge_kernel(self) -> None:
        """Merge kernel of the bottleneck."""
        k1, b1, k2, b2 = self.get_same_kernel_bias()
        self.conv7x7 = nn.Conv2d(
            self.conv3x3.in_channels,
            self.conv3x3.out_channels,
            self.conv3x3.kernel_size,
            self.conv3x3.stride,
            self.conv3x3.padding,
            self.conv3x3.dilation,
            self.conv3x3.groups,
        )
        self.conv7x7_ = nn.Conv2d(
            self.conv3x3_.in_channels,
            self.conv3x3_.out_channels,
            self.conv3x3_.kernel_size,
            self.conv3x3_.stride,
            self.conv3x3_.padding,
            self.conv3x3_.dilation,
            self.conv3x3_.groups,
        )
        self.conv7x7.weight.data = k1
        self.conv7x7.bias.data = b1
        self.conv7x7_.weight.data = k2
        self.conv7x7_.bias.data = b2

    @staticmethod
    def _process_branch(
        branch: torch.Tensor,
        conv1: nn.Module,
        bn1: nn.Module,
        conv2: nn.Module,
        bn2: nn.Module,
        relu: nn.Module,
    ) -> torch.Tensor:

        out = conv1(branch)
        out = bn1(out)
        out = relu(out)
        out = conv2(out)
        return bn2(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.halve == 1:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

        else:
            num_channels = x.shape[1]
            x_split = torch.split(x, [num_channels // 2, num_channels // 2], dim=1)

            out1 = self._process_branch(x_split[0], self.conv3x3, self.bn2, self.conv3x3_, self.bn5, self.relu)
            out2 = self._process_branch(x_split[-1], self.conv7x7, self.bn4, self.conv7x7_, self.bn6, self.relu)

            out = torch.cat([out1, out2], dim=1)

            out = self.conv3(out)
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)


class BottleneckLayer(nn.Module):
    def __init__(
        self,
        block: type[AttentionBottleneck],
        layers: int,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Callable[..., nn.Module] | None = None,
        halve: int = 2,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 256 * block.channel_expansion
        self.halve = halve
        self.bn_layer = nn.Sequential(self._make_layer(block, 512, layers, stride=2))
        self.conv1 = conv3x3(64 * block.channel_expansion, 128 * block.channel_expansion, 2)
        self.bn1 = norm_layer(128 * block.channel_expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(128 * block.channel_expansion, 256 * block.channel_expansion, 2)
        self.bn2 = norm_layer(256 * block.channel_expansion)
        self.conv3 = conv3x3(128 * block.channel_expansion, 256 * block.channel_expansion, 2)
        self.bn3 = norm_layer(256 * block.channel_expansion)

        self.conv4 = conv1x1(1024 * block.channel_expansion, 512 * block.channel_expansion, 1)
        self.bn4 = norm_layer(512 * block.channel_expansion)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d | nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif hasattr(module, "merge_kernel") and self.halve == 2:
                module.merge_kernel()

    def _make_layer(
        self,
        block: type[AttentionBottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:

        norm_layer = self._norm_layer
        downsample = None
        if dilate:
            stride = 1
        if stride != 1 or self.inplanes != planes * block.channel_expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes * 3, planes * block.channel_expansion, stride),
                norm_layer(planes * block.channel_expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes * 3,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                norm_layer,
                halve=self.halve,
            ),
        )
        self.inplanes = planes * block.channel_expansion
        layers.extend(
            block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                norm_layer=norm_layer,
                halve=self.halve,
            )
            for _ in range(1, blocks)
        )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x[0]))))))
        l2 = self.relu(self.bn3(self.conv3(x[1])))
        feature = torch.cat([l1, l2, x[2]], 1)
        output = self.bn_layer(feature)  # 16*2048*8*8
        return output.contiguous()


#####################################################################
# anomalib/src/anomalib/models/image/uninet/torch_model.py
#####################################################################

class UniNetLoss(nn.Module):
    def __init__(self, lambda_weight: float = 0.7, temperature: float = 2.0) -> None:
        super().__init__()
        self.lambda_weight = lambda_weight
        self.temperature = temperature

    def forward(
        self,
        student_features: list[torch.Tensor],
        teacher_features: list[torch.Tensor],
        margin: int = 1,
        mask: torch.Tensor | None = None,
        stop_gradient: bool = False,
    ) -> torch.Tensor:

        loss = 0.0
        margin_loss_a = 0.0

        for idx in range(len(student_features)):
            student_feature = student_features[idx]
            teacher_feature = teacher_features[idx].detach() if stop_gradient else teacher_features[idx]

            n, c, h, w = student_feature.shape
            student_feature = student_feature.view(n, c, -1).transpose(1, 2)  # (N, H+W, C)
            teacher_feature = teacher_feature.view(n, c, -1).transpose(1, 2)  # (N, H+W, C)

            student_feature_normalized = F.normalize(student_feature, p=2, dim=2)
            teacher_feature_normalized = F.normalize(teacher_feature, p=2, dim=2)

            cosine_loss = 1 - F.cosine_similarity(
                student_feature_normalized,
                teacher_feature_normalized,
                dim=2,
            )
            cosine_loss = cosine_loss.mean()

            similarity = (
                torch.matmul(student_feature_normalized, teacher_feature_normalized.transpose(1, 2)) / self.temperature
            )
            similarity = torch.exp(similarity)
            similarity_sum = similarity.sum(dim=2, keepdim=True)
            similarity = similarity / (similarity_sum + 1e-8)
            diag_sum = torch.diagonal(similarity, dim1=1, dim2=2)

            # unsupervised and only normal (or abnormal)
            if mask is None:
                contrastive_loss = -torch.log(diag_sum + 1e-8).mean()
                margin_loss_n = F.relu(margin - diag_sum).mean()

            # supervised
            else:
                # gt label
                if len(mask.shape) < 3:
                    normal_mask = mask == 0
                    abnormal_mask = mask == 1
                # gt mask
                else:
                    mask_ = F.interpolate(mask, size=(h, w), mode="nearest").squeeze(1)
                    mask_flat = mask_.view(mask_.size(0), -1)

                    normal_mask = mask_flat == 0
                    abnormal_mask = mask_flat == 1

                if normal_mask.sum() > 0:
                    diag_sim_normal = diag_sum[normal_mask]
                    contrastive_loss = -torch.log(diag_sim_normal + 1e-8).mean()
                    margin_loss_n = F.relu(margin - diag_sim_normal).mean()

                if abnormal_mask.sum() > 0:
                    diag_sim_abnormal = diag_sum[abnormal_mask]
                    margin_loss_a = F.relu(diag_sim_abnormal - margin / 2).mean()

            margin_loss = margin_loss_n + margin_loss_a

            loss += cosine_loss * self.lambda_weight + contrastive_loss * (1 - self.lambda_weight) + margin_loss

        return loss

#####################################################################
# anomalib/src/anomalib/models/image/uninet/torch_model.py
#####################################################################

class UniNetModel(nn.Module):
    def __init__(
        self,
        student_backbone: str,
        teacher_backbone: str,
        loss: nn.Module,
    ) -> None:
        super().__init__()
        self.teachers = Teachers(teacher_backbone)
        self.student = get_decoder(student_backbone)
        self.bottleneck = BottleneckLayer(block=AttentionBottleneck, layers=3)
        self.dfs = DomainRelatedFeatureSelection()

        self.loss = loss
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        # Used to post-process the student features from the de_resnet model to get the predictions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1)

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:

        source_target_features, bottleneck_inputs = self.teachers(images)
        bottleneck_outputs = self.bottleneck(bottleneck_inputs)

        student_features = self.student(bottleneck_outputs)

        # These predictions are part of the de_resnet model of the original code.
        # since we are using the de_resnet model from anomalib, we need to compute predictions here
        predictions = self.avgpool(student_features[0])
        predictions = torch.flatten(predictions, 1)
        predictions = self.fc(predictions).squeeze()
        predictions = predictions.chunk(dim=0, chunks=2)

        student_features = [d.chunk(dim=0, chunks=2) for d in student_features]
        student_features = [
            student_features[0][0],
            student_features[1][0],
            student_features[2][0],
            student_features[0][1],
            student_features[1][1],
            student_features[2][1],
        ]
        if self.training:
            student_features = self._feature_selection(source_target_features, student_features)
            return self._compute_loss(
                student_features,
                source_target_features,
                predictions,
                labels,
                masks,
            )

        output_list: list[torch.Tensor] = []
        for target_feature, student_feature in zip(source_target_features, student_features, strict=True):
            output = 1 - F.cosine_similarity(target_feature, student_feature)  # B*64*64
            output_list.append(output)

        anomaly_score, anomaly_map = weighted_decision_mechanism(
            batch_size=images.shape[0],
            output_list=output_list,
            alpha=0.01,
            beta=3e-05,
            output_size=images.shape[-2:],
        )
        return dict(
            pred_score=anomaly_score,
            anomaly_map=anomaly_map,
        )

    def _compute_loss(
        self,
        student_features: list[torch.Tensor],
        teacher_features: list[torch.Tensor],
        predictions: tuple[torch.Tensor, torch.Tensor] | None = None,
        label: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        stop_gradient: bool = False,
    ) -> torch.Tensor:
        if mask is not None:
            mask_ = mask.float().unsqueeze(1)  # Bx1xHxW
        else:
            assert label is not None, "Label is required when mask is not provided"
            mask_ = label.float()

        loss = self.loss(student_features, teacher_features, mask=mask_, stop_gradient=stop_gradient)
        if predictions is not None and label is not None:
            loss += self.bce_loss(predictions[0], label.float()) + self.bce_loss(predictions[1], label.float())

        return loss

    def _feature_selection(
        self,
        target_features: list[torch.Tensor],
        source_features: list[torch.Tensor],
        maximize: bool = True,
    ) -> list[torch.Tensor]:
        return self.dfs(source_features, target_features, maximize=maximize)


class Teachers(nn.Module):
    def __init__(self, teacher_backbone: str) -> None:
        super().__init__()
        self.source_teacher = self._get_teacher(teacher_backbone).eval()
        self.target_teacher = self._get_teacher(teacher_backbone)

    # @staticmethod
    # def _get_teacher(backbone: str) -> GraphModule:
    #     model = getattr(torchvision.models, backbone)(pretrained=True)
    #     return create_feature_extractor(model, return_nodes=["layer3", "layer2", "layer1"])
    
    @staticmethod
    def _get_teacher(backbone: str) -> GraphModule:
        model = getattr(torchvision.models, backbone)(weights=None)
        weight_path = get_backbone_path(backbone)

        if os.path.isfile(weight_path):
            state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print(f" > Loaded teacher backbone from {weight_path}")
        else:
            print(f" > Warning: Teacher weights not found at {weight_path}")
        return create_feature_extractor(model, return_nodes=["layer3", "layer2", "layer1"])
    

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        with torch.no_grad():
            source_features = self.source_teacher(images)

        target_features = self.target_teacher(images)

        bottleneck_inputs = [
            torch.cat([a, b], dim=0) for a, b in zip(target_features.values(), source_features.values(), strict=True)
        ]  # 512, 1024, 2048

        return list(source_features.values()) + list(target_features.values()), bottleneck_inputs


#####################################################################
# Trainer for UniNet Model
#####################################################################
from .components.trainer import BaseTrainer, EarlyStopper

class UniNetTrainer(BaseTrainer):
    def __init__(self, model=None, optimizer=None, loss_fn=None, metrics=None, device=None,
                 scheduler=None, early_stopper_loss=None, early_stopper_auroc=None,
                 student_backbone="wide_resnet50_2", teacher_backbone="wide_resnet50_2", temperature=0.1):

        if loss_fn is None:
            loss_fn = UniNetLoss(temperature=temperature)
        if model is None:
            model = UniNetModel(student_backbone, teacher_backbone, loss=loss_fn)
        if optimizer is None:
            optimizer = torch.optim.AdamW([
                {"params": model.student.parameters()},
                {"params": model.bottleneck.parameters()},
                {"params": model.dfs.parameters()},
                {"params": model.teachers.target_teacher.parameters(), "lr": 1e-6},
            ], lr=5e-3, betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-10, amsgrad=True)
        if early_stopper_loss is None:
            early_stopper_loss = EarlyStopper(patience=10, min_delta=0.01, mode='min', target_value=1.0)
        if early_stopper_auroc is None:
            early_stopper_auroc = EarlyStopper(patience=10, min_delta=0.001, mode='max', target_value=0.995)

        super().__init__(model, optimizer, loss_fn, metrics, device,
                         scheduler, early_stopper_loss, early_stopper_auroc)
        self.eval_period = 5
        
    def on_fit_start(self):
        super().on_fit_start()
        
        if self.scheduler is None and self.optimizer is not None:
            milestone = int(self.num_epochs * 0.8)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                milestones=[milestone], gamma=0.2)
            print(f" > Scheduler milestone: [epoch {milestone}]\n")
        
    @torch.enable_grad()
    def train_step(self, batch):
        images = batch["image"].to(self.device)
        masks = batch["mask"].to(self.device)
        labels = batch["label"].to(self.device)

        self.optimizer.zero_grad()
        loss = self.model(images, masks, labels)
        loss.backward()
        self.optimizer.step()

        results = {"loss": loss.item()}
        return results