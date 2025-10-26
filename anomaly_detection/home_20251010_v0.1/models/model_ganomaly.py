"""
- GANomaly (2018): Semi-Supervised Anomaly Detection via Adversarial Training
  - https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/ganomaly
  - https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/ganomaly.html
  - https://arxiv.org/abs/1805.06725
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

# from anomalib.data import InferenceBatch


###########################################################
# anomalib/src/anomalib/models/image/ganomaly/loss.py
###########################################################

class GeneratorLoss(nn.Module):
    def __init__(self, wadv: int = 1, wcon: int = 50, wenc: int = 1) -> None:
        super().__init__()

        self.loss_enc = nn.SmoothL1Loss()
        self.loss_adv = nn.MSELoss()
        self.loss_con = nn.L1Loss()

        self.wadv = wadv
        self.wcon = wcon
        self.wenc = wenc

    def forward(
        self,
        latent_i: torch.Tensor,
        latent_o: torch.Tensor,
        images: torch.Tensor,
        fake: torch.Tensor,
        pred_real: torch.Tensor,
        pred_fake: torch.Tensor,
    ) -> torch.Tensor:
        error_enc = self.loss_enc(latent_i, latent_o)
        error_con = self.loss_con(images, fake)
        error_adv = self.loss_adv(pred_real, pred_fake)
        return error_adv * self.wadv + error_con * self.wcon + error_enc * self.wenc


class DiscriminatorLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_bce = nn.BCELoss()

    def forward(self, pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
        error_discriminator_real = self.loss_bce(
            pred_real,
            torch.ones(size=pred_real.shape, dtype=torch.float32, device=pred_real.device),
        )
        error_discriminator_fake = self.loss_bce(
            pred_fake,
            torch.zeros(size=pred_fake.shape, dtype=torch.float32, device=pred_fake.device),
        )
        return (error_discriminator_fake + error_discriminator_real) * 0.5


###########################################################
# anomalib/src/anomalib/data/utils/image.py
###########################################################

def pad_nextpow2(batch: torch.Tensor) -> torch.Tensor:
    l_dim = 2 ** math.ceil(math.log(max(*batch.shape[-2:]), 2))
    padding_w = [math.ceil((l_dim - batch.shape[-2]) / 2), math.floor((l_dim - batch.shape[-2]) / 2)]
    padding_h = [math.ceil((l_dim - batch.shape[-1]) / 2), math.floor((l_dim - batch.shape[-1]) / 2)]
    return F.pad(batch, pad=[*padding_h, *padding_w])


###########################################################
# anomalib\models\images\ganomaly\torch_model.py
###########################################################

class Encoder(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
    ) -> None:
        super().__init__()

        self.input_layers = nn.Sequential()
        self.input_layers.add_module(
            f"initial-conv-{num_input_channels}-{n_features}",
            nn.Conv2d(num_input_channels, n_features, kernel_size=4, stride=2, padding=4, bias=False),
        )
        self.input_layers.add_module(f"initial-relu-{n_features}", nn.LeakyReLU(0.2, inplace=True))

        # Extra Layers
        self.extra_layers = nn.Sequential()

        for layer in range(extra_layers):
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_features}-conv",
                nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.extra_layers.add_module(f"extra-layers-{layer}-{n_features}-batchnorm", nn.BatchNorm2d(n_features))
            self.extra_layers.add_module(f"extra-layers-{layer}-{n_features}-relu", nn.LeakyReLU(0.2, inplace=True))

        # Create pyramid features to reach latent vector
        self.pyramid_features = nn.Sequential()
        pyramid_dim = min(*input_size) // 2  # Use the smaller dimension to create pyramid.
        while pyramid_dim > 4:
            in_features = n_features
            out_features = n_features * 2
            self.pyramid_features.add_module(
                f"pyramid-{in_features}-{out_features}-conv",
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False),
            )
            self.pyramid_features.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            self.pyramid_features.add_module(f"pyramid-{out_features}-relu", nn.LeakyReLU(0.2, inplace=True))
            n_features = out_features
            pyramid_dim = pyramid_dim // 2

        # Final conv
        if add_final_conv_layer:
            self.final_conv_layer = nn.Conv2d(
                n_features,
                latent_vec_size,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self.input_layers(input_tensor)
        output = self.extra_layers(output)
        output = self.pyramid_features(output)
        if self.final_conv_layer is not None:
            output = self.final_conv_layer(output)

        return output


class Decoder(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
    ) -> None:
        super().__init__()

        self.latent_input = nn.Sequential()

        # Calculate input channel size to recreate inverse pyramid
        exp_factor = math.ceil(math.log(min(input_size) // 2, 2)) - 2
        n_input_features = n_features * (2**exp_factor)

        # CNN layer for latent vector input
        self.latent_input.add_module(
            f"initial-{latent_vec_size}-{n_input_features}-convt",
            nn.ConvTranspose2d(
                latent_vec_size,
                n_input_features,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
        self.latent_input.add_module(f"initial-{n_input_features}-batchnorm", nn.BatchNorm2d(n_input_features))
        self.latent_input.add_module(f"initial-{n_input_features}-relu", nn.ReLU(inplace=True))

        # Create inverse pyramid
        self.inverse_pyramid = nn.Sequential()
        pyramid_dim = min(*input_size) // 2  # Use the smaller dimension to create pyramid.
        while pyramid_dim > 4:
            in_features = n_input_features
            out_features = n_input_features // 2
            self.inverse_pyramid.add_module(
                f"pyramid-{in_features}-{out_features}-convt",
                nn.ConvTranspose2d(
                    in_features,
                    out_features,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            )
            self.inverse_pyramid.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            self.inverse_pyramid.add_module(f"pyramid-{out_features}-relu", nn.ReLU(inplace=True))
            n_input_features = out_features
            pyramid_dim = pyramid_dim // 2

        # Extra Layers
        self.extra_layers = nn.Sequential()
        for layer in range(extra_layers):
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-conv",
                nn.Conv2d(n_input_features, n_input_features, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-batchnorm",
                nn.BatchNorm2d(n_input_features),
            )
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-relu",
                nn.LeakyReLU(0.2, inplace=True),
            )

        # Final layers
        self.final_layers = nn.Sequential()
        self.final_layers.add_module(
            f"final-{n_input_features}-{num_input_channels}-convt",
            nn.ConvTranspose2d(
                n_input_features,
                num_input_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
        )
        self.final_layers.add_module(f"final-{num_input_channels}-tanh", nn.Tanh())

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self.latent_input(input_tensor)
        output = self.inverse_pyramid(output)
        output = self.extra_layers(output)
        return self.final_layers(output)


class Discriminator(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
    ) -> None:
        super().__init__()
        encoder = Encoder(input_size, 1, num_input_channels, n_features, extra_layers)
        layers = []
        for block in encoder.children():
            if isinstance(block, nn.Sequential):
                layers.extend(list(block.children()))
            else:
                layers.append(block)

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(input_tensor)
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)
        return classifier, features


class Generator(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
    ) -> None:
        super().__init__()
        self.encoder1 = Encoder(
            input_size,
            latent_vec_size,
            num_input_channels,
            n_features,
            extra_layers,
            add_final_conv_layer,
        )
        self.decoder = Decoder(input_size, latent_vec_size, num_input_channels, n_features, extra_layers)
        self.encoder2 = Encoder(
            input_size,
            latent_vec_size,
            num_input_channels,
            n_features,
            extra_layers,
            add_final_conv_layer,
        )

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_i = self.encoder1(input_tensor)
        gen_image = self.decoder(latent_i)
        latent_o = self.encoder2(gen_image)
        return gen_image, latent_i, latent_o


class GanomalyModel(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        num_input_channels: int,
        n_features: int,
        latent_vec_size: int,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
    ) -> None:
        super().__init__()
        self.generator: Generator = Generator(
            input_size=input_size,
            latent_vec_size=latent_vec_size,
            num_input_channels=num_input_channels,
            n_features=n_features,
            extra_layers=extra_layers,
            add_final_conv_layer=add_final_conv_layer,
        )
        self.discriminator: Discriminator = Discriminator(
            input_size=input_size,
            num_input_channels=num_input_channels,
            n_features=n_features,
            extra_layers=extra_layers,
        )
        self.weights_init(self.generator)
        self.weights_init(self.discriminator)

    @staticmethod
    def weights_init(module: nn.Module) -> None:

        classname = module.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def forward(
        self,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:

        padded_batch = pad_nextpow2(batch)
        fake, latent_i, latent_o = self.generator(padded_batch)
        if self.training:
            return padded_batch, fake, latent_i, latent_o
        scores = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)  # convert nx1x1 to n
        return dict(pred_score=scores)


#####################################################################
# Trainer for Ganomaly Model
#####################################################################
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from .components.trainer import BaseTrainer, EarlyStopper


class GanomalyTrainer(BaseTrainer):
    def __init__(self, model=None, optimizer=None, loss_fn=None, metrics=None, device=None,
                 scheduler=None, early_stopper_loss=None, early_stopper_auroc=None,
                 input_size=(256, 256), n_features=64, latent_vec_size=100, gamma=1):

        if model is None:
            model = GanomalyModel(input_size=input_size, num_input_channels=3, n_features=n_features,
                latent_vec_size=latent_vec_size, extra_layers=0, add_final_conv_layer=True)
        super().__init__(model, optimizer, loss_fn, metrics, device,
                         scheduler, early_stopper_loss, early_stopper_auroc)
        self.eval_period = 5

        self.generator_loss = GeneratorLoss(wadv=1, wcon=50, wenc=1).to(self.device)
        self.discriminator_loss = DiscriminatorLoss().to(self.device)

        lr = 0.0002 * gamma
        self.optimizer_g = torch.optim.Adam(model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        self.min_scores = float('inf')
        self.max_scores = float('-inf')


    def on_train_start(self, train_loader):
        self.model.train()

    @torch.enable_grad()
    def train_step(self, batch):
        images = batch["image"].to(self.device)

        padded, fake, latent_i, latent_o = self.model(images)

        self.optimizer_g.zero_grad()
        pred_real, _ = self.model.discriminator(padded)
        pred_fake, _ = self.model.discriminator(fake)
        g_loss = self.generator_loss(latent_i, latent_o, padded, fake, pred_real, pred_fake)
        g_loss.backward(retain_graph=True)
        self.optimizer_g.step()

        self.optimizer_d.zero_grad()
        pred_real, _ = self.model.discriminator(padded)
        pred_fake, _ = self.model.discriminator(fake.detach())
        d_loss = self.discriminator_loss(pred_real, pred_fake)
        d_loss.backward()
        self.optimizer_d.step()

        return {"loss": g_loss.item() + d_loss.item(), "g_loss": g_loss.item(), "d_loss": d_loss.item()}

    def on_validation_start(self, valid_loader):
        self.model.eval()
        self.min_scores = float('inf')
        self.max_scores = float('-inf')

    @torch.no_grad()
    def validation_step(self, batch):
        images = batch["image"].to(self.device)
        predictions = self.model(images)
        scores = predictions['pred_score'].cpu()

        self.min_scores = min(self.min_scores, torch.min(scores).item())
        self.max_scores = max(self.max_scores, torch.max(scores).item())

        if scores.ndim > 1:
            scores = scores.view(scores.size(0))
        return scores

    @torch.no_grad()
    def validation_epoch(self, loader):
        all_scores, all_labels = [], []

        with tqdm(loader, desc=" > Validation", leave=False, ascii=True) as pbar:
            for batch in pbar:
                scores = self.validation_step(batch)
                labels = batch["label"].cpu()
                all_scores.append(scores)
                all_labels.append(labels)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)

        if self.max_scores > self.min_scores:
            scores = (scores - self.min_scores) / (self.max_scores - self.min_scores)

        results = {"auroc": roc_auc_score(labels, scores),
                   "aupr": average_precision_score(labels, scores)}
        return results, scores, labels

    @torch.no_grad()
    def save_maps(self, test_loader, result_dir=None, desc=None, show_image=False,
                  skip_normal=False, skip_anomaly=False, num_max=-1, normalize=True):
        print("\n > GANomaly model does not support anomaly map visualization.")
        print(" > Skipping anomaly map generation.\n")

    def save_model(self, weight_path):
        if weight_path is not None:
            output_dir = os.path.abspath(os.path.dirname(weight_path))
            os.makedirs(output_dir, exist_ok=True)

            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer_g": self.optimizer_g.state_dict() if self.optimizer_g else None,
                "optimizer_d": self.optimizer_d.state_dict() if self.optimizer_d else None,
            }
            torch.save(checkpoint, weight_path)
            print(f" > Model and optimizers weights saved to: {weight_path}\n")

    def load_model(self, weight_path):
        if os.path.isfile(weight_path):
            checkpoint = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            print(f" > Model weights loaded from: {weight_path}")

            if self.optimizer_g and checkpoint.get("optimizer_g"):
                self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
                print(" > Generator optimizer state loaded.")

            if self.optimizer_d and checkpoint.get("optimizer_d"):
                self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
                print(" > Discriminator optimizer state loaded.")
        else:
            print(f" > No model weights found at: {weight_path}\n")
