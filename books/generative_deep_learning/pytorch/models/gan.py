import torch
import torch.nn as nn

from torchmetrics.classification import BinaryAccuracy  


class GAN(nn.Module):
    def __init__(self, discriminator, generator, latent_dim=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = discriminator.to(self.device)
        self.g_model = generator.to(self.device)

        self.d_optimizer = torch.optim.Adam(self.d_model.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.g_model.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.latent_dim = latent_dim or generator.latent_dim
        self.loss_fn = nn.BCELoss()
        self.acc_metric = BinaryAccuracy().to(self.device)

    # @property
    # def device(self):
    #     return next(self.parameters()).device

    def set_optimziers(self, d_optimizer, g_optimizer):
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def train_step(self, batch):
        batch_size = batch["image"].shape[0]
        z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        real_labels = torch.ones((batch_size, 1)).to(self.device)
        fake_labels = torch.zeros((batch_size, 1)).to(self.device)

        # Train discriminator
        self.d_optimizer.zero_grad()
        real_images = batch["image"].to(self.device)
        real_preds = self.d_model(real_images)
        d_real_loss = self.loss_fn(real_preds, real_labels)
        d_real_loss.backward()

        fake_images = self.g_model(z).detach()
        fake_preds = self.d_model(fake_images)
        d_fake_loss = self.loss_fn(fake_preds, fake_labels)
        d_fake_loss.backward()
        self.d_optimizer.step()

        with torch.no_grad():
            d_real_acc = self.acc_metric(real_preds, real_labels)
            d_fake_acc = self.acc_metric(fake_preds, fake_labels)

        # Train generator
        self.g_optimizer.zero_grad()
        fake_images = self.g_model(z)
        fake_preds = self.d_model(fake_images)
        g_loss = self.loss_fn(fake_preds, real_labels)
        g_loss.backward()
        self.g_optimizer.step()

        with torch.no_grad():
            g_acc = self.acc_metric(fake_preds, real_labels)

        return dict(real_loss=d_real_loss, fake_loss=d_fake_loss, gen_loss=g_loss,
                    real_acc=d_real_acc, fake_acc=d_fake_acc, gen_acc=g_acc)