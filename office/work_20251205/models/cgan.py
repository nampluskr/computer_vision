import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gan import GAN


# class CGenerator(nn.Module):
#     def __init__(self, generator, num_classes=10, embedding_dim=64):
#         super().__init__()
#         self.generator = generator
#         self.num_classes = num_classes
#         self.embedding_dim = embedding_dim
#         self.labels_embedding = nn.Sequential(
#             nn.Embedding(num_classes, embedding_dim),
#             nn.Unflatten(dim=1, unflattened_size=(embedding_dim, 1, 1))
#         )

#     def forward(self, noises, labels):
#         labels = self.labels_embedding(labels)
#         z = torch.cat([noises, labels], dim=1)
#         return self.generator(z)


# class CDiscriminator(nn.Module):
#     def __init__(self, discriminator, num_classes=10, embedding_channels=64):
#         super().__init__()
#         self.discriminator = discriminator
#         self.num_classes = num_classes
#         self.embedding_channels = embedding_channels
#         self.labels_embedding = nn.Embedding(num_classes, embedding_channels)

#     def forward(self, images, labels):
#         h, w = images.size(-2), images.size(-1)
#         labels = self.labels_embedding(labels)
#         labels = labels.view(-1, self.embedding_channels, 1, 1).expand(-1, -1, h, w)
#         x = torch.cat([images, labels], dim=1)
#         return self.discriminator(x)


class CGenerator32(nn.Module):
    def __init__(self, latent_dim=100, out_channels=3, base=64, num_classes=10, embedding_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.labels_embedding = nn.Sequential(
            nn.Embedding(num_classes, embedding_dim),
            nn.Unflatten(dim=1, unflattened_size=(embedding_dim, 1, 1))
        )
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + embedding_dim, base*4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base*4),
            nn.ReLU(True),
        )
        self.blocks = nn.Sequential(
            DeconvBlock(base*4, base*2),
            DeconvBlock(base*2, base),
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, noises, labels):
        noises = noises.view(-1, self.latent_dim, 1, 1)
        labels = self.labels_embedding(labels)
        z = torch.cat([noises, labels], dim=1)
        x = self.initial(z)
        x = self.blocks(x)
        x = self.final(x)
        return x


class CDiscriminator32(nn.Module):
    def __init__(self, in_channels=3, base=64, num_classes=10, embedding_channels=1):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_channels = embedding_channels
        self.labels_embedding = nn.Embedding(num_classes, embedding_channels)

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels + embedding_channels, base, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(
            ConvBlock(base, base*2),
            ConvBlock(base*2, base*4),
        )
        self.final = nn.Conv2d(base*4, 1, kernel_size=4, stride=1, padding=0, bias=False)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, images, labels):
        h, w = images.size(-2), images.size(-1)
        labels = self.labels_embedding(labels)
        labels = labels.view(-1, self.embedding_channels, 1, 1).expand(-1, -1, h, w)
        x = torch.cat([images, labels], dim=1)  # (N, in_channels + C_emb, 32, 32)
        x = self.initial(x)
        x = self.blocks(x)
        logits = self.final(x).view(-1, 1)
        return logits


class CGAN(GAN):
    def train_step(self, batch):
        batch_size = batch["image"].size(0)
        labels = batch["label"].to(self.device)

        # (1) Update Discriminator
        real_images = batch["image"].to(self.device)
        real_logits = self.discriminator(real_images, labels)

        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.generator(noises, labels).detach()
        fake_logits = self.discriminator(fake_images, labels)

        d_loss, d_real_loss, d_fake_loss = self.d_loss_fn(real_logits, fake_logits)

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # (2) Update Generator
        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.generator(noises, labels)
        fake_logits = self.discriminator(fake_images, labels)
        g_loss = self.g_loss_fn(fake_logits)

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return dict(
            d_loss=d_loss.item(),
            real_loss=d_real_loss.item(),
            fake_loss=d_fake_loss.item(),
            g_loss=g_loss.item()
        )
