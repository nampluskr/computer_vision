import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class WGAN(nn.Module):
    def __init__(self, critic, generator, latent_dim=None, device=None, use_gp=True, one_sided=True):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.critic = critic.to(self.device)
        self.generator = generator.to(self.device)

        if use_gp:
            self.c_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
            self.g_optimizer = optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
        else:
            self.c_optimizer = optim.RMSprop(self.critic.parameters(), lr=5e-5)
            self.g_optimizer = optim.RMSprop(self.generator.parameters(), lr=5e-5)

        self.latent_dim = latent_dim or generator.latent_dim
        self.clip_value = 0.01
        self.gp_lambda = 10.0
        self.d_steps = 5
        self.use_gp = use_gp
        self.one_sided = one_sided
        self.global_epoch = 0

    def gradient_penalty(self, real_images, fake_images):
        batch_size = real_images.size(0)

        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interpolates = alpha * real_images + (1 - alpha) * fake_images
        interpolates = interpolates.requires_grad_(True)
        score = self.critic(interpolates)

        gradients = torch.autograd.grad(
            outputs=score,
            inputs=interpolates,
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradients_norm = gradients.norm(2, dim=1)

        if self.one_sided:
            return torch.mean(torch.relu(gradients_norm - 1) ** 2)
        else:
            return torch.mean((gradients_norm - 1) ** 2)

    def d_loss_fn(self, real_logits, fake_logits):
        d_real_loss = -real_logits.mean()
        d_fake_loss = fake_logits.mean()
        d_loss = d_real_loss + d_fake_loss
        return d_loss, d_real_loss, d_fake_loss

    def g_loss_fn(self, fake_logits):
        return -fake_logits.mean()

    def train_step(self, batch):
        batch_size = batch["image"].size(0)

        # (1) Update Discriminator (Critic)
        real_images = batch["image"].to(self.device)
        for _ in range(self.d_steps):
            real_logits = self.critic(real_images)
            noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
            fake_images = self.generator(noises).detach()
            fake_logits = self.critic(fake_images)

            d_loss, d_real_loss, d_fake_loss = self.d_loss_fn(real_logits, fake_logits)
            if self.use_gp:
                gp = self.gradient_penalty(real_images, fake_images)
            else:
                gp = torch.tensor(0.0).to(self.device)
            d_loss_gp = d_loss + gp * self.gp_lambda

            self.c_optimizer.zero_grad()
            d_loss_gp.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.c_optimizer.step()

            if not self.use_gp:
                for param in self.critic.parameters():
                    param.data.clamp_(-self.clip_value, self.clip_value)

        # (2) Update Generator
        noises = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.generator(noises)
        fake_logits = self.critic(fake_images)
        g_loss = self.g_loss_fn(fake_logits)

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        results = dict(d_loss=d_loss.item(),
            real_loss=d_real_loss.item(),
            fake_loss=d_fake_loss.item(),
            g_loss=g_loss.item(),
        )
        if self.use_gp:
            results["gp"] = gp.item()
        return results
