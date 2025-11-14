```python
class Critic(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3)
        )
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.view(-1, 1)

class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_channels=1):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(True)
        )
        self.conv5 = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, z):
        x = z.view(-1, z.size(1), 1, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return torch.tanh(x)

    @torch.no_grad()
    def pred_step(self, z):
        images = self.forward(z)
        return dict(image=images)
```

```python
import torch
import torch.nn as nn

class WGANGP(nn.Module):
    def __init__(self, critic, generator, latent_dim, critic_steps=5, gp_weight=10.0, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.critic = critic.to(self.device)
        self.generator = generator.to(self.device)
        self.latent_dim = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight

        self.g_optimizer = None
        self.c_optimizer = None

    def gradient_penalty(self, real_images, fake_images, batch_size):
        alpha = torch.randn(batch_size, 1, 1, 1).to(real_images.device)
        interpolated = real_images * alpha + fake_images * (1 - alpha)
        interpolated.requires_grad_(True)

        pred = self.critic(interpolated)

        gradients = torch.autograd.grad(
            outputs=pred, inputs=interpolated,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        norm = gradients.norm(2, dim=1)
        gp = ((norm - 1.0) ** 2).mean()
        return gp

    def train_step(self, batch):
        real_images = batch["image"].to(self.device)
        batch_size = real_images.size(0)

        # Critic 학습 (여러 스텝)
        for _ in range(self.critic_steps):
            self.c_optimizer.zero_grad()
            z = torch.randn(batch_size, self.latent_dim).to(self.device)

            fake_images = self.generator(z).detach()
            real_preds = self.critic(real_images)
            fake_preds = self.critic(fake_images)

            c_wass_loss = fake_preds.mean() - real_preds.mean()
            gp = self.gradient_penalty(real_images, fake_images, batch_size)
            c_loss = c_wass_loss + gp * self.gp_weight

            c_loss.backward()
            self.c_optimizer.step()

        # Generator 학습
        self.g_optimizer.zero_grad()
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_images = self.generator(z)
        fake_preds = self.critic(fake_images)
        g_loss = -fake_preds.mean()

        g_loss.backward()
        self.g_optimizer.step()

        return dict(
            c_loss=c_loss,
            c_wass_loss=c_wass_loss,
            c_gp=gp,
            g_loss=g_loss,
            real_score=real_preds.mean(),
            fake_score=fake_preds.mean()
        )
```
