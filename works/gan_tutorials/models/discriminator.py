import torch
import torch.nn as nn
import torch.nn.functional as F


#####################################################################
# Discriminators for GAN
#####################################################################

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Discriminator32(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=4, stride=2, padding=1, bias=False),
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

    def forward(self, images):
        x = self.initial(images)
        x = self.blocks(x)
        logits = self.final(x).view(-1, 1)
        return logits


class Discriminator64(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(
            ConvBlock(base, base*2),
            ConvBlock(base*2, base*4),
            ConvBlock(base*4, base*8),
        )
        self.final = nn.Conv2d(base*8, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, images):
        x = self.initial(images)
        x = self.blocks(x)
        logits = self.final(x).view(-1, 1)
        return logits


class Discriminator(nn.Module):
    def __init__(self, img_size=32, in_channels=3, base=64):
        super().__init__()
        self.img_size = img_size

        num_blocks = {32:  2, 64:  3, 128: 4, 256: 5}
        if img_size not in num_blocks:
            raise ValueError(f"Unsupported img_size: {img_size}")

        out_channels = base
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        blocks = []
        for i in range(num_blocks[img_size]):
            blocks.append(ConvBlock(out_channels, out_channels * 2))
            out_channels *= 2
        self.blocks = nn.Sequential(*blocks)

        self.final = nn.Conv2d(out_channels, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, images):
        x = self.initial(images)
        x = self.blocks(x)
        logits = self.final(x).view(-1, 1)
        return logits


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


class CDiscriminator(nn.Module):
    def __init__(self, img_size=32, in_channels=3, base=64, 
                 num_classes=10, embedding_channels=1):
        super().__init__()
        self.img_size = img_size
        self.embedding_channels = embedding_channels
        self.labels_embedding = nn.Embedding(num_classes, embedding_channels)

        num_blocks = {32: 2, 64: 3, 128: 4, 256: 5}
        if img_size not in num_blocks:
            raise ValueError(f"Unsupported img_size: {img_size}")

        out_channels = base
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels + embedding_channels, out_channels,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        blocks = []
        for i in range(num_blocks[img_size]):
            blocks.append(ConvBlock(out_channels, out_channels * 2))
            out_channels *= 2
        self.blocks = nn.Sequential(*blocks)

        self.final = nn.Conv2d(out_channels, 1, kernel_size=4, stride=1, padding=0, bias=False)
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
        # images: (B, C, H, W), labels: (B,)
        h, w = images.size(-2), images.size(-1)
        labels = self.labels_embedding(labels)                  # (B, embedding_channels)
        labels = labels.view(-1, self.embedding_channels, 1, 1) # (B, embedding_channels, 1, 1)
        labels = labels.expand(-1, -1, h, w)                    # (B, embedding_channels, H, W)
        x = torch.cat([images, labels], dim=1)                  # (B, in_channels + embedding_channels, H, W)

        x = self.initial(x)
        x = self.blocks(x)
        logits = self.final(x).view(-1, 1)
        return logits


class ACDiscriminator(nn.Module):
    def __init__(self, img_size=32, in_channels=3, base=64, num_classes=10):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes

        num_blocks = {32: 2, 64: 3, 128: 4, 256: 5}
        if img_size not in num_blocks:
            raise ValueError(f"Unsupported img_size: {img_size}")

        out_channels = base
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        blocks = []
        for i in range(num_blocks[img_size]):
            blocks.append(ConvBlock(out_channels, out_channels * 2))
            out_channels = out_channels * 2
        self.blocks = nn.Sequential(*blocks)

        self.adv_head = nn.Conv2d(out_channels, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.aux_head = nn.Conv2d(out_channels, num_classes, kernel_size=4, stride=1, padding=0, bias=False)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, images):
        # Shared feature extraction
        x = self.initial(images)
        x = self.blocks(x)

        validity = self.adv_head(x).view(-1, 1)
        class_logits = self.aux_head(x).view(-1, self.num_classes)
        return validity, class_logits