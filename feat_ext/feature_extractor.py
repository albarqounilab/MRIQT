import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG3DFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1):
        """
        3D VGG-like feature extractor.
        Args:
            in_channels (int): Number of input channels.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            # nn.Dropout3d(0.2),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            # nn.Dropout3d(0.2),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            # nn.Dropout3d(0.2),
            nn.MaxPool3d(2),
            nn.Conv3d(128, 256, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            # nn.Dropout3d(0.2),
            nn.MaxPool3d(2)
        )
    def forward(self, x):
        feats = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                feats.append(x)
        return feats  # List of feature maps

class VGG3DAutoencoder(nn.Module):
    def __init__(self, in_channels=1):
        """
        3D VGG-like autoencoder used to train the feature extractor.
        Args:
            in_channels (int): Number of input channels.
        """
        super().__init__()
        self.encoder = VGG3DFeatureExtractor(in_channels)
        self.decoder = nn.Sequential(
            nn.Conv3d(256, 128, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout3d(0.2),
            nn.Conv3d(128, 64, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout3d(0.2),
            nn.Conv3d(64, 32, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout3d(0.2),
            nn.Conv3d(32, 1, 3, padding=1)
        )
    def forward(self, x):
        feats = self.encoder(x)
        out = self.decoder(feats[-1])
        return out
