import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # [1, 128, 128] -> [16, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [16, 128, 128] -> [16, 64, 64]
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),  # [16, 64, 64] -> [8, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [8, 64, 64] -> [8, 32, 32]
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),  # [8, 32, 32] -> [4, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [4, 32, 32] -> [4, 16, 16]
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # [4, 16, 16] -> [8, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [8, 32, 32] -> [16, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [16, 64, 64] -> [1, 128, 128]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded