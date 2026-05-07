import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, embedding_dim=512, noise_dim=100):
        super(Generator, self).__init__()

        self.noise_dim = noise_dim

        self.model = nn.Sequential(
            nn.Linear(embedding_dim + noise_dim, 8*8*256),
            nn.ReLU(),
            nn.Unflatten(1, (256,8,8)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    def forward(self, audio_embedding):
        batch_size = audio_embedding.size(0)
        noise = torch.randn(batch_size, self.noise_dim, device=audio_embedding.device)
        combined_input = torch.cat([audio_embedding, noise], dim=1)
        return self.model(combined_input)
    
class Discriminator(nn.Module):
    def __init__(self, embedding_dim=512):
        super(Discriminator, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        
        self.fc_blocks = nn.Sequential(
            nn.Linear((8 * 8 * 256) + embedding_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, audio_embedding):
        img_features = self.conv_blocks(img)
        combined_features = torch.cat([img_features, audio_embedding], dim=1)
        return self.fc_blocks(combined_features)
