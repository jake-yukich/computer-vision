import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64x16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 128x8x8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 128x4x4
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 3x32x32
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def add_noise(self, x, factor=0.2):
        noisy = x + factor * torch.randn_like(x)
        return torch.clamp(noisy, 0., 1.)
    
# -------------------------------------------------------------------------------------------------

def train(model, train_loader, epochs, learning_rate, device='mps'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            noisy_data = model.add_noise(data)
            
            optimizer.zero_grad()
            output = model(noisy_data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
