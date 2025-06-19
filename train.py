import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import numpy as np
from models.autoencoder import ConvAutoencoder
from utils.video_utils import load_ucsd_dataset

class UCSDDataset(Dataset):
    def __init__(self, data_dir, size=(128, 128)):
        self.data_dir = data_dir
        self.size = size
        self.frames = load_ucsd_dataset(data_dir)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Updated data path to match your local setup
    data_path = "C:/Users/prave/Survillence_AI/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
    dataset = UCSDDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), 'autoencoder.pth')
    print("Model saved as 'autoencoder.pth'")

if __name__ == "__main__":
    if not os.path.exists(r'C:\Users\prave\Surveillance AI\UCSD_Anomaly_Dataset\UCSD_Anomaly_Dataset.v1p2'):
        print("Dataset not found. Please verify the path 'C:\\Users\\prave\\Surveillance AI\\UCSD_Anomaly_Dataset\\UCSD_Anomaly_Dataset.v1p2'.")
    else:
        train_model()