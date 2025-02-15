import torch
import torch.nn as nn
from icecream import ic
import numpy as np

from dataset import CHEDARDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys, os
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from losses import LSD
from utils import plot_mag_Angle_vs_Freq

output_dir = "./outputs/out_20240918_FIAE_500239/"

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.layers = nn.ModuleList().to('cuda')

        # Define the dimensions for each layer
        layer_dims = [input_dim, 64, 256, 1024, output_dim]

        # Create the layers dynamically
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]).to('cuda'))
            if i < len(layer_dims) - 2:  # No batch norm, activation, dropout for the output layer
                self.layers.append(nn.LayerNorm(layer_dims[i + 1]).to('cuda'))
                self.layers.append(nn.ReLU().to('cuda'))
                self.layers.append(nn.Dropout(0.3).to('cuda'))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        # Reshape the output to (2522, 2, 128)
        x = x.view(-1, 2522, 2, 128)
        return x

def predict(model, criterion, anthropometric_features, hrtf_mag, train=True, optimizer=None):

    # Predict the HRTF magnitude
    hrtf_pred = model(anthropometric_features)

    # ic(hrtf_pred.shape, hrtf_mag.shape)

    # Compute loss
    loss = criterion(hrtf_pred, hrtf_mag, dim=-1, data_kind='HRTF_mag', mean=True)

    if train:
        # ic(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Clip gradients
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    return loss.item(), hrtf_pred

if __name__ == '__main__':
    model = SimpleMLP(12, 2522 * 2 * 128)
    ic(model)

    #========== set random seed ===========
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #======================================

    for layer in model.modules():
        if hasattr(layer, 'weight'):
            nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
        if hasattr(layer, 'bias'):
            nn.init.normal_(layer.bias, mean=0.0, std=1e-3)

    def create_dataset(range, fft_length=256, max_frequency=24000):
        return CHEDARDataset(
            "../CHEDAR",
            range,
            filter_length=round(fft_length / 2),
            max_f=max_frequency
        )

    # Create training, validation, and test datasets
    train_dataset = create_dataset(range(0, 1003)) # 1003 subjects
    valid_dataset = create_dataset(range(1003, 1128)) # 125 subjects
    test_dataset = create_dataset(range(1128, 1253)) # 125 subjects
    
    batch_size = 1
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    num_train = len(train_dataloader)
    num_valid = len(valid_dataloader)
    num_test = len(test_dataloader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = LSD()

    anthro_standardize = train_dataset.Anthro_Standardize

    best_loss = 1000.0

    num_epochs = 3

    # Training and Validation
    for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0

        # Training
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            anthro_features_norm = (batch["AnthroFeatures"].to('cuda') - anthro_standardize["mean"]) / anthro_standardize["std"]
            loss, _ = predict(model, criterion, anthro_features_norm, batch["HRTF_mag"].to('cuda'), train=True, optimizer=optimizer)
            train_loss += loss

        # Validation
        model.eval()
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc=f"Validation {epoch + 1}/{num_epochs}"):
                anthro_features_norm = (batch["AnthroFeatures"].to('cuda') - anthro_standardize["mean"]) / anthro_standardize["std"]
                loss, _ = predict(model, criterion, anthro_features_norm, batch["HRTF_mag"].to('cuda'), train=False)
                valid_loss += loss

        # Compute the average loss
        avg_train_loss = train_loss / num_train
        avg_valid_loss = valid_loss / num_valid
        ic(avg_train_loss, avg_valid_loss)

        # Save the best model
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            torch.save(model.state_dict(), "best_model.pth")
            ic("Model saved.")

    # Testing
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            anthro_features_norm = (batch["AnthroFeatures"].to('cuda') - anthro_standardize["mean"]) / anthro_standardize["std"]
            loss, pred = predict(model, criterion, anthro_features_norm, batch["HRTF_mag"].to('cuda'), train=False)

            if total_loss == 0.0:
                ic(batch['HRTF_mag'].shape, pred.shape)
                plot_mag_Angle_vs_Freq(HRTF_mag=batch['HRTF_mag'][0,:,:,:], SrcPos=batch['SrcPos'][0,:,:].cpu(), fs=24000, figdir=output_dir + "figure/HRTF_mag_2D/", fname=output_dir + "figure/HRTF_mag_2D/basic_HRTF_Mag_orig")
                plot_mag_Angle_vs_Freq(HRTF_mag=pred[0,:,:,:], SrcPos=batch['SrcPos'][0,:,:].cpu(), fs=24000, figdir=output_dir + "figure/HRTF_mag_2D/", fname=output_dir + "figure/HRTF_mag_2D/basic_HRTF_Mag_latent")

            total_loss += loss
    
    avg_loss = total_loss / num_test
    ic(avg_loss)

    print("Done.")