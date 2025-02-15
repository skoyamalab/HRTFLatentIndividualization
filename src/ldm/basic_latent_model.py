import torch
import torch.nn as nn
from icecream import ic
import numpy as np

from dataset import CHEDARDataset
from multidataset import MultiDataset, sub_indices
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # tensorboard
from tqdm import tqdm

import sys, os
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from losses import LSD
from models import HRTFApproxNetwork_FIAE
from utils import replace_activation_function, plot_mag_Angle_vs_Freq
from configs import * # configs

output_dir = "./outputs/out_20240918_FIAE_500239/"

class LatentMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.layers = nn.ModuleList().to('cuda')

        # Define the dimensions for each layer
        layer_dims = [input_dim, 128, 512, 2048, 8192, output_dim]

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
        
        # Reshape the output to (1, 1, 128, 64)
        # x = x.view(-1, 1, 1, 256, 64)
        x = x.view(-1, 1, 1, 128, 64)
        return x

def load_net():
    config = {
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-6,
        "mask_beginning": 0,
        "save_frequency": 250,
        "epochs": 500, 
        "num_gpus": 2,
        "model": "FIAE",
        "fft_length": 256,
        "use_freq_as_input": False,
        "timestamp": ""
    }
    config_name = f"config_FIAE_500239"
    config.update(eval(config_name))
    # ic(config)
    net = HRTFApproxNetwork_FIAE(config=config)

    net.load_from_file(output_dir + 'hrtf_approx_network.best.net')
    replace_activation_function(net, act_func_new=eval(config["activation_function"]))
    return config, net

def predict(model, criterion, anthropometric_features, latents):
    # Predict the latent variable
    # ic(anthropometric_features.shape)
    latent_pred = model(anthropometric_features).squeeze(0)

    # ic(latent_pred.shape, latents.shape)

    loss = criterion(latent_pred, latents)

    return loss, latent_pred

def split_anthro_features_norm(anthro_features, anthro_standardize, db_name):
    # ic(data_copy["AnthroFeatures"].shape)
    anthro_features = data_copy["AnthroFeatures"].squeeze(0)
    # ic(anthro_features.shape)
    anthro_L, anthro_R = torch.split(anthro_features, 1, dim=0)
    anthro_L, anthro_R = anthro_L.squeeze(0), anthro_R.squeeze(0)
    # ic(anthro_L.shape, anthro_R.shape)
    anthro_L_norm = (anthro_L - anthro_standardize[db_name]["mean"]) / anthro_standardize[db_name]["std"]
    anthro_R_norm = (anthro_R - anthro_standardize[db_name]["mean"]) / anthro_standardize[db_name]["std"]

    return anthro_L_norm, anthro_R_norm

if __name__ == '__main__':
    model = LatentMLP(18, 128 * 64)
    config, encoder = load_net()

    writer = SummaryWriter(output_dir + 'personalization/logs')
    
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

    # Create training, validation, and test datasets
    train_dataset = MultiDataset(sub_indices, phase="train") # 2214 subjects
    valid_dataset = MultiDataset(sub_indices, phase="valid") # 278 subjects
    test_dataset = MultiDataset(sub_indices, phase="test") # 270 subjects
    
    batch_size = 1
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    num_train = len(train_dataloader)
    num_valid = len(valid_dataloader)
    num_test = len(test_dataloader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.001)
    lsd = LSD()
    criterion = nn.HuberLoss()

    config["Val_Standardize"] = train_dataset.Val_Standardize
    anthro_standardize = train_dataset.Anthro_Standardize

    num_epochs = 20
    best_loss = 1000.0

    # Training and Validation
    for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        lsd_loss = 0.0

        # Training
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            data_copy = {}
            db_name = batch["db_name"][0]

            for data_kind in batch:
                # ic(data_kind, batch[data_kind])
                if data_kind != "db_name":
                    batch[data_kind] = batch[data_kind].to('cuda')
                data_copy[data_kind] = batch[data_kind]
                # ic(data_copy[data_kind].device)
            
            latents = encoder.encode(data_copy)
            # ic(latents["z"].shape)
            latents_L, latents_R = torch.split(latents["z"], 128, dim=2)

            anthro_L_norm, anthro_R_norm = split_anthro_features_norm(data_copy["AnthroFeatures"], anthro_standardize, db_name)

            # ic(latents["z"].shape)
            # ic(latents_L.shape, latents_R.shape)

            optimizer.zero_grad()
            loss_L, _ = predict(model, criterion, anthro_L_norm, latents_L)
            loss_R, _ = predict(model, criterion, anthro_R_norm, latents_R)
            combined_loss = loss_L + loss_R
            combined_loss.backward()
            optimizer.step()

            train_loss += combined_loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc=f"Validation {epoch + 1}/{num_epochs}"):
                data_copy = {}
                db_name = batch["db_name"][0]

                for data_kind in batch:
                    # ic(data_kind, batch[data_kind])
                    if data_kind != "db_name":
                        batch[data_kind] = batch[data_kind].to('cuda')
                    data_copy[data_kind] = batch[data_kind]
                    # ic(data_copy[data_kind].device)

                latents = encoder.encode(data_copy)
                recon = encoder.decode(data_copy, latents["z"])
                latents_L, latents_R = torch.split(latents["z"], 128, dim=2)
                anthro_L_norm, anthro_R_norm = split_anthro_features_norm(data_copy["AnthroFeatures"], anthro_standardize, db_name)
                loss_L, pred_latents_L = predict(model, criterion, anthro_L_norm, latents_L)
                loss_R, pred_latents_R = predict(model, criterion, anthro_R_norm, latents_R)
                combined_loss = loss_L + loss_R

                pred_latents = torch.cat((pred_latents_L, pred_latents_R), dim=2)
                # ic(pred_latents.shape)

                if valid_loss == 0.0:
                    # ic(latents["z"].shape, pred_latents.shape)
                    # ic(latents["z"], pred_latents)
                    recon_loss = lsd(recon["HRTF_mag"], batch["HRTF_mag"], dim=-1, data_kind='HRTF_mag', mean=True).item()
                    ic(recon_loss)
                    # ic(nn.MSELoss()(pred_latents, latents["z"]).item())
                    # ic(combined_loss.item() / 2)

                returns = encoder.decode(data_copy, pred_latents)

                # if valid_loss == 0.0:
                #     plot_mag_Angle_vs_Freq(HRTF_mag=batch['HRTF_mag'][0,:,:,:], SrcPos=batch['SrcPos'][0,:,:].cpu(), fs=config['max_frequency'], figdir=output_dir + "figure/HRTF_mag_2D/", fname=output_dir + "figure/HRTF_mag_2D/HRTF_Mag_orig")
                #     plot_mag_Angle_vs_Freq(HRTF_mag=returns['HRTF_mag'][0,:,:,:], SrcPos=batch['SrcPos'][0,:,:].cpu(), fs=config['max_frequency'], figdir=output_dir + "figure/HRTF_mag_2D/", fname=output_dir + "figure/HRTF_mag_2D/HRTF_Mag_latent")
                #     plot_mag_Angle_vs_Freq(HRTF_mag=recon['HRTF_mag'][0,:,:,:], SrcPos=batch['SrcPos'][0,:,:].cpu(), fs=config['max_frequency'], figdir=output_dir + "figure/HRTF_mag_2D/", fname=output_dir + "figure/HRTF_mag_2D/HRTF_Mag_recon")

                valid_loss += combined_loss.item()
                lsd_loss += lsd(returns["HRTF_mag"], batch["HRTF_mag"], dim=-1, data_kind='HRTF_mag', mean=True).item()

        # Compute the average loss
        avg_train_loss = train_loss / (num_train * 2)
        avg_valid_loss = valid_loss / (num_valid * 2)
        avg_lsd_loss = lsd_loss / num_valid
        ic(avg_train_loss, avg_valid_loss, avg_lsd_loss)

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/valid', avg_valid_loss, epoch)
        writer.add_scalar('Loss/lsd_valid', avg_lsd_loss, epoch)

        # Save the best model
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            torch.save(model.state_dict(), "best_model.pth")
            ic("Model saved.")
    
    # Testing
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    total_loss = 0.0
    lsd_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Testing"):
            data_copy = {}
            db_name = batch["db_name"][0]

            for data_kind in batch:
                # ic(data_kind, batch[data_kind])
                if data_kind != "db_name":
                    batch[data_kind] = batch[data_kind].to('cuda')
                data_copy[data_kind] = batch[data_kind]

            latents = encoder.encode(data_copy)
            latents_L, latents_R = torch.split(latents["z"], 128, dim=2)
            recon = encoder.decode(data_copy, latents["z"])
            anthro_L_norm, anthro_R_norm = split_anthro_features_norm(data_copy["AnthroFeatures"], anthro_standardize, db_name)
            loss_L, pred_latents_L = predict(model, criterion, anthro_L_norm, latents_L)
            loss_R, pred_latents_R = predict(model, criterion, anthro_R_norm, latents_R)
            combined_loss = loss_L + loss_R

            pred_latents = torch.cat((pred_latents_L, pred_latents_R), dim=2)
            # ic(pred_latents.shape)

            returns = encoder.decode(data_copy, pred_latents)

            if total_loss == 0.0:
                recon_loss = lsd(recon["HRTF_mag"], batch["HRTF_mag"], dim=-1, data_kind='HRTF_mag', mean=True).item()
                ic(recon_loss)
                plot_mag_Angle_vs_Freq(HRTF_mag=batch['HRTF_mag'][0,:,:,:], SrcPos=batch['SrcPos'][0,:,:].cpu(), fs=config['max_frequency'], figdir=output_dir + "figure/HRTF_mag_2D/", fname=output_dir + "figure/HRTF_mag_2D/HRTF_Mag_orig")
                plot_mag_Angle_vs_Freq(HRTF_mag=returns['HRTF_mag'][0,:,:,:], SrcPos=batch['SrcPos'][0,:,:].cpu(), fs=config['max_frequency'], figdir=output_dir + "figure/HRTF_mag_2D/", fname=output_dir + "figure/HRTF_mag_2D/HRTF_Mag_latent")
                plot_mag_Angle_vs_Freq(HRTF_mag=recon['HRTF_mag'][0,:,:,:], SrcPos=batch['SrcPos'][0,:,:].cpu(), fs=config['max_frequency'], figdir=output_dir + "figure/HRTF_mag_2D/", fname=output_dir + "figure/HRTF_mag_2D/HRTF_Mag_recon")

            total_loss += combined_loss.item()
            lsd_loss += lsd(returns["HRTF_mag"], batch["HRTF_mag"], dim=-1, data_kind='HRTF_mag', mean=True).item()
    
    # Compute the average loss
    avg_test_loss = total_loss / (num_test * 2)
    avg_lsd_loss = lsd_loss / num_test
    ic(avg_test_loss, avg_lsd_loss)

    writer.flush()
    print("Done.")