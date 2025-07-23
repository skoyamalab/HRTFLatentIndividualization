import torch
import torch.nn as nn
import torch.multiprocessing as mp
from icecream import ic
import numpy as np
from multidataset import MultiDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, LeaveOneOut
from torch.utils.tensorboard import SummaryWriter # tensorboard
from tqdm import tqdm


import os

from losses import LSD
from models import HRTFApproxNetwork_FIAE, FourierFeatureMapping
from utils import replace_activation_function, plot_mag_Angle_vs_Freq
from configs import * # configs

output_dir = "./outputs/out_20240918_FIAE_500239/"

# --- Helper FiLM Layer ---
class FiLMLayer(nn.Module):
    """
    Applies Feature-wise Linear Modulation.
    Uses a conditioning embedding to predict scale and shift parameters,
    which are then applied to the input feature map.
    """
    def __init__(self, feature_channels: int, embedding_dim: int):
        """
        Args:
            feature_channels: Number of channels in the input feature map (C)
                              that will be modulated.
            embedding_dim: Dimension of the conditioning embedding (EmbDim).
        """
        super().__init__()
        # MLP to predict scale (gamma) and shift (beta)
        # Takes the conditioning embedding and outputs 2*C parameters
        self.generator = nn.Linear(embedding_dim, 2 * feature_channels)
        # Initialize the generator's final layer weights/bias to zero
        # So initially gamma=1, beta=0 (identity transform)
        nn.init.zeros_(self.generator.weight)
        nn.init.zeros_(self.generator.bias)

    def forward(self, x: torch.Tensor, cond_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor to be modulated (e.g., shape [BatchSize, Channels]).
               In this MLP case, BatchSize corresponds to 2*L.
            cond_embedding: Conditioning embedding tensor (e.g., shape [BatchSize, EmbDim]).
                            Should have the same BatchSize as x.

        Returns:
            Modulated tensor with the same shape as x.
        """
        # Generate scale (gamma) and shift (beta) parameters
        gamma_beta = self.generator(cond_embedding) # Shape: [BatchSize, 2 * Channels]

        # Split into gamma and beta
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1) # Shape: [BatchSize, Channels] each

        # Apply FiLM: output = x * gamma + beta
        # Note: We initialize generator bias to 0, so beta starts at 0.
        # We initialize generator weight to 0, so gamma starts at 0.
        # To make gamma start around 1 (identity scaling), we add 1.
        return x * (gamma + 1) + beta

# --- Latent MLP with FiLM ---
class LatentMLP_FiLM(nn.Module):
    def __init__(self, anthro_dim=23, num_freq_features=16, subj_emb_dim=64, output_dim=64, dropout_rate=0.2):
        """
        MLP to predict latent vectors, conditioned on anthropometry and frequency
        using FiLM layers for frequency conditioning.

        Args:
            anthro_dim: Dimension of the input anthropometric features per ear.
            num_freq_features: Number of base features for Fourier Feature Mapping.
            subj_emb_dim: Dimension of the intermediate subject/ear embedding.
            output_dim: Dimension of the output latent vector (d).
            dropout_rate: Dropout probability.
        """
        super().__init__()
        self.output_dim = output_dim
        self.freq_emb_dim = 2 * num_freq_features

        # 1. Frequency Feature Mapping
        self.freq_ids = (torch.arange(128+1)[1:]/128).unsqueeze(-1) # (128, 1)
        self.ffm_freq = FourierFeatureMapping(
            num_features=num_freq_features,
            dim_data=1,
            trainable=True
        )

        # 2. Anthropometry Embedding MLP
        # Processes the per-ear anthropometry features into a subject embedding
        self.anthro_mlp = nn.Sequential(
            nn.Linear(anthro_dim, subj_emb_dim * 2),
            nn.LayerNorm(subj_emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(subj_emb_dim * 2, subj_emb_dim)
            # Output shape: (2, subj_emb_dim)
        )

        # 3. Main MLP Pathway (processes subject embedding)
        # We will insert FiLM layers to modulate activations based on frequency
        hidden_dims = [128, 256, 128] # Define hidden dimensions for the main path

        # Input layer for the main path
        self.main_linear_in = nn.Linear(subj_emb_dim, hidden_dims[0])
        self.main_norm_in = nn.LayerNorm(hidden_dims[0])
        self.main_act_in = nn.ReLU()
        self.main_dropout_in = nn.Dropout(dropout_rate)
        # FiLM layer to apply after the first activation
        self.film1 = FiLMLayer(feature_channels=hidden_dims[0], embedding_dim=self.freq_emb_dim)

        # Hidden layers
        self.main_hidden_layers = nn.ModuleList()
        self.film_layers = nn.ModuleList() # Store FiLM layers separately
        current_dim = hidden_dims[0]
        for i, hidden_dim in enumerate(hidden_dims[1:]):
            self.main_hidden_layers.append(nn.Linear(current_dim, hidden_dim))
            self.main_hidden_layers.append(nn.LayerNorm(hidden_dim))
            self.main_hidden_layers.append(nn.ReLU())
            self.main_hidden_layers.append(nn.Dropout(dropout_rate))
            # Add a FiLM layer corresponding to this hidden block
            self.film_layers.append(FiLMLayer(feature_channels=hidden_dim, embedding_dim=self.freq_emb_dim))
            current_dim = hidden_dim

        # Output layer
        self.main_linear_out = nn.Linear(current_dim, output_dim)

    def forward(self, x_in):
        """
        Args:
            x_in: A tuple containing:
                  - freq_ids: Tensor of frequency IDs, shape (L=128, 1)
                  - anthro_features: Tensor of normalized anthropometric features, shape (2, anthro_dim)

        Returns:
            Predicted latent vectors, shape (2L, output_dim)
        """
        anthro_features = x_in
        self.freq_ids = self.freq_ids.to(anthro_features.device)

        # --- Prepare Embeddings ---
        # Calculate frequency embeddings for L bins
        freq_embs = self.ffm_freq(self.freq_ids) # Shape: (L, freq_emb_dim)
        # Repeat/stack for left and right ears (simple concatenation)
        freq_embs_repeated = torch.cat([freq_embs, freq_embs], dim=0) # Shape: (2L, freq_emb_dim)

        # Calculate subject/ear embeddings
        subj_emb = self.anthro_mlp(anthro_features) # Shape: (2, subj_emb_dim)
        # Repeat subject embedding for each frequency bin
        # repeat_interleave(L, dim=0) repeats [e1, e2] -> [e1]*L + [e2]*L
        subj_emb_repeated = subj_emb.repeat_interleave(self.freq_ids.shape[0], dim=0) # Shape: (2L, subj_emb_dim)

        # --- Pass through Main MLP with FiLM ---
        # Input block
        # ic(subj_emb_repeated.device, freq_embs_repeated.device)
        h = self.main_linear_in(subj_emb_repeated)
        h = self.main_norm_in(h)
        h = self.main_act_in(h)
        h = self.main_dropout_in(h)
        # Apply first FiLM modulation
        h = self.film1(h, freq_embs_repeated)

        # Hidden blocks
        film_layer_idx = 0
        for i in range(0, len(self.main_hidden_layers), 4): # Process in chunks of Linear, Norm, Act, Dropout
            h = self.main_hidden_layers[i](h)   # Linear
            h = self.main_hidden_layers[i+1](h) # LayerNorm
            h = self.main_hidden_layers[i+2](h) # ReLU
            h = self.main_hidden_layers[i+3](h) # Dropout
            # Apply corresponding FiLM modulation
            h = self.film_layers[film_layer_idx](h, freq_embs_repeated)
            film_layer_idx += 1

        # Output layer
        output_latents = self.main_linear_out(h) # Shape: (2L, output_dim)

        return output_latents

class LatentMLP(nn.Module):
    def __init__(self, anthro_dim=23, num_freq_features=16, output_dim=64, dropout_rate=0.3):
        super().__init__()

        self.freq_ids = (torch.arange(128+1)[1:]/128).unsqueeze(-1) # (128, 1)
        self.ffm_freq = FourierFeatureMapping(
            num_features=num_freq_features,
            dim_data=1,
            trainable=True
        )

        self.layers = nn.ModuleList()
        layer_dims = [anthro_dim + 2 * num_freq_features, 128, 512, 128, output_dim]

        # Create the layers dynamically
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:  # No batch norm, activation, dropout for the output layer
                self.layers.append(nn.LayerNorm(layer_dims[i + 1]))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout_rate))

    def forward(self, x):
        # ic(freq_embs.shape, anthro_features.shape)
        freq_embs = self.ffm_freq(self.freq_ids.to(x.device)).repeat(2, 1)  # (2L, 2 * num_freq_features)
        anthro_features = x.repeat(1, 128).reshape(-1, x.size(-1)) # (2L, 23)
        x = torch.cat([freq_embs, anthro_features], dim=-1) # (S, 2 * num_freq_features + anthro_dim = 32 + 23)

        # ic(freq_embs.shape, anthro_features.shape, x.shape)
        for layer in self.layers:
            x = layer(x)
        
        return x

def compute_latent_stats(encoder, dataloader, device):
    all_latents = []

    encoder.eval()
    with torch.no_grad():
        # for batch in tqdm(dataloader, desc="Encoding latents"):
        for batch in dataloader:
            batch = {data_kind: (batch[data_kind].to(device) if isinstance(batch[data_kind], torch.Tensor) else batch[data_kind]) for data_kind in batch}
            data_copy = {data_kind: batch[data_kind] for data_kind in batch}
            
            latents = encoder.encode(data_copy)  # shape: (B, 1, 2L, d)
            latents_batch = torch.cat(torch.split(latents["z"], 128, dim=2), dim=0)
            all_latents.append(latents_batch)

    all_latents = torch.cat(all_latents, dim=0)  # shape: (2S, 1, L, d)

    # Compute subject-wise mean and std
    mean = all_latents.mean(dim=0, keepdim=True).view(-1, latents_batch.size(-1)).repeat(2, 1)  # shape: (2L, d)
    std = all_latents.std(dim=0, keepdim=True).view(-1, latents_batch.size(-1)).repeat(2, 1)  # shape: (2L, d)

    # ic(all_latents.shape, mean.shape, std.shape)
    return mean, std

def init_model(dropout_rate, device):
    # model = LatentMLP_FiLM(dropout_rate=dropout_rate).to(device)
    model = LatentMLP(dropout_rate=dropout_rate).to(device)

    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    return model

def load_nets(model_name, devices):
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

    nets = [HRTFApproxNetwork_FIAE(config=config, device=device) for device in devices]
    for net in nets:
        net.load_from_file(output_dir + f'hrtf_approx_network.best_{model_name}.net')
        replace_activation_function(net, act_func_new=eval(config["activation_function"]))
    return nets

def train_1ep(model, train_dataloader, optimizer, criterion, encoder, anthro_standardize, latent_standardize, device):
    model.train()

    train_loss = 0.0
    for batch in train_dataloader:
        batch = {data_kind: (batch[data_kind].to(device) if isinstance(batch[data_kind], torch.Tensor) else batch[data_kind]) for data_kind in batch}
        data_copy = {data_kind: batch[data_kind] for data_kind in batch}

        anthro_features = data_copy["AnthroFeatures"].reshape(-1, data_copy["AnthroFeatures"].size(-1))
        anthro_features_norm = (anthro_features - anthro_standardize["mean"]) / anthro_standardize["std"]

        latents = encoder.encode(data_copy)
        latents_batch = (latents["z"].view(-1, latents["z"].size(-1)) - latent_standardize["mean"]) / latent_standardize["std"]

        optimizer.zero_grad()
        latent_preds = model(anthro_features_norm)
        loss = criterion(latent_preds, latents_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    return train_loss / len(train_dataloader)

def valid_1ep(model, valid_dataloader, criterion, encoder, anthro_standardize, latent_standardize, lsd_metric, device):
    model.eval()

    valid_loss, lsd_loss = 0.0, 0.0
    with torch.no_grad():
        for batch in valid_dataloader:
            batch = {data_kind: (batch[data_kind].to(device) if isinstance(batch[data_kind], torch.Tensor) else batch[data_kind]) for data_kind in batch}
            data_copy = {data_kind: batch[data_kind] for data_kind in batch}

            anthro_features = data_copy["AnthroFeatures"].reshape(-1, data_copy["AnthroFeatures"].size(-1))
            anthro_features_norm = (anthro_features - anthro_standardize["mean"]) / anthro_standardize["std"]

            latents = encoder.encode(data_copy)
            latents_batch = (latents["z"].view(-1, latents["z"].size(-1)) - latent_standardize["mean"]) / latent_standardize["std"]

            latent_preds = model(anthro_features_norm)
            loss = criterion(latent_preds, latents_batch)
            valid_loss += loss.item()
            
            latent_preds = (latent_preds * latent_standardize["std"]) + latent_standardize["mean"]
            latent_stack = latent_preds.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 2L, d)

            hrtf_preds = encoder.decode(data_copy, latent_stack)
            lsd_loss += lsd_metric(hrtf_preds["HRTF_mag"], batch["HRTF_mag"], dim=-1, data_kind='HRTF_mag', mean=True).item()

            # recon = encoder.decode(data_copy, latents["z"])
            # recon_loss += lsd_metric(hrtf_preds["HRTF_mag"], recon["HRTF_mag"], dim=-1, data_kind='HRTF_mag', mean=True).item()

    return valid_loss / len(valid_dataloader), lsd_loss / len(valid_dataloader)

def test_model(train_dataset, test_dataset, encoder, training_configs):
    #========== set fixed seed ===========
    seed = training_configs["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #=====================================
    
    #========== set training configs ===========
    device = encoder.device
    dropout_rate, num_epochs, patience, save_file_dir, draw_figures, figure_dir, max_f = (
        training_configs[key]
        for key in [
            "dropout_rate",
            "num_epochs",
            "patience",
            "save_file_dir",
            "draw_figures",
            "figure_dir",
            "max_f",
        ]
    )
    writer = SummaryWriter(training_configs["logs_dir"] + f"test/")
    #===========================================
    train_dataloader = DataLoader(train_dataset, batch_size=training_configs["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=training_configs["batch_size"], shuffle=False)

    model = init_model(dropout_rate=dropout_rate, device=device)
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Number of encoder trainable parameters: {sum(p.numel() for p in encoder.parameters() if p.requires_grad)}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_configs["learning_rate"], weight_decay=training_configs["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=training_configs["lr_factor"], patience=training_configs["lr_patience"], min_lr=training_configs["lr_min"])
    criterion = nn.MSELoss()
    lsd_metric = LSD()

    encoder.config["Val_Standardize"] = train_dataset.getValStandardize(device)
    anthro_standardize = train_dataset.getAnthroStandardize(device)
    latent_mean, latent_std = compute_latent_stats(encoder, train_dataloader, device)
    latent_standardize = {
        "mean": latent_mean,
        "std": latent_std,
    }

    curr_patience = patience
    best_train_loss = float('inf')
    bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}]"
    t = tqdm(range(num_epochs), desc='Training', bar_format=bar_format)
    for epoch in t:
        train_loss = train_1ep(model, train_dataloader, optimizer, criterion, encoder, anthro_standardize, latent_standardize, device)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Save the best model
        if train_loss < best_train_loss:
            curr_patience = patience 
            best_train_loss = train_loss
            # if train_loss < 0.01:
            torch.save(model.state_dict(), save_file_dir + "best.pth")
        else:
            curr_patience -= 1
            if curr_patience <= 0:
                break
            
        t.set_postfix_str(f"MSE={train_loss:.6f}>{best_train_loss:.6f}:{patience}")

        # Adjust the learning rate
        lr_scheduler.step(train_loss)

    #========== set fixed seed ===========
    seed = training_configs["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #=====================================

    model.load_state_dict(torch.load(save_file_dir + "best.pth"))
    model.eval()
    test_losses, lsd_losses, recon_losses = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
            batch = {data_kind: (batch[data_kind].to(device) if isinstance(batch[data_kind], torch.Tensor) else batch[data_kind]) for data_kind in batch}
            data_copy = {data_kind: batch[data_kind] for data_kind in batch}

            anthro_features = data_copy["AnthroFeatures"].reshape(-1, data_copy["AnthroFeatures"].size(-1))
            anthro_features_norm = (anthro_features - anthro_standardize["mean"]) / anthro_standardize["std"]

            latents = encoder.encode(data_copy)
            latents_batch = (latents["z"].view(-1, latents["z"].size(-1)) - latent_mean) / latent_std

            latent_preds = model(anthro_features_norm)
            loss = criterion(latent_preds, latents_batch)
            test_losses.append(loss.item())

            latent_preds = (latent_preds * latent_std) + latent_mean
            latent_stack = latent_preds.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 2L, d)
            hrtf_preds = encoder.decode(data_copy, latent_stack)

            loss = lsd_metric(hrtf_preds["HRTF_mag"], batch["HRTF_mag"], dim=-1, data_kind='HRTF_mag', mean=True).item()
            lsd_losses.append(loss)

            recon = encoder.decode(data_copy, latents["z"])
            recon_loss = lsd_metric(hrtf_preds['HRTF_mag'], recon['HRTF_mag'], dim=-1, data_kind='HRTF_mag', mean=True).item()
            recon_losses.append(recon_loss)

            if draw_figures:
                plot_mag_Angle_vs_Freq(HRTF_mag=batch['HRTF_mag'][0,:,:,:], SrcPos=batch['SrcPos'][0,:,:].cpu(), vmin=-25, vmax=5, fs=max_f, figdir=figure_dir, fname=figure_dir + f"orig_{i}")
                plot_mag_Angle_vs_Freq(HRTF_mag=recon['HRTF_mag'][0,:,:,:], SrcPos=batch['SrcPos'][0,:,:].cpu(), vmin=-25, vmax=5, fs=max_f, figdir=figure_dir, fname=figure_dir + f"recon_{i}")
                plot_mag_Angle_vs_Freq(HRTF_mag=hrtf_preds['HRTF_mag'][0,:,:,:], SrcPos=batch['SrcPos'][0,:,:].cpu(), vmin=-25, vmax=5, fs=max_f, figdir=figure_dir, fname=figure_dir + f"pred_{i}")

    writer.flush()
    writer.close()

    return test_losses, lsd_losses, recon_losses

def train_fold(fold, gpu_id, train_subset, valid_subset, encoder, training_configs, results_queue):
    #========== set fixed seed ===========
    seed = training_configs["seed"] + fold
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #=====================================

    # Set the device for this process
    fold_device = encoder.device
    encoder.config["Val_Standardize"] = training_configs["Val_Standardize"]

    train_dataloader = DataLoader(train_subset, batch_size=training_configs["batch_size"], shuffle=True)
    valid_dataloader = DataLoader(valid_subset, batch_size=training_configs["batch_size"], shuffle=False)

    #========== set training configs ===========
    dropout_rate, num_epochs, patience, anthro_standardize, save_file_dir = (
        training_configs[key]
        for key in [
            "dropout_rate",
            "num_epochs",
            "patience",
            "anthro_standardize",
            "save_file_dir",
        ]
    )
    anthro_standardize["mean"] = anthro_standardize["mean"].to(fold_device)
    anthro_standardize["std"] = anthro_standardize["std"].to(fold_device)
    writer = SummaryWriter(training_configs["logs_dir"] + f"fold_{fold + 1}/")
    #===========================================

    model = init_model(dropout_rate=dropout_rate, device=fold_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_configs["learning_rate"], weight_decay=training_configs["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=training_configs["lr_factor"], patience=training_configs["lr_patience"], min_lr=training_configs["lr_min"])
    criterion = nn.MSELoss()
    lsd_metric = LSD()

    latent_mean, latent_std = compute_latent_stats(encoder, train_dataloader, fold_device)
    latent_standardize = {
        "mean": latent_mean,
        "std": latent_std,
    }

    best_val_loss, best_lsd_loss = float('inf'), float('inf')
    best_model_path = save_file_dir + f"fold_{fold}.pth"

    curr_patience = patience
    bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed},{rate_fmt}{postfix}]"
    t = tqdm(range(num_epochs), desc=f'Fold {fold+1} (GPU {gpu_id})', bar_format=bar_format, position=gpu_id, colour="green" if gpu_id % 2 == 0 else "blue")
    for epoch in t:
        train_loss = train_1ep(model, train_dataloader, optimizer, criterion, encoder, anthro_standardize, latent_standardize, fold_device)
        valid_loss, lsd_loss = valid_1ep(model, valid_dataloader, criterion, encoder, anthro_standardize, latent_standardize, lsd_metric, fold_device)
        
        writer.add_scalars('Loss/MSE', {
            'train': train_loss,
            'valid': valid_loss,
        }, epoch)
        writer.add_scalars('Loss/LSD', {
            'orig': lsd_loss,
        }, epoch)

        # Save the best model
        if valid_loss < best_val_loss:
            curr_patience = patience
            best_val_loss = valid_loss
            best_lsd_loss = lsd_loss
        else:
            curr_patience -= 1
            if curr_patience <= 0:
                break
        
        t.set_postfix_str(f"{best_val_loss:.6f}:{best_lsd_loss:.3f}:{curr_patience}")
        
        # Adjust the learning rate
        lr_scheduler.step(valid_loss)
    
    print(f"Fold {fold+1} (GPU {gpu_id}) best loss: (MSE={best_val_loss:.6f}:LSD={best_lsd_loss:.3f})\n")
    results_queue.put({
        'fold': fold + 1,
        'val_loss': best_val_loss,
        'lsd_loss': best_lsd_loss,
        'model_path': best_model_path,
        'last_epoch': epoch + 1,
        'gpu_id': gpu_id,
    })

    writer.flush()
    writer.close()

def cross_validation(train_dataset, encoders, training_configs):
    #========== set fixed seed ===========
    seed = training_configs["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #=====================================

    gpu_count = training_configs["gpu_count"]

    train_len = len(train_dataset)
    splits = KFold(n_splits=training_configs["k_folds"], shuffle=True, random_state=seed).split(range(train_len)) if training_configs["k_folds"] > 1 else LeaveOneOut().split(range(train_len))
    splits = list(splits)
    if training_configs["k_folds"] == 0:
        training_configs["k_folds"] = train_len

    mp.set_start_method('spawn', force=True)
    active_processes = {}
    results_queue = mp.Queue()
    results = []
    next_fold = 0

    for gpu_id in range(gpu_count):
        if next_fold < len(splits):
            encoder = encoders[gpu_id]
            train_idx, valid_idx = splits[next_fold]

            fold_configs = training_configs.copy()
            fold_configs["Val_Standardize"] = train_dataset.getValStandardize(encoder.device, indices=train_idx)
            fold_configs["anthro_standardize"] = train_dataset.getAnthroStandardize(encoder.device, indices=train_idx)

            train_subset = Subset(train_dataset, train_idx)
            valid_subset = Subset(train_dataset, valid_idx)

            p = mp.Process(
                target=train_fold,
                args=(next_fold, gpu_id, train_subset, valid_subset, encoder, fold_configs, results_queue)
            )
            p.start()
            active_processes[gpu_id] = (p, next_fold)
            # print(f"Fold {next_fold + 1}/{k_folds} started on GPU {gpu_id}.")
            next_fold += 1
    
    while active_processes:
        result = results_queue.get()
        completed_gpu = result.pop("gpu_id")
        results.append(result)

        p, fold = active_processes[completed_gpu]
        p.join()
        # print(f"Fold {fold + 1}/{k_folds} completed on GPU {completed_gpu}.")
        
        if next_fold < len(splits):
            encoder = encoders[completed_gpu]
            train_idx, valid_idx = splits[next_fold]

            fold_configs = training_configs.copy()
            fold_configs["Val_Standardize"] = train_dataset.getValStandardize(encoder.device, indices=train_idx)
            fold_configs["anthro_standardize"] = train_dataset.getAnthroStandardize(encoder.device, indices=train_idx)

            train_subset = Subset(train_dataset, train_idx)
            valid_subset = Subset(train_dataset, valid_idx)

            p = mp.Process(
                target=train_fold,
                args=(next_fold, completed_gpu, train_subset, valid_subset, encoder, fold_configs, results_queue)
            )
            active_processes[completed_gpu] = (p, next_fold)
            p.start()
            # print(f"Fold {next_fold + 1}/{k_folds} started on GPU {completed_gpu}.")
            next_fold += 1
        else:
            del active_processes[completed_gpu]

    results.sort(key=lambda x: x['fold'])

    val_losses = [result['val_loss'] for result in results]
    lsd_losses = [result['lsd_loss'] for result in results]

    avg_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)
    avg_lsd_loss = np.mean(lsd_losses)
    std_lsd_loss = np.std(lsd_losses)

    print(f"Cross-validation results: Avg Loss (MSE = {avg_val_loss:.6f} ± {std_val_loss:.6f}, LSD = {avg_lsd_loss:.6f} ± {std_lsd_loss:.6f})")

    # Find best fold based on validation LSD
    best_fold_idx = np.argmin(lsd_losses)
    best_fold = results[best_fold_idx]
    print(f"Best fold: {best_fold['fold']} with LSD = {best_fold['lsd_loss']:.6f}")
    
    # Save results summary to file
    results_summary = {
        'k_folds': training_configs["k_folds"] if training_configs["k_folds"] != train_len else 'LeaveOneOut',
        'avg_val_loss': float(avg_val_loss),
        'std_val_loss': float(std_val_loss),
        'avg_val_lsd': float(avg_lsd_loss),
        'std_val_lsd': float(std_lsd_loss),
        'best_fold': int(best_fold['fold']),
        'best_fold_val_loss': float(best_fold['val_loss']),
        'best_fold_val_lsd': float(best_fold['lsd_loss']),
        'avg_last_epoch': int(np.mean([result['last_epoch'] for result in results])),
    }
    
    import json
    with open(training_configs["figure_dir"] + "results_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=4)

    return results, results_summary

if __name__ == '__main__':
    # Check device availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPUs")
        devices = [torch.device(f"cuda:{i}") for i in range(gpu_count)]
    else:
        devices = [torch.device("cpu")]
        print("No GPU found, using CPU")

    train_db_names = ['CIPIC', 'HUTUBS']
    test_db_names = ['CIPIC']
    encoders = load_nets('_'.join(train_db_names), devices=devices)

    # learning_rates = [1e-4, 3e-4, 5e-4, 1e-3]
    # weight_decays = [1e-4, 1e-3]
    
    save_prefix = output_dir + "personalization/latent_MLP/" + '_'.join(test_db_names) + '/' + '_'.join(train_db_names) + '/'
    os.makedirs(save_prefix, exist_ok=True)
    os.makedirs(save_prefix + 'figures/', exist_ok=True)
    os.makedirs(save_prefix + 'models/', exist_ok=True)
    os.makedirs(save_prefix + 'logs/', exist_ok=True)

    training_configs = {
        "gpu_count": gpu_count,
        "seed": 0,
        "k_folds": 5,
        "batch_size": 1,
        "learning_rate": 7e-4,
        "weight_decay": 1e-3,
        "dropout_rate": 0.3,
        "num_epochs": 300,
        "patience": 15,
        "lr_patience": 5,
        "lr_factor": 0.8,
        "lr_min": 0,
        #========== metric logging ===========
        "max_f": 20000,
        "draw_figures": True,
        "figure_dir": save_prefix + 'figures/',
        "save_file_dir": save_prefix + "models/",
        "logs_dir": save_prefix + 'logs/',
        #=====================================
    }

    train_dataset = MultiDataset(train_db_names, phase='train')
    test_dataset = MultiDataset(test_db_names, phase='test')

    # _, cv_results = cross_validation(train_dataset, encoders, training_configs)
    
    # print(f'Setting num_epochs to {cv_results["avg_last_epoch"]} based on cross-validation results.')
    # training_configs["num_epochs"] = cv_results["avg_last_epoch"]
    
    # training_configs["num_epochs"] = 46
    test_losses, lsd_losses, recon_losses = test_model(train_dataset, test_dataset, encoders[0], training_configs)
    print(f"Avg Test Loss: {np.mean(test_losses):.6f} ± {np.std(test_losses):.6f}")
    print(f"Avg Test LSD: {np.mean(lsd_losses):.4f} ± {np.std(lsd_losses):.4f}")
    print(f"Avg Test LSD (recon): {np.mean(recon_losses):.4f} ± {np.std(recon_losses):.4f}")