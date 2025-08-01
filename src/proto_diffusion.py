import torch
import torch.nn as nn
import torch.multiprocessing as mp
from icecream import ic
import numpy as np
from transformers import get_scheduler

from unet import ProtoUNet
from ddim import DDIMScheduler
from multidataset import MultiDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, LeaveOneOut
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


import os

from losses import LSD
from models import HRTFApproxNetwork_FIAE
from utils import replace_activation_function, plot_mag_Angle_vs_Freq
from configs import * # configs

output_dir = "./outputs/out_20240918_FIAE_500239/"

def init_model(device, deterministic=True):
    model = ProtoUNet(deterministic=deterministic).to(device)

    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

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

def compute_latent_stats(encoder, dataloader, device):
    all_latents = []

    encoder.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {data_kind: (batch[data_kind].to(device) if isinstance(batch[data_kind], torch.Tensor) else batch[data_kind]) for data_kind in batch}
            data_copy = {data_kind: batch[data_kind] for data_kind in batch}
            
            # Encode to latent
            latents = encoder.encode(data_copy)  # shape: (B, C, L)
            latents_batch = torch.cat(torch.split(latents["z"], 128, dim=2), dim=0).squeeze(1).permute(0, 2, 1)
            all_latents.append(latents_batch)

    # Stack all latent batches into (N, C, L)
    all_latents = torch.cat(all_latents, dim=0)  # shape: (2S, C, L)

    # Compute subject-wise mean and std
    mean = all_latents.mean(dim=0, keepdim=True)  # shape: (1, C, L)
    std = all_latents.std(dim=0, keepdim=True)

    # ic(all_latents.shape, mean.shape, std.shape)

    return mean, std

def train_step(model, scheduler, latents, condition_embeddings, criterion, condition_dropout_prob=0.1):
    # Sample random timesteps
    timesteps = torch.randint(0, scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device)
    # ic(timesteps)

    # Generate time embeddings
    time_embeddings = model.time_step_embedding(timesteps)

    # Generate random noise
    noise = torch.randn_like(latents)
    # ic(noise[0][0][0][0])

    # Add noise to the latents
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    # ic(noisy_latents.shape, latents.shape)

    # Randomly decide whether to use the condition
    use_condition = torch.rand(1).item() > condition_dropout_prob

    # ic(noise.shape, noisy_latents.shape, time_embeddings.shape, condition_embeddings.shape)
    # Predict the noise
    noise_pred = model(noisy_latents, time_embeddings, condition_embeddings if use_condition else None)
    # ic(noise_pred[0][0][0][0].item())

    # Compute loss
    loss = criterion(noise_pred, noise)

    return loss

def test_step(model, scheduler, latents, condition_embeddings, criterion):
    with torch.no_grad():
        # Sample random timesteps
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device)

        # Generate time embeddings
        time_embeddings = model.time_step_embedding(timesteps)
        # ic(time_embeddings.shape)

        # Generate random noise
        noise = torch.randn_like(latents)

        # Add noise to the latents
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        # ic(noisy_latents.shape, time_embeddings.shape, condition_embeddings.shape)

        # Single forward pass
        noise_pred = model(noisy_latents, time_embeddings, condition_embeddings)

        # Compute loss
        loss = criterion(noise_pred, noise)

    return loss

def test_generation(model, scheduler, condition_embeddings, guidance_scale=7.5, eta=0.0):
    # Generate random latents
    # (2, 64, 128): 1 left, 1 right
    device = condition_embeddings.device
    denoised_latents = torch.randn((1, 64, 128), device=device).repeat(2, 1, 1)
    # denoised_latents = torch.randn((2, 64, 128), device=device)
    # ic(torch.mean(denoised_latents), torch.std(denoised_latents))

    if condition_embeddings is not None:
        condition_embeddings = torch.cat([torch.zeros_like(condition_embeddings), condition_embeddings], dim=0)
    else:
        condition_embeddings = torch.zeros((2, condition_embeddings.shape[1]), device=device)

    timestep_tensor = scheduler.timesteps.to(device)
    time_embeddings_all = model.time_step_embedding(timestep_tensor)

    with torch.no_grad():
        # Iterate over timesteps in reverse order
        for i, t in enumerate(tqdm(scheduler.timesteps, desc="Denoising")):
            time_embeddings = time_embeddings_all[i].unsqueeze(0)

            # Single forward pass
            # (4, 64, 128): 2 left (conditioned and unconditioned), 2 right (conditioned and unconditioned)
            # noise_pred = model(denoised_latents.repeat_interleave(2, dim=0), time_embeddings, condition_embeddings)
            noise_pred = model(denoised_latents.repeat(2, 1, 1), time_embeddings, condition_embeddings)

            # Split predictions
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)

            # Apply classifier-free guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            denoised_latents = scheduler.step(noise_pred, t, denoised_latents, eta=eta)
            
            # ic(noise_pred[:, 0, 0], denoised_latents[:, 0, 0])

    # (2, 64, 128): 1 left, 1 right
    return denoised_latents


def train_1ep(model, train_dataloader, optimizer, criterion, encoder, scheduler, anthro_standardize, latent_standardize, device):
    model.train()

    train_loss = 0.0
    for batch in train_dataloader:
        batch = {data_kind: (batch[data_kind].to(device) if isinstance(batch[data_kind], torch.Tensor) else batch[data_kind]) for data_kind in batch}
        data_copy = {data_kind: batch[data_kind] for data_kind in batch}

        anthro_features = data_copy["AnthroFeatures"].reshape(-1, data_copy["AnthroFeatures"].size(-1))
        anthro_features_norm = (anthro_features - anthro_standardize["mean"]) / anthro_standardize["std"]

        latents = encoder.encode(data_copy)
        # latents_batch = (latents["z"].view(-1, latents["z"].size(-1)) - latent_standardize["mean"]) / latent_standardize["std"]
        latents_batch = torch.cat(torch.split(latents["z"], 128, dim=2), dim=0).squeeze(1).permute(0, 2, 1)
        latents_batch = (latents_batch - latent_standardize["mean"]) / latent_standardize["std"]

        optimizer.zero_grad()
        loss = train_step(model, scheduler, latents_batch, anthro_features_norm, criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()
    
    return train_loss / len(train_dataloader)

def valid_1ep(model, valid_dataloader, criterion, encoder, scheduler, anthro_standardize, latent_standardize, device):
    model.eval()

    valid_loss = 0.0
    with torch.no_grad():
        for batch in valid_dataloader:
            batch = {data_kind: (batch[data_kind].to(device) if isinstance(batch[data_kind], torch.Tensor) else batch[data_kind]) for data_kind in batch}
            data_copy = {data_kind: batch[data_kind] for data_kind in batch}

            anthro_features = data_copy["AnthroFeatures"].reshape(-1, data_copy["AnthroFeatures"].size(-1))
            anthro_features_norm = (anthro_features - anthro_standardize["mean"]) / anthro_standardize["std"]

            latents = encoder.encode(data_copy)
            # latents_batch = (latents["z"].view(-1, latents["z"].size(-1)) - latent_standardize["mean"]) / latent_standardize["std"]
            latents_batch = torch.cat(torch.split(latents["z"], 128, dim=2), dim=0).squeeze(1).permute(0, 2, 1)
            latents_batch = (latents_batch - latent_standardize["mean"]) / latent_standardize["std"]

            loss = test_step(model, scheduler, latents_batch, anthro_features_norm, criterion)
            valid_loss += loss.item()

    return valid_loss / len(valid_dataloader)

def test_model(train_dataset, test_dataset, encoder, training_configs):
    #========== set fixed seed ===========
    seed = training_configs["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #=====================================
    
    #========== set training configs ===========
    device = encoder.device
    num_epochs, save_file_dir, draw_figures, figure_dir, max_f = (
        training_configs[key]
        for key in [
            "num_epochs",
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

    model = init_model(device=device)
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    scheduler = DDIMScheduler(device=device, num_train_timesteps=1000)
    scheduler.set_timesteps(500)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_configs["learning_rate"], weight_decay=training_configs["weight_decay"])
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=training_configs["lr_warmup"], num_training_steps=num_epochs)
    criterion = nn.MSELoss()
    lsd_metric = LSD()

    encoder.config["Val_Standardize"] = train_dataset.getValStandardize(device)
    anthro_standardize = train_dataset.getAnthroStandardize(device)
    latent_mean, latent_std = compute_latent_stats(encoder, train_dataloader, device)
    latent_standardize = {
        "mean": latent_mean,
        "std": latent_std,
    }

    best_train_loss = float('inf')
    bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}]"
    t = tqdm(range(training_configs["num_test_epochs"]), desc='Training', bar_format=bar_format)
    for epoch in t:
        train_loss = train_1ep(model, train_dataloader, optimizer, criterion, encoder, scheduler, anthro_standardize, latent_standardize, device)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Save the best model
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            # if train_loss < 0.01:
            torch.save(model.state_dict(), save_file_dir + "best.pth")
            
        t.set_postfix_str(f"MSE={train_loss:.6f}>{best_train_loss:.6f}")

        # Adjust the learning rate
        lr_scheduler.step()

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
            # latents_batch = (latents["z"].view(-1, latents["z"].size(-1)) - latent_mean) / latent_std
            latents_batch = torch.cat(torch.split(latents["z"], 128, dim=2), dim=0).squeeze(1).permute(0, 2, 1)
            latents_batch = (latents_batch - latent_mean) / latent_std

            loss = test_step(model, scheduler, latents_batch, anthro_features_norm, criterion)
            latent_preds = test_generation(model, scheduler, anthro_features_norm, guidance_scale=4, eta=0.2)
            latent_preds = (latent_preds * latent_std) + latent_mean

            latent_stack = torch.cat(torch.chunk(latent_preds.permute(0, 2, 1).unsqueeze(1), 2, dim=0), dim=2)
            hrtf_preds = encoder.decode(data_copy, latent_stack)

            test_losses.append(loss.item())

            lsd_loss = lsd_metric(hrtf_preds['HRTF_mag'], batch['HRTF_mag'], dim=-1, data_kind='HRTF_mag', mean=True).item()
            lsd_losses.append(lsd_loss)
            recon = encoder.decode(data_copy, latents["z"])
            recon_loss = lsd_metric(hrtf_preds['HRTF_mag'], recon['HRTF_mag'], dim=-1, data_kind='HRTF_mag', mean=True).item()
            recon_losses.append(recon_loss)
            ic(recon_loss, lsd_loss)

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
    num_epochs, patience, anthro_standardize, save_file_dir = (
        training_configs[key]
        for key in [
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

    model = init_model(device=fold_device)

    scheduler = DDIMScheduler(device=fold_device, num_train_timesteps=1000)
    scheduler.set_timesteps(500)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_configs["learning_rate"], weight_decay=training_configs["weight_decay"])
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=training_configs["lr_warmup"], num_training_steps=training_configs["num_epochs"])
    criterion = nn.MSELoss()

    latent_mean, latent_std = compute_latent_stats(encoder, train_dataloader, fold_device)
    latent_standardize = {
        "mean": latent_mean,
        "std": latent_std,
    }

    best_val_loss = float('inf')
    best_model_path = save_file_dir + f"fold_{fold}.pth"

    curr_patience = patience
    bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed},{rate_fmt}{postfix}]"
    t = tqdm(range(num_epochs), desc=f'Fold {fold+1} (GPU {gpu_id})', bar_format=bar_format, position=gpu_id, colour="green" if gpu_id % 2 == 0 else "blue")
    for epoch in t:
        train_loss = train_1ep(model, train_dataloader, optimizer, criterion, encoder, scheduler, anthro_standardize, latent_standardize, fold_device)
        valid_loss = valid_1ep(model, valid_dataloader, criterion, encoder, scheduler, anthro_standardize, latent_standardize, fold_device)
        
        writer.add_scalars('Loss/MSE', {
            'train': train_loss,
            'valid': valid_loss,
        }, epoch)

        # Save the best model
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            curr_patience = patience
        else:
            curr_patience -= 1
            if curr_patience <= 0:
                break
        
        t.set_postfix_str(f"{best_val_loss:.6f}:{curr_patience}")
        
        # Adjust the learning rate
        lr_scheduler.step()
    
    print(f"Fold {fold+1} (GPU {gpu_id}) best loss: (MSE={best_val_loss:.6f})\n")
    results_queue.put({
        'fold': fold + 1,
        'val_loss': best_val_loss,
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

    avg_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)

    print(f"Cross-validation results: Avg Loss (MSE = {avg_val_loss:.6f} ± {std_val_loss:.6f}")

    # Find best fold based on validation LSD
    best_fold_idx = np.argmin(val_losses)
    best_fold = results[best_fold_idx]
    print(f"Best fold: {best_fold['fold']} with MSE = {best_fold['val_loss']:.6f}")
    
    # Save results summary to file
    results_summary = {
        'k_folds': training_configs["k_folds"] if training_configs["k_folds"] != train_len else 'LeaveOneOut',
        'avg_val_loss': float(avg_val_loss),
        'std_val_loss': float(std_val_loss),
        'best_fold': int(best_fold['fold']),
        'best_fold_val_loss': float(best_fold['val_loss']),
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

    train_db_names = ['HUTUBS']
    test_db_names = ['HUTUBS']
    encoders = load_nets('_'.join(train_db_names), devices=devices)

    # learning_rates = [1e-4, 3e-4, 5e-4, 1e-3]
    # weight_decays = [1e-4, 1e-3]
    
    save_prefix = output_dir + "personalization/latent_diffusion/" + '_'.join(test_db_names) + '/' + '_'.join(train_db_names) + '/'
    os.makedirs(save_prefix, exist_ok=True)
    os.makedirs(save_prefix + 'figures/', exist_ok=True)
    os.makedirs(save_prefix + 'models/', exist_ok=True)
    os.makedirs(save_prefix + 'logs/', exist_ok=True)

    training_configs = {
        "gpu_count": gpu_count,
        "seed": 0,
        "k_folds": 5,
        "batch_size": 1,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "num_epochs": 300,
        "patience": 30,
        "lr_warmup": 60,
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

    _, cv_results = cross_validation(train_dataset, encoders, training_configs)
    
    print(f'Setting num_epochs to {cv_results["avg_last_epoch"]} based on cross-validation results.')
    training_configs["num_test_epochs"] = cv_results["avg_last_epoch"]
    
    # training_configs["num_test_epochs"] = 158
    test_losses, lsd_losses, recon_losses = test_model(train_dataset, test_dataset, encoders[0], training_configs)
    print(f"Avg Test Loss: {np.mean(test_losses):.6f} ± {np.std(test_losses):.6f}")
    print(f"Avg Test LSD: {np.mean(lsd_losses):.4f} ± {np.std(lsd_losses):.4f}")
    print(f"Avg Test LSD (recon): {np.mean(recon_losses):.4f} ± {np.std(recon_losses):.4f}")