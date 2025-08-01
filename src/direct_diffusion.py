import torch
import torch.nn as nn
import torch.multiprocessing as mp
from icecream import ic
import numpy as np
from transformers import get_scheduler

from unet import DirectUNet
from ddim import DDIMScheduler
from multidataset import MultiDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, LeaveOneOut
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import os

from losses import LSD
from utils import plot_mag_Angle_vs_Freq

output_dir = "./outputs/out_20240918_FIAE_500239/"

def train_step(model, scheduler, HRTF_mag, condition_embeddings, criterion, condition_dropout_prob=0.1):
    # Sample random timesteps
    timesteps = torch.randint(0, scheduler.num_train_timesteps, (HRTF_mag.shape[0],), device=HRTF_mag.device)
    # ic(timesteps.shape)

    # Generate time embeddings
    time_embeddings = model.time_step_embedding(timesteps)
    # ic(time_embeddings.shape)

    # Generate random noise
    noise = torch.randn_like(HRTF_mag)

    # Add noise to the HRTF_mag
    noisy_HRTF = scheduler.add_noise(HRTF_mag, noise, timesteps)

    # ic(HRTF_mag.shape, noise.shape, timesteps.shape, noisy_HRTF.shape)

    # Randomly decide whether to use the condition
    use_condition = torch.rand(1).item() > condition_dropout_prob

    # ic(noisy_HRTF.shape, time_embeddings.shape, condition_embeddings.shape)

    # Predict the noise
    noise_pred = model(noisy_HRTF, time_embeddings, condition_embeddings if use_condition else None)

    # Compute loss
    loss = criterion(noise_pred, noise)

    return loss

def test_step(model, scheduler, HRTF_mag, condition_embeddings, criterion):
    with torch.no_grad():
        # Sample random timesteps
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (HRTF_mag.shape[0],), device=HRTF_mag.device)
        # ic(timesteps.shape)

        # Generate time embeddings
        time_embeddings = model.time_step_embedding(timesteps)
        # ic(time_embeddings.shape)

        # Generate random noise
        noise = torch.randn_like(HRTF_mag)

        # Add noise to the HRTF_mag
        noisy_HRTF = scheduler.add_noise(HRTF_mag, noise, timesteps)

        # Predict the noise
        noise_pred = model(noisy_HRTF, time_embeddings, condition_embeddings)

        # Compute loss
        loss = criterion(noise_pred, noise)

    return loss

def test_generation(model, scheduler, condition_embeddings, hrtf_mag_shape, guidance_scale=7.5, eta=0.5, chunk_size=128):
    #ic(condition_embeddings.shape)
    # Output shape: (S=batch_size=1, B=num_src_pos, LR=2, L=freq_bins=128)
    device = condition_embeddings.device
    S, B, _, L = hrtf_mag_shape
    N = S * 2 * B
    hrtf_input_shape = (S * 2 * B, 1, L)
    # ic(S, B, L, hrtf_input_shape)
    if condition_embeddings is not None:
        condition_embeddings = torch.cat([torch.zeros_like(condition_embeddings), condition_embeddings], dim=0)
    else:
        condition_embeddings = torch.zeros((2, condition_embeddings.shape[1]), device=device)

    timestep_tensor = scheduler.timesteps.to(device)
    time_embeddings_all = model.time_step_embedding(timestep_tensor)

    # ic(condition_embeddings.shape, timestep_tensor.shape, time_embeddings_all.shape)

    denoised_hrtf = torch.randn(hrtf_input_shape, device=device)

    with torch.no_grad():
        for i, t in enumerate(tqdm(scheduler.timesteps, desc="Denoising")):
            # ic(i, t, N)

            noise_pred = torch.zeros_like(denoised_hrtf, device=device)

            # (1, 256): broadcasted to (2 * chunk_size, 256)
            time_chunk = time_embeddings_all[i].unsqueeze(0)

            for j in range(0, N, chunk_size):
                start_idx = j
                end_idx = min(j + chunk_size, N)

                # (2 * chunk_size, 1, L)
                hrtf_chunk = denoised_hrtf[start_idx:end_idx].repeat(2, 1, 1)
                # (2 * chunk_size, 21)
                cond_chunk = condition_embeddings[torch.cat([torch.arange(start_idx, end_idx), torch.arange(start_idx + N, end_idx + N)])]
                # ic(time_chunk.shape, hrtf_chunk.shape, cond_chunk.shape)

                # Predict the noise
                noise_pred_chunk = model(hrtf_chunk, time_chunk, cond_chunk)

                noise_pred_uncond, noise_pred_cond = noise_pred_chunk.chunk(2, dim=0)

                noise_pred_chunk = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # ic(noise_pred_chunk.shape, hrtf_chunk.shape)
                noise_pred[start_idx:end_idx] = noise_pred_chunk
            
            denoised_hrtf = scheduler.step(noise_pred, t, denoised_hrtf, eta=eta)
    # ic(denoised_hrtf.shape)
    return denoised_hrtf.view(S, 2, B, L).permute(0, 2, 1, 3)


def init_model(device, deterministic=True):
    model = DirectUNet(deterministic=deterministic).to(device)

    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

    return model

def train_1ep(model, train_dataloader, optimizer, criterion, scheduler, anthro_standardize, Val_Standardize, training_configs, device):
    model.train()
    max_r = training_configs["max_r"]
    effective_batch_size = training_configs["effective_batch_size"]

    train_loss = 0.0
    for batch in train_dataloader:
        db_name = batch["db_name"][0]
        S, B, _, L = batch["HRTF_mag"].shape

        # Anthro Features: (S, 2, 23)
        # Source Position Cart: (S, B, 3)
        # HRTF Mag: (S, B, 2, L)
        anthro_features = batch["AnthroFeatures"].to(device)
        anthro_features_norm = (anthro_features - anthro_standardize["mean"]) / anthro_standardize["std"]

        srcpos_cart = batch["SrcPos_Cart"].to(device) / max_r
        hrtf_mag = batch["HRTF_mag"].to(device)
        hrtf_mag_norm = (hrtf_mag - Val_Standardize[db_name]['HRTF_mag']['mean']) / Val_Standardize[db_name]['HRTF_mag']['std']

        N = S * 2 * B
        all_hrtfs = hrtf_mag_norm.permute(0, 2, 1, 3).reshape(S * 2 * B, 1, L)

        expanded_srcpos = srcpos_cart.unsqueeze(1).expand(S, 2, B, 3).reshape(N, 3)
        expanded_anthro = anthro_features_norm.unsqueeze(2).expand(S, 2, B, 23).reshape(N, 23)
        cond_embs = torch.cat([expanded_srcpos, expanded_anthro], dim=-1)

        num_chunks = (S * B + effective_batch_size - 1) // effective_batch_size
        indices = torch.randperm(S * B)
        all_hrtfs_shuffled = all_hrtfs[indices]
        all_conditions_shuffled = cond_embs[indices]

        optimizer.zero_grad()
        batch_loss = 0.0
        for i in range(num_chunks):
            start_idx = i * effective_batch_size
            end_idx = min((i + 1) * effective_batch_size, S * B)
            hrtf_chunk = all_hrtfs_shuffled[start_idx:end_idx]
            condition_chunk = all_conditions_shuffled[start_idx:end_idx]
            # ic(condition_chunk.shape, hrtf_chunk.shape)
            loss = train_step(model, scheduler, hrtf_chunk, condition_chunk, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            batch_loss += loss.item()
        train_loss += batch_loss / num_chunks
        optimizer.step()
    
    return train_loss / len(train_dataloader)

def valid_1ep(model, valid_dataloader, criterion, scheduler, anthro_standardize, Val_Standardize, training_configs, device):
    model.eval()
    max_r = training_configs["max_r"]
    effective_batch_size = training_configs["effective_batch_size"]

    valid_loss = 0.0
    with torch.no_grad():
        for batch in valid_dataloader:
            db_name = batch["db_name"][0]
            S, B, _, L = batch["HRTF_mag"].shape

            # Anthro Features: (S, 2, 23)
            # Source Position Cart: (S, B, 3)
            # HRTF Mag: (S, B, 2, L)
            anthro_features = batch["AnthroFeatures"].to(device)
            anthro_features_norm = (anthro_features - anthro_standardize["mean"]) / anthro_standardize["std"]

            srcpos_cart = batch["SrcPos_Cart"].to(device) / max_r
            hrtf_mag = batch["HRTF_mag"].to(device)
            hrtf_mag_norm = (hrtf_mag - Val_Standardize[db_name]['HRTF_mag']['mean']) / Val_Standardize[db_name]['HRTF_mag']['std']

            N = S * 2 * B
            all_hrtfs = hrtf_mag_norm.permute(0, 2, 1, 3).reshape(S * 2 * B, 1, L)

            expanded_srcpos = srcpos_cart.unsqueeze(1).expand(S, 2, B, 3).reshape(N, 3)
            expanded_anthro = anthro_features_norm.unsqueeze(2).expand(S, 2, B, 23).reshape(N, 23)
            cond_embs = torch.cat([expanded_srcpos, expanded_anthro], dim=-1)

            num_chunks = (2 * B + effective_batch_size - 1) // effective_batch_size
            batch_loss = 0.0
            for i in range(num_chunks):
                start_idx = i * effective_batch_size
                end_idx = min((i + 1) * effective_batch_size, 2 * B)
                hrtf_chunk = all_hrtfs[start_idx:end_idx]
                condition_chunk = cond_embs[start_idx:end_idx]

                loss = test_step(model, scheduler, hrtf_chunk, condition_chunk, criterion)
                batch_loss += loss.item()
            valid_loss += batch_loss / num_chunks

    return valid_loss / len(valid_dataloader)

def test_model(train_dataset, test_dataset, training_configs):
    #========== set fixed seed ===========
    seed = training_configs["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #=====================================
    
    #========== set training configs ===========
    device = 'cuda:0'
    num_epochs, save_file_dir, draw_figures, figure_dir, max_f, max_r, effective_batch_size = (
        training_configs[key]
        for key in [
            "num_epochs",
            "save_file_dir",
            "draw_figures",
            "figure_dir",
            "max_f",
            "max_r",
            "effective_batch_size",
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

    Val_Standardize = train_dataset.getValStandardize(device)
    anthro_standardize = train_dataset.getAnthroStandardize(device)

    best_train_loss = float('inf')
    bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}]"
    t = tqdm(range(training_configs["num_test_epochs"]), desc='Training', bar_format=bar_format)
    curr_patience = training_configs["patience"]
    for epoch in t:
        train_loss = train_1ep(model, train_dataloader, optimizer, criterion, scheduler, anthro_standardize, Val_Standardize, training_configs, device)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Save the best model
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            curr_patience = training_configs["patience"]
            if best_train_loss < 0.03:
                torch.save(model.state_dict(), save_file_dir + "best.pth")
        else:
            curr_patience -= 1
            if curr_patience <= 0:
                break
            
        t.set_postfix_str(f"LSD={train_loss:.6f}>{best_train_loss:.6f}")

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
    test_losses, lsd_losses = [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
            db_name = batch["db_name"][0]

            anthro_features = batch["AnthroFeatures"].to(device)
            anthro_features_norm = (anthro_features - anthro_standardize["mean"]) / anthro_standardize["std"]

            srcpos_cart = batch["SrcPos_Cart"].to(device) / max_r
            hrtf_mag = batch["HRTF_mag"].to(device)
            hrtf_mag_norm = (hrtf_mag - Val_Standardize[db_name]['HRTF_mag']['mean']) / Val_Standardize[db_name]['HRTF_mag']['std']

            S, B, _, L = hrtf_mag_norm.shape
            N = S * 2 * B
            all_hrtfs = hrtf_mag_norm.permute(0, 2, 1, 3).reshape(S * 2 * B, 1, L)

            expanded_srcpos = srcpos_cart.unsqueeze(1).expand(S, 2, B, 3).reshape(N, 3)
            expanded_anthro = anthro_features_norm.unsqueeze(2).expand(S, 2, B, 23).reshape(N, 23)
            cond_embs = torch.cat([expanded_srcpos, expanded_anthro], dim=-1)

            num_chunks = (2 * B + effective_batch_size - 1) // effective_batch_size
            batch_loss = 0.0
            for j in range(num_chunks):
                start_idx = j * effective_batch_size
                end_idx = min((j + 1) * effective_batch_size, 2 * B)
                hrtf_chunk = all_hrtfs[start_idx:end_idx]
                condition_chunk = cond_embs[start_idx:end_idx]

                loss = test_step(model, scheduler, hrtf_chunk, condition_chunk, criterion)
                batch_loss += loss.item()
            test_losses.append(batch_loss / num_chunks)

            # Generate predicted HRTFs
            hrtf_preds_norm = test_generation(model, scheduler, cond_embs, hrtf_mag.shape, guidance_scale=2, eta=0.25)
            hrtf_preds = hrtf_preds_norm * Val_Standardize[db_name]['HRTF_mag']['std'] + Val_Standardize[db_name]['HRTF_mag']['mean']

            lsd_loss_val = lsd_metric(hrtf_preds, hrtf_mag, dim=-1, data_kind='HRTF_mag', mean=True).item()
            ic(lsd_loss_val)
            lsd_losses.append(lsd_loss_val)

            if draw_figures:
                plot_mag_Angle_vs_Freq(HRTF_mag=hrtf_mag[0,:,:,:], SrcPos=batch['SrcPos'][0,:,:].cpu(), vmin=-25, vmax=5, fs=max_f, figdir=figure_dir, fname=figure_dir + f"orig_{i}")
                plot_mag_Angle_vs_Freq(HRTF_mag=hrtf_preds[0,:,:,:], SrcPos=batch['SrcPos'][0,:,:].cpu(), vmin=-25, vmax=5, fs=max_f, figdir=figure_dir, fname=figure_dir + f"pred_{i}")

    writer.flush()
    writer.close()

    return test_losses, lsd_losses

def train_fold(fold, gpu_id, train_subset, valid_subset, training_configs, results_queue):
    #========== set fixed seed ===========
    seed = training_configs["seed"] + fold
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #=====================================

    # Set the device for this process
    fold_device = f'cuda:{gpu_id}'

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
    Val_Standardize = training_configs["Val_Standardize"]
    writer = SummaryWriter(training_configs["logs_dir"] + f"fold_{fold + 1}/")
    #===========================================

    model = init_model(device=fold_device)

    scheduler = DDIMScheduler(device=fold_device, num_train_timesteps=1000)
    scheduler.set_timesteps(500)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_configs["learning_rate"], weight_decay=training_configs["weight_decay"])
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=training_configs["lr_warmup"], num_training_steps=training_configs["num_epochs"])
    criterion = LSD()

    best_val_loss = float('inf')
    best_model_path = save_file_dir + f"fold_{fold}.pth"

    curr_patience = patience
    bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed},{rate_fmt}{postfix}]"
    t = tqdm(range(num_epochs), desc=f'Fold {fold+1} (GPU {gpu_id})', bar_format=bar_format, position=gpu_id, colour="green" if gpu_id % 2 == 0 else "blue")
    for epoch in t:
        train_loss = train_1ep(model, train_dataloader, optimizer, criterion, scheduler, anthro_standardize, Val_Standardize, training_configs, fold_device)
        valid_loss = valid_1ep(model, valid_dataloader, criterion, scheduler, anthro_standardize, Val_Standardize, training_configs, fold_device)
        
        writer.add_scalars('Loss/MSE', {
            'train': train_loss,
            'valid': valid_loss,
        }, epoch)

        # Save the best model
        if valid_loss < best_val_loss:
            curr_patience = patience
            best_val_loss = valid_loss
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

def cross_validation(train_dataset, training_configs):
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
            train_idx, valid_idx = splits[next_fold]

            fold_configs = training_configs.copy()
            fold_configs["Val_Standardize"] = train_dataset.getValStandardize(f'cuda:{gpu_id}', indices=train_idx)
            fold_configs["anthro_standardize"] = train_dataset.getAnthroStandardize(f'cuda:{gpu_id}', indices=train_idx)

            train_subset = Subset(train_dataset, train_idx)
            valid_subset = Subset(train_dataset, valid_idx)

            p = mp.Process(
                target=train_fold,
                args=(next_fold, gpu_id, train_subset, valid_subset, fold_configs, results_queue)
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
            train_idx, valid_idx = splits[next_fold]

            fold_configs = training_configs.copy()
            fold_configs["Val_Standardize"] = train_dataset.getValStandardize(f'cuda:{completed_gpu}', indices=train_idx)
            fold_configs["anthro_standardize"] = train_dataset.getAnthroStandardize(f'cuda:{completed_gpu}', indices=train_idx)

            train_subset = Subset(train_dataset, train_idx)
            valid_subset = Subset(train_dataset, valid_idx)

            p = mp.Process(
                target=train_fold,
                args=(next_fold, completed_gpu, train_subset, valid_subset, fold_configs, results_queue)
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

    print(f"Cross-validation results: Avg Loss (MSE = {avg_val_loss:.4f} ± {std_val_loss:.4f})")

    # Find best fold based on validation MSE
    best_fold_idx = np.argmin(val_losses)
    best_fold = results[best_fold_idx]
    print(f"Best fold: {best_fold['fold']} with MSE = {best_fold['val_loss']:.4f}")
    
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

    train_db_names = ['CIPIC']
    test_db_names = ['CIPIC']
    
    save_prefix = output_dir + "personalization/direct_diffusion/" + '_'.join(test_db_names) + '/' + '_'.join(train_db_names) + '/'
    os.makedirs(save_prefix, exist_ok=True)
    os.makedirs(save_prefix + 'figures/', exist_ok=True)
    os.makedirs(save_prefix + 'models/', exist_ok=True)
    os.makedirs(save_prefix + 'logs/', exist_ok=True)

    training_configs = {
        "gpu_count": gpu_count,
        "seed": 0,
        "k_folds": 5,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "num_epochs": 300,
        "patience": 25,
        "lr_warmup": 30,
        "max_f": 20000,
        "max_r": 1.5,
        "effective_batch_size": 256,
        #========== metric logging ===========
        "draw_figures": True,
        "figure_dir": save_prefix + 'figures/',
        "save_file_dir": save_prefix + "models/",
        "logs_dir": save_prefix + 'logs/',
        #=====================================
    }

    train_dataset = MultiDataset(train_db_names, phase='train')
    test_dataset = MultiDataset(test_db_names, phase='test')

    _, cv_results = cross_validation(train_dataset, training_configs)
    
    print(f'Setting num_epochs to {cv_results["avg_last_epoch"]} based on cross-validation results.')
    training_configs["num_test_epochs"] = cv_results["avg_last_epoch"]
    
    # training_configs["num_test_epochs"] = 0
    test_losses, lsd_losses = test_model(train_dataset, test_dataset, training_configs)
    print(f"Avg Test Loss: {np.mean(test_losses):.6f} ± {np.std(test_losses):.6f}")
    print(f"Avg Test LSD: {np.mean(lsd_losses):.4f} ± {np.std(lsd_losses):.4f}")