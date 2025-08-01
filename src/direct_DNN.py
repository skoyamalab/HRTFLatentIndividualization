import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
from multidataset import MultiDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, LeaveOneOut
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import os

from losses import LSD
from utils import plot_mag_Angle_vs_Freq
output_dir = "./outputs/out_20240918_FIAE_500239/"

class DirectDNN(nn.Module):
    def __init__(self, source_dim, input_dim=23, dropout_rate=0.4):
        super().__init__()
        self.source_dim = source_dim

        self.layers = nn.ModuleList()
        layer_dims = [input_dim, 64, 512, source_dim * 128]

        # Layer creation: Linear -> LayerNorm -> ReLU -> Dropout
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                self.layers.append(nn.LayerNorm(layer_dims[i + 1]))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        # Shape: (S=batch_size, B, LR=2, LL=128)
        return x.view(-1, self.source_dim, 2, 128)

def init_model(source_dim, dropout_rate, device):
    model = DirectDNN(source_dim, dropout_rate=dropout_rate).to(device)

    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    return model

def train_1ep(model, train_dataloader, optimizer, criterion, anthro_standardize, Val_Standardize, device):
    model.train()

    train_loss = 0.0
    for batch in train_dataloader:
        batch = {data_kind: (batch[data_kind].to(device) if isinstance(batch[data_kind], torch.Tensor) else batch[data_kind]) for data_kind in batch}
        data_copy = {data_kind: batch[data_kind] for data_kind in batch}

        anthro_features = data_copy["AnthroFeatures"].reshape(-1, data_copy["AnthroFeatures"].size(-1))
        anthro_features_norm = (anthro_features - anthro_standardize["mean"]) / anthro_standardize["std"]

        optimizer.zero_grad()
        hrtf_pred_norm = model(anthro_features_norm)
        hrtf_pred = (hrtf_pred_norm * Val_Standardize['std']) + Val_Standardize['mean']
        loss = criterion(hrtf_pred, batch["HRTF_mag"].to(device), dim=-1, data_kind='HRTF_mag', mean=True)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    return train_loss / len(train_dataloader)

def valid_1ep(model, valid_dataloader, criterion, anthro_standardize, Val_Standardize, device):
    model.eval()

    valid_loss = 0.0
    with torch.no_grad():
        for batch in valid_dataloader:
            batch = {data_kind: (batch[data_kind].to(device) if isinstance(batch[data_kind], torch.Tensor) else batch[data_kind]) for data_kind in batch}
            data_copy = {data_kind: batch[data_kind] for data_kind in batch}

            anthro_features = data_copy["AnthroFeatures"].reshape(-1, data_copy["AnthroFeatures"].size(-1))
            anthro_features_norm = (anthro_features - anthro_standardize["mean"]) / anthro_standardize["std"]

            hrtf_pred_norm = model(anthro_features_norm)
            hrtf_pred = (hrtf_pred_norm * Val_Standardize['std']) + Val_Standardize['mean']
            loss = criterion(hrtf_pred, batch["HRTF_mag"].to(device), dim=-1, data_kind='HRTF_mag', mean=True)
            valid_loss += loss.item()

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
    dropout_rate, num_epochs, save_file_dir, draw_figures, figure_dir, max_f = (
        training_configs[key]
        for key in [
            "dropout_rate",
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

    model = init_model(1250 if training_configs['db_name'] == 'CIPIC' else 440, dropout_rate=dropout_rate, device=device)
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_configs["learning_rate"], weight_decay=training_configs["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=training_configs["lr_factor"], patience=training_configs["lr_patience"], min_lr=training_configs["lr_min"])
    criterion = LSD()

    anthro_standardize = train_dataset.getAnthroStandardize(device)
    Val_Standardize = train_dataset.getValStandardize(device)[training_configs["db_name"]]["HRTF_mag"]

    best_train_loss = float('inf')
    bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}]"
    t = tqdm(range(num_epochs), desc='Training', bar_format=bar_format)
    for epoch in t:
        train_loss = train_1ep(model, train_dataloader, optimizer, criterion, anthro_standardize, Val_Standardize, device)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Save the best model
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            # if train_loss < 0.01:
            torch.save(model.state_dict(), save_file_dir + "best.pth")
            
        t.set_postfix_str(f"LSD={train_loss:.4f}>{best_train_loss:.4f}")

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
    test_losses = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
            batch = {data_kind: (batch[data_kind].to(device) if isinstance(batch[data_kind], torch.Tensor) else batch[data_kind]) for data_kind in batch}
            data_copy = {data_kind: batch[data_kind] for data_kind in batch}

            anthro_features = data_copy["AnthroFeatures"].reshape(-1, data_copy["AnthroFeatures"].size(-1))
            anthro_features_norm = (anthro_features - anthro_standardize["mean"]) / anthro_standardize["std"]

            hrtf_pred_norm = model(anthro_features_norm)
            hrtf_pred = (hrtf_pred_norm * Val_Standardize['std']) + Val_Standardize['mean']

            loss = criterion(hrtf_pred, batch["HRTF_mag"].to(device), dim=-1, data_kind='HRTF_mag', mean=True)
            test_losses.append(loss.item())

            if draw_figures:
                plot_mag_Angle_vs_Freq(HRTF_mag=batch['HRTF_mag'][0,:,:,:], SrcPos=batch['SrcPos'][0,:,:].cpu(), vmin=-25, vmax=5, fs=max_f, figdir=figure_dir, fname=figure_dir + f"orig_{i}")
                plot_mag_Angle_vs_Freq(HRTF_mag=hrtf_pred[0,:,:,:], SrcPos=batch['SrcPos'][0,:,:].cpu(), vmin=-25, vmax=5, fs=max_f, figdir=figure_dir, fname=figure_dir + f"pred_{i}")

    writer.flush()
    writer.close()

    return test_losses

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
    Val_Standardize = training_configs["Val_Standardize"]
    writer = SummaryWriter(training_configs["logs_dir"] + f"fold_{fold + 1}/")
    #===========================================

    model = init_model(1250 if training_configs['db_name'] == 'CIPIC' else 440, dropout_rate=dropout_rate, device=fold_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_configs["learning_rate"], weight_decay=training_configs["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=training_configs["lr_factor"], patience=training_configs["lr_patience"], min_lr=training_configs["lr_min"])
    criterion = LSD()

    best_val_loss = float('inf')
    best_model_path = save_file_dir + f"fold_{fold}.pth"

    curr_patience = patience
    bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed},{rate_fmt}{postfix}]"
    t = tqdm(range(num_epochs), desc=f'Fold {fold+1} (GPU {gpu_id})', bar_format=bar_format, position=gpu_id, colour="green" if gpu_id % 2 == 0 else "blue")
    for epoch in t:
        train_loss = train_1ep(model, train_dataloader, optimizer, criterion, anthro_standardize, Val_Standardize, fold_device)
        valid_loss = valid_1ep(model, valid_dataloader, criterion, anthro_standardize, Val_Standardize, fold_device)
        
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
        lr_scheduler.step(valid_loss)
    
    print(f"Fold {fold+1} (GPU {gpu_id}) best loss: (LSD={best_val_loss:.4f})\n")
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
            fold_configs["Val_Standardize"] = train_dataset.getValStandardize(f'cuda:{gpu_id}', indices=train_idx)[training_configs["db_name"]]["HRTF_mag"]
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
            fold_configs["Val_Standardize"] = train_dataset.getValStandardize(f'cuda:{completed_gpu}', indices=train_idx)[training_configs["db_name"]]["HRTF_mag"]
            fold_configs["anthro_standardize"] = train_dataset.getAnthroStandardize(f'cuda:{completed_gpu}', indices=train_idx)

            train_subset = Subset(train_dataset, train_idx)
            valid_subset = Subset(train_dataset, valid_idx)

            p = mp.Process(
                target=train_fold,
                args=(next_fold, completed_gpu, train_subset, valid_subset, fold_configs, results_queue)
            )
            active_processes[completed_gpu] = (p, next_fold)
            p.start()
            next_fold += 1
        else:
            del active_processes[completed_gpu]

    results.sort(key=lambda x: x['fold'])

    val_losses = [result['val_loss'] for result in results]

    avg_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)

    print(f"Cross-validation results: Avg Loss (LSD = {avg_val_loss:.4f} ± {std_val_loss:.4f})")

    # Find best fold based on validation LSD
    best_fold_idx = np.argmin(val_losses)
    best_fold = results[best_fold_idx]
    print(f"Best fold: {best_fold['fold']} with LSD = {best_fold['val_loss']:.4f}")
    
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

    db_name = 'CIPIC'

    # learning_rates = [1e-4, 3e-4, 5e-4, 1e-3]
    # weight_decays = [1e-4, 1e-3]
    
    save_prefix = output_dir + "personalization/basic_MLP/" + db_name + '/'
    os.makedirs(save_prefix, exist_ok=True)
    os.makedirs(save_prefix + 'figures/', exist_ok=True)
    os.makedirs(save_prefix + 'models/', exist_ok=True)
    os.makedirs(save_prefix + 'logs/', exist_ok=True)

    training_configs = {
        "gpu_count": gpu_count,
        "db_name": db_name,
        "seed": 0,
        "k_folds": 5,
        "batch_size": 1,
        "learning_rate": 3e-4,
        "weight_decay": 1e-2,
        "dropout_rate": 0.5,
        "num_epochs": 300,
        "patience": 5,
        "lr_patience": 15, # intentionally not triggered
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

    train_dataset = MultiDataset([db_name], phase='train')
    test_dataset = MultiDataset([db_name], phase='test')

    _, cv_results = cross_validation(train_dataset, training_configs)
    
    print(f'Setting num_epochs to {cv_results["avg_last_epoch"]} based on cross-validation results.')
    training_configs["num_epochs"] = cv_results["avg_last_epoch"]
    
    # training_configs["num_epochs"] = 0
    test_losses = test_model(train_dataset, test_dataset, training_configs)
    print(f"Avg Test LSD: {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}")