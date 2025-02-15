import os
import sys
import argparse
import datetime
import time
import pprint

import math

import numpy as np
import torch as th
import torchaudio as ta   
import torch.nn as nn
from torch.utils.data import DataLoader
from icecream import ic
ic.configureOutput(includeContext=True)

from torch.utils.tensorboard import SummaryWriter # tensorboard
from tqdm import tqdm

from src.ldm.dataset import CHEDARDataset
from src.ldm.conditional_unet import ConditionalUNet
from src.ldm.latent_diffusion import DDIMScheduler, train_step, test_step, test_generation

from src.models import HRTFApproxNetwork_FIAE
from src.losses import LSD
from src.configs import * # configs
from src.utils import replace_activation_function, plot_mag_Angle_vs_Freq, plotmaghrtf


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-datadir","--dataset_directory",
                        type=str,
                        default='../CHEDAR', # AE
                        # default='data/EvaluationGrid_', # SWFE
                        # default="data/mit_kemar_normal_pinna.sofa",
                        help="path to the train data")

    parser.add_argument("-a","--artifacts_directory",
                        type=str,
                        default="",
                        help="directory to write model files to")
    parser.add_argument("-n","--num_gpus",
                        type=int,
                        default=1,
                        help="number of GPUs used during training")
    parser.add_argument("-m","--model",
                        type=str,
                        default="AE",
                        help="AE or CAE or HCAE or AE_PINN or SWFE（球波動関数展開） or PINN")
    parser.add_argument("-c","--num_config",
                        type=int,
                        default=1,
                        help="idx of config you use")
    parser.add_argument("-load","--load", action="store_true", help="load model")
    parser.add_argument("-mf", "--model_file",
                    type=str,
                    default="",
                    help="model file containing the trained binaural network weights")
    parser.add_argument("-lininv", "--lininv", action="store_true", help="If declared, linear inverse problem is solved.")
    parser.add_argument("-freq_in", "--freq_in", action="store_true", help="If declared, frequency is used as input of DNN.")
    parser.add_argument("-test", "--test", action="store_true", help="")
    parser.add_argument("-dbg","--debug", action="store_true", help="debug mode")
    parser.add_argument("-opt","--optuna", action="store_true", help="optimize hyper parameter by optuna")
    parser.add_argument("-start_ep","--start_ep", type=int, default=0,
                        help="途中から学習を再開する場合")
    parser.add_argument("-save_freq","--save_freq", type=int, default=250,
                        help="何epごとにモデルを保存するか")
    parser.add_argument("-mkdir", "--flg_mkdir", action="store_true", help="")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    #========== set random seed ===========
    seed = 0
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    #======================================
    dt_now = datetime.datetime.now() + datetime.timedelta(hours=9)
    datestamp = dt_now.strftime('_%Y%m%d')
    timestamp = dt_now.strftime('_%m%d_%H%M')

    args = arg_parse()
    print("=====================")
    print("args: ")
    print(args)

    config = {
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-6,
        "mask_beginning": 0,
        "save_frequency": args.save_freq,
        "epochs": 500, 
        "num_gpus": args.num_gpus,
        "model": args.model,
        "fft_length": 256, 
        "use_freq_as_input": args.freq_in,
        "timestamp": ""
    }
    config_name = f"config_{args.model}_{args.num_config}"
    config.update(eval(config_name))
    net = HRTFApproxNetwork_FIAE(config=config)
        
    # print("=====================")
    # print("config: ")
    # pprint.pprint(config)
    # print("=====================")

    if args.load and args.model_file != '': # test or continue train
        net.load_from_file(args.model_file)
    
    if not config["activation_function"].startswith('nn.ReLU'):
        replace_activation_function(net, act_func_new=eval(config["activation_function"]))

    if args.artifacts_directory == "":
        config["artifacts_dir"] = "outputs/out"+ datestamp + '_' + config["model"]
    else:
        config["artifacts_dir"] = args.artifacts_directory
    print("artifacts_dir: " + config["artifacts_dir"])

    os.makedirs(config["artifacts_dir"], exist_ok=True)
    if args.flg_mkdir:
        sys.exit()

    if args.test:
        flg_train = False
        flg_save = True
        print("Test")
    else:
        flg_train = True
        flg_save = False
        print("Train")
    print("---------------------")

    writer = SummaryWriter(log_dir=config["artifacts_dir"]+'/personalization/logs')

    def create_dataset(config, dataset_type):
        return CHEDARDataset(
            "../CHEDAR",
            config["sub_index"]['CHEDAR'][dataset_type],
            filter_length=round(config["fft_length"] / 2),
            max_f=config["max_frequency"]
        )

    # Create training, validation, and test datasets
    train_dataset = create_dataset(config, 'train')
    valid_dataset = create_dataset(config, 'valid')
    test_dataset = create_dataset(config, 'test')
    
    batch_size = 1
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    lsd_loss = LSD()

    scheduler = DDIMScheduler()
    scheduler.set_timesteps(300)
    unet = ConditionalUNet()
    optimizer = th.optim.AdamW(unet.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.MSELoss()

    config["Val_Standardize"] = train_dataset.Val_Standardize
    anthro_standardize = train_dataset.Anthro_Standardize
    ptins = "debug_" if args.debug else ""

    print("test mode.")
    print("=====================")
    
    if args.test and args.load and args.model_file != '':
        net.load_from_file(args.model_file)
    else:
        net.load_from_file(config["artifacts_dir"]+'/hrtf_approx_network.best.net')
    # print(net)
    print("=====================")

    config["pln_smp"] = False
    config["pln_smp_paral"] = False

    print("---------------")
    print(f'All pts (2522)')
    print("----")
    
    config["num_pts"] = 2522

    total_loss = 0.0
    total_mse_loss = 0.0
    best_loss = 1000.0
    num_train = len(train_dataloader)
    num_valid = len(valid_dataloader)
    num_test = len(test_dataloader)

    patience = 10
    num_epochs = 0 # 50

    # Training and validation
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        total_loss = 0.0
        for i, data in enumerate(tqdm(train_dataloader, mininterval=15, maxinterval=15, desc="Training Progress")):
            data_copy = {}

            for data_kind in data:
                data[data_kind] = data[data_kind].to('cuda')
                data_copy[data_kind] = data[data_kind]

            latents = net.encode(data_copy)
            anthro_features_norm = (data_copy["AnthroFeatures"] - anthro_standardize["mean"]) / anthro_standardize["std"]

            loss = train_step(unet, scheduler, optimizer, latents["z"], anthro_features_norm, criterion)
            total_loss += loss

        train_loss = total_loss/num_train
        ic(train_loss)
        writer.add_scalar('Loss/train', train_loss, epoch)

        unet.eval()
        total_loss = 0.0
        total_lsd_loss = 0.0
        total_mse_loss = 0.0

        # num_valid = 50
        for i, data in enumerate(tqdm(valid_dataloader, mininterval=60, maxinterval=60, desc="Validation Progress")):
            # if i == num_valid:
            #     break

            data_copy = {}

            for data_kind in data:
                data[data_kind] = data[data_kind].to('cuda')
                data_copy[data_kind] = data[data_kind]

            latents = net.encode(data_copy)
            anthro_features_norm = (data_copy["AnthroFeatures"] - anthro_standardize["mean"]) / anthro_standardize["std"]

            # returns = net.decode(data_copy, latents["z"])

            loss = test_step(unet, scheduler, latents["z"], anthro_features_norm, criterion)

            denoised_latents = test_generation(unet, scheduler, data_copy["AnthroFeatures"])

            decoded_data = net.decode(data_copy, denoised_latents)

            lsd_loss_val = lsd_loss(decoded_data['HRTF_mag'], data['HRTF_mag'], dim=-1, data_kind='HRTF_mag', mean=True).item()

            total_lsd_loss += lsd_loss_val
            total_mse_loss += nn.MSELoss()(denoised_latents, latents["z"]).item()
            total_loss += loss

        valid_loss = total_loss / num_valid
        valid_lsd_loss = total_lsd_loss / num_valid
        valid_mse_loss = total_mse_loss / num_valid
        ic(valid_loss, valid_lsd_loss, valid_mse_loss)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Loss/valid_lsd', valid_lsd_loss, epoch)
        writer.add_scalar('Loss/valid_mse', valid_mse_loss, epoch)

        if valid_loss < best_loss:
            best_loss = valid_loss
            ic("Saving best model...")
            th.save(unet.state_dict(), config["artifacts_dir"]+'/unet.best.net')
            patience = 15
        else:
            patience -= 1
        
        ic(patience)
        
        if patience == 0:
            ic("Early stopping...")
            break
        
        unet.train()

    ic("Loading best model...")
    unet.load_state_dict(th.load(config["artifacts_dir"]+'/unet.best.net'))
    
    unet.eval()
    total_loss = 0.0
    total_lsd_loss = 0.0
    total_mse_loss = 0.0

    # Testing
    scheduler.set_timesteps(1000)
    for i, data in enumerate(tqdm(test_dataloader, mininterval=60, maxinterval=60, desc="Testing Progress")):
        data_copy = {}

        for data_kind in data:
            data[data_kind] = data[data_kind].to('cuda')
            data_copy[data_kind] = data[data_kind]

        latents = net.encode(data_copy)
        returns = net.decode(data_copy, latents["z"])
        anthro_features_norm = (data_copy["AnthroFeatures"] - anthro_standardize["mean"]) / anthro_standardize["std"]

        loss = test_step(unet, scheduler, latents["z"], anthro_features_norm, criterion)

        denoised_latents = test_generation(unet, scheduler, anthro_features_norm)

        mse_loss_val = nn.MSELoss()(denoised_latents, latents["z"]).item()

        # ic(denoised_latents, latents["z"])
        # ic(mse_loss_val)

        decoded_data = net.decode(data_copy, denoised_latents)

        lsd_loss_val = lsd_loss(decoded_data['HRTF_mag'], data['HRTF_mag'], dim=-1, data_kind='HRTF_mag', mean=True).item()

        if i == 0:
            plot_mag_Angle_vs_Freq(HRTF_mag=data['HRTF_mag'][0,:,:,:], SrcPos=data['SrcPos'][0,:,:].cpu(), fs=config['max_frequency'], figdir=f"{config['artifacts_dir']}/figure/HRTF_mag_2D/", fname=f"{config['artifacts_dir']}/figure/HRTF_mag_2D/HRTF_Mag_orig_{i}")
            plot_mag_Angle_vs_Freq(HRTF_mag=returns['HRTF_mag'][0,:,:,:], SrcPos=data['SrcPos'][0,:,:].cpu(), fs=config['max_frequency'], figdir=f"{config['artifacts_dir']}/figure/HRTF_mag_2D/", fname=f"{config['artifacts_dir']}/figure/HRTF_mag_2D/HRTF_Mag_recon_{i}")
            plot_mag_Angle_vs_Freq(HRTF_mag=decoded_data['HRTF_mag'][0,:,:,:], SrcPos=data['SrcPos'][0,:,:].cpu(), fs=config['max_frequency'], figdir=f"{config['artifacts_dir']}/figure/HRTF_mag_2D/", fname=f"{config['artifacts_dir']}/figure/HRTF_mag_2D/HRTF_Mag_latent_{i}")

        ic(loss, lsd_loss_val)

        total_lsd_loss += lsd_loss_val
        total_mse_loss += mse_loss_val
        total_loss += loss

    test_loss = total_loss / num_test
    test_lsd_loss = total_lsd_loss / num_test
    test_mse_loss = total_mse_loss / num_test
    ic(test_loss, test_lsd_loss, test_mse_loss)
    
    writer.flush()
    print("Done.")
