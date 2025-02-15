import os
import sys
import argparse
import datetime
import time
import pprint

import numpy as np
import torch as th
import torchaudio as ta   
import torch.nn as nn
import scipy
import matplotlib.pyplot as plt   
import math      
import optuna
import pandas
import sqlite3
import pymysql
from icecream import ic
ic.configureOutput(includeContext=True)

import pickle

from src.dataset import HRTFDataset #, HRTFTestset
from src.models import HRTFApproxNetwork, HRTFApproxNetwork_AE, HRTFApproxNetwork_CAE, HRTFApproxNetwork_HyperCAE, HRTFApproxNetwork_AE_PINN,HRTFApproxNetwork_PINN, HRTFApproxNetwork_CNN, HRTFApproxNetwork_Lemaire05, HRTFApproxNetwork_FIAE, GCU
from src.trainer import Trainer
from src.losses import MAE, MSE, NMSE, L2Loss_angle, LSD_before_mean,VarLoss, CosSimLoss, RegLoss, LSD, HelmholtzLoss, HelmholtzLoss_Cart, HuberLoss, LSDdiffLoss, WeightedLSD, RMS, CosDistIntra, CosDistIntraSquared, PhaseLoss, LogSpecDiff_1st, LogSpecDiff_2nd, LogSpecDiff_4th, NotchPriorWeightedLSD, ItakuraSaito, LSD_before_mean, ILD_AE
from src.configs import * # configs
from src.utils import SpecialFunc, plothrir, sph2cart, plotmaghrtf, plotcz, plotcolonpos, aprox_t_des, hrir2itd, minphase_recon, assign_itd, posTF2IR_dim4, plotazimzeni, replace_activation_function, plot_mag_Angle_vs_Freq, Data_ITD_2_HRIR

from torch.utils.tensorboard import SummaryWriter # tensorboard

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

def train(trainer, ptins=None):
    BEST_LOSS = 1e+16
    LAST_SAVED = -1
    for epoch in range(args.start_ep+1, args.start_ep+config["epochs"]+1):
        trainer.net.cuda()
        trainer.net.train()
        if epoch == 1 and config["freeze_de"]:
            for param in net.Decoder_z.parameters():
                param.requires_grad = False
            print("Freeze Decoder.")
        ## Train
        if config["model"] in ["AE", "CAE", "HCAE"]:
            loss_train, coeff_pred, z_train = trainer.train_1ep(epoch, coeff = coeff_train.cuda()) 
            if config["freeze_de"]:
                for param in net.parameters():
                    param.requires_grad = True
                print("Unfreeze Decoder.")
        elif config["model"] == "CNN":
            loss_train, coeff_pred, coeff_in, z_train = trainer.train_1ep(epoch, coeff = coeff_train.cuda()) 
        else:
            loss_train = trainer.train_1ep(epoch) 
            # print("trained")
        if config["lr_update"] == 'train':
            ## Update Learning rate
            trainer.optimizer.update_lr_one(loss_train["loss"])
        ## TensorBoard
        for k, v in loss_train.items():
            writer.add_scalar(k + " / train", v, epoch)

        ##### for debug
        if config["model"] in ["AE", "CAE", "HCAE"]:
            # save predicted coeff.
            reg_w = config["reg_w"]
            # reg_w = 1e-7
            # reg_w = config["loss_weights"]["reg"]
            ptname_c = f'coeff_pred_{ptins}{config["max_frequency"]}_{round(config["fft_length"]/2)}_{config["max_truncation_order"]}_{reg_w:.0e}.pt'
            # ptname_c = "coeff_pred_"+ ptins + str(config["max_frequency"]) + "_" + str(round(config["fft_length"]/2)) + "_" + str(config["max_truncation_order"]) + ".pt"
            ptpath_c = config["artifacts_dir"] + "/" + ptname_c
            th.save(coeff_pred.cpu(), ptpath_c)
            if config["model"] == "CNN":
                ptname_c_in = f'coeff_in_{ptins}{config["max_frequency"]}_{round(config["fft_length"]/2)}_{config["max_truncation_order"]}_{reg_w:.0e}.pt'
                ptpath_c_in = config["artifacts_dir"] + "/" + ptname_c_in
                th.save(coeff_in.cpu(), ptpath_c_in)
            # save embedded vector
            ptname_z = f'z_train_{ptins}{config["max_frequency"]}_{round(config["fft_length"]/2)}_{config["max_truncation_order"]}_{reg_w:.0e}.pt'
            # ptname_z = "z_train_"+ ptins + str(config["max_frequency"]) + "_" + str(round(config["fft_length"]/2)) + "_" + str(config["max_truncation_order"]) + ".pt"
            ptpath_z = config["artifacts_dir"] + "/" + ptname_z
            th.save(z_train.cpu(), ptpath_z)
            #
            ####################

        ## Validation
        print("----------")
        print(f"epoch {epoch} (valid)")
        t_start = time.time()
        use_cuda = False if config["model"] == 'AE_PINN' else True
        
        loss_valid = test(trainer.net, flg_save, BEST_LOSS, validdataset, 'valid', use_cuda=use_cuda)
        t_end = time.time()
        time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
        print(time_str)
        ## TensorBoard
        for k, v in loss_valid.items():
            writer.add_scalar(k + " / valid", v, epoch)
        if config["lr_update"] == 'valid':
            ## Update Learning rate
            trainer.optimizer.update_lr_one(loss_valid["loss"])
        
        if config["lr_update"] == 'step':
            if epoch in config["lr_milestones"]:
                trainer.optimizer.update_lr_step(config["lr_gamma"])

        if loss_valid["loss"] < BEST_LOSS:
            BEST_LOSS = loss_valid["loss"]
            LAST_SAVED = epoch
            print("Best Loss. Saving model!")
            # trainer.save(suffix="best_"+f'{epoch:03}'+"ep")
            trainer.save(suffix="best")
        elif config["save_frequency"] > 0 and epoch % config["save_frequency"] == 0 and not epoch == config["epochs"]:
            print("Saving model!")
            trainer.save(suffix="log_"+f'{epoch:03}'+"ep")
        else:
            print("Not saving model! Last saved: {}".format(LAST_SAVED))
        print("---------------------")
        if not config["model"] == "Lemaire05" and not config["model"] == "FIAE":
            plotcz(dir=config["artifacts_dir"],config=config)
        plt.clf()
        plt.close()
    # Save final model
    trainer.save(suffix="final_"+f'{epoch:03}'+"ep")
    # plotcz(dir=config["artifacts_dir"],config=config)

def train_optuna(trainer):
    print("---------------------")
    for k,v in config.items():
        if isinstance(v, float):
            print(f"{k}: {v:.1e}")
        elif isinstance(v, dict):
            print(f"{k}:")
            for kk,vv in v.items():
                print(f"{kk}: {vv:.1e}")
        else:
            print(f"{k}: {v}")
    print("-------")
    BEST_LOSS = 1e+16
    LAST_SAVED = -1
    for epoch in range(args.start_ep+1, args.start_ep+config["epochs"]+1):
        trainer.net.cuda()
        trainer.net.train()
        ## Train
        if config["model"] in ["AE", "CAE", "HCAE", "CNN"]:
            loss_train, _, _ = trainer.train_1ep(epoch, coeff = coeff_train.cuda()) 
        else:
            loss_train = trainer.train_1ep(epoch) 
        ## Update Learning rate
        trainer.optimizer.update_lr_one(loss_train["loss"])
        ## Validation
        print("----------")
        print(f"epoch {epoch} (valid)")
        t_start = time.time()
        loss_valid = test(trainer.net, flg_save, BEST_LOSS, validdataset, 'valid'+config["timestamp"])
        t_end = time.time()
        time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
        print(time_str)

        score = loss_valid["lsd"]
        # score = loss_valid["loss"]
        if score < BEST_LOSS:
            BEST_LOSS = score
            LAST_SAVED = epoch
            print("Best Loss!")
        else:
            print("Last saved: {}".format(LAST_SAVED))
        print("---------------------")
        if th.isnan(score):
            break
        if epoch >= 100 and BEST_LOSS > 15.0:
            break
    return {"best_loss_valid": BEST_LOSS}

def test(net, flg_save, best_loss, data, mode, use_coeff=False, coeff=None, use_cuda=True):
    device = 'cuda' if use_cuda else 'cpu'
    # net.cpu()
    net.to(device)
    net.eval()
    use_cuda_forward = True

    if mode.startswith('train'):
        mode_str = 'train'
    elif mode.startswith('valid'):
        mode_str = 'valid'
    elif mode.startswith('test'):
        mode_str = 'test'
    else:
        raise NotImplementedError

    num_sub = sum([len(config['sub_index'][db_name][mode_str]) for db_name in config['database']])
    subject_list = np.arange(0, num_sub)

    returns = {}
    prediction_all = {}
    keys = []

    for data_kind in config["data_kind_interp"]:
        if hasattr(data[data_kind], 'device'):
            data[data_kind] = data[data_kind].to(device)
        if data_kind == 'ITD':
            keys.append("mae_itd")
            mae_loss = MAE()
        elif data_kind == 'HRTF_mag':
            keys.append("lsd")
            lsd_loss = LSD()
            keys.append("mae_ild")
            ild_loss = ILD_AE()


    # ic(data.keys())

    for k in keys:
        returns.setdefault(k,0)
    for index in subject_list:
        db_name = config["Table"][mode_str]["db_name"][index]
        sub_id = config["Table"][mode_str]["sub_idx"][index] - config['sub_index'][db_name][mode_str][0]
        data_slice = {}
        for data_kind in data:
            data_slice[data_kind] = data[data_kind][db_name][..., sub_id:sub_id+1].permute([-1] + th.arange(data[data_kind][db_name].dim())[:-1].tolist()).to(device)
            # if data_kind == 'HRTF_mag':
            #     ic(data_slice[data_kind].shape)
            #     ic(th.max(data_slice[data_kind]))
            #     ic(th.min(data_slice[data_kind]))
            #     ic(th.max(data[data_kind][db_name][..., sub_id:sub_id+1]))
            #     ic(th.min(data[data_kind][db_name][..., sub_id:sub_id+1]))
            # ic(data_kind, data_slice[data_kind].dtype, data_slice[data_kind].device, data_slice[data_kind].shape)
        data_slice['db_name'] = [db_name]
        data_slice['sub_idx'] = [sub_id]
        # ic(sub_id)
        # ic(data_slice['SrcPos'].shape)
        
        prediction = net.forward(data=data_slice.copy(), mode=mode)
        prediction["idx_mes_pos"] = range(0, data_slice["SrcPos"].shape[1]) # new (?)
        # if index == 0:
        #     ic(data_slice["HRTF_mag"], prediction["HRTF_mag"])
        for k in prediction:
            if hasattr(prediction[k], 'detach'):
                prediction[k] = prediction[k].detach().clone()
 
        additional_data_kind = [] if config["data_kind_interp"]==['ITD'] else ['lsd_bm'] #, 'HRIR']

        for data_kind in config["data_kind_interp"] + additional_data_kind:
            if data_kind == 'ITD':
                returns["mae_itd"] += mae_loss(prediction['ITD'], data_slice['ITD']) # S,B
            elif data_kind == 'HRTF_mag':
                returns["lsd"] += lsd_loss(prediction['HRTF_mag'], data_slice['HRTF_mag'], dim=-1, data_kind='HRTF_mag', mean=True) # S,B,2,L
                returns["mae_ild"] += ild_loss(prediction['HRTF_mag'], data_slice['HRTF_mag'], dim=-2, data_kind='HRTF_mag', mean=True) # S,B,2,L

            elif data_kind == 'HRIR':
                if 'ITD' in config["data_kind_interp"]:
                    t = prediction['ITD'].permute(1,0) # B,S
                elif config["use_itd_gt"]:
                    t = data_slice['ITD'].permute(1,0) # B,S
                else:
                    t = None
                prediction['HRIR'] = Data_ITD_2_HRIR(prediction['HRTF_mag'].permute(1,2,3,0), itd=t, data_kind='HRTF_mag', config=config).permute(3,0,1,2)
            elif data_kind == 'lsd_bm':
                prediction['lsd_bm'] = lsd_loss(prediction['HRTF_mag'], data_slice['HRTF_mag'], dim=-1, data_kind='HRTF_mag', mean=False) # S,B,2
            
            if not data_kind in prediction_all:
                prediction_all[data_kind] = {}
            if not db_name in prediction_all[data_kind]:
                if data_kind in data:
                    prediction_all[data_kind][db_name] = th.zeros_like(data[data_kind][db_name])
                else:
                    if data_kind == 'lsd_bm':
                        prediction_all[data_kind][db_name] = th.zeros_like(data['HRTF_mag'][db_name][:,:,0,:])
            prediction_all[data_kind][db_name][..., sub_id:sub_id+1] = prediction[data_kind].permute(th.arange(prediction[data_kind].dim())[1:].tolist() + [0])
        
        if flg_save or sub_id == config['sub_index'][db_name][mode_str][0] - config['sub_index'][db_name][mode_str][0]:

            if 'ITD' in config["data_kind_interp"]:
                #=== plot ITD ====
                fig_dir_itd = f'{config["artifacts_dir"]}/figure/ITD/'
                os.makedirs(fig_dir_itd, exist_ok=True)

                plotazimzeni(pos=data_slice['SrcPos'][0,:,:].cpu(), c=prediction['ITD'][0,:].cpu().detach()*1000,fname=f"{fig_dir_itd}ITD_{data_slice['db_name'][0]}-sub-{data_slice['sub_idx'][0]+1}_{mode}",title=f'sub:{sub_id+1}',cblabel=f'ITD (ms)',cmap='bwr',figsize=(10.5,5),dpi=300,emphasize_mes_pos=True, idx_mes_pos=prediction["idx_mes_pos"])

                plotazimzeni(pos=data_slice['SrcPos'][0,:,:].cpu(), c=th.abs(prediction['ITD'][0,:]-data_slice['ITD'][0,:]).cpu().detach()*1000,fname=f"{fig_dir_itd}ITD_AE_{data_slice['db_name'][0]}-sub-{data_slice['sub_idx'][0]+1}_{mode}",title=f'sub:{sub_id+1}',cblabel=f'AE of ITD (ms)',cmap='gist_heat',figsize=(10.5,5),dpi=300,emphasize_mes_pos=True, idx_mes_pos=prediction["idx_mes_pos"])
            
            if 'HRTF_mag' in config["data_kind_interp"]:
                #=== plot HRTF_mag ====
                plotmaghrtf(srcpos=data_slice['SrcPos'][0,:,:].cpu(), sig_gt=data_slice['HRTF_mag'][0,:,:,:], sig_pred=prediction['HRTF_mag'][0,:,:,:], idx_plot_list=np.array(config["idx_plot_list"][data_slice['db_name'][0]]),config=config,  figdir=f"{config['artifacts_dir']}/figure/HRTF/", fname=f"{config['artifacts_dir']}/figure/HRTF/HRTF_Mag_{data_slice['db_name'][0]}-sub-{data_slice['sub_idx'][0]+1}_{mode}", data_kind = 'HRTF_mag')

                plot_mag_Angle_vs_Freq(HRTF_mag=prediction['HRTF_mag'][0,:,:,:], SrcPos=data_slice['SrcPos'][0,:,:].cpu(), fs=config['max_frequency'], figdir=f"{config['artifacts_dir']}/figure/HRTF_mag_2D/", fname=f"{config['artifacts_dir']}/figure/HRTF_mag_2D/HRTF_Mag_{data_slice['db_name'][0]}-sub-{data_slice['sub_idx'][0]+1}_{mode}")

                #=== plot HRIR ====
                # _, _ = plothrir(srcpos=data_slice['SrcPos'][0,:,:].cpu(), hrtf_gt=None,hrtf_pred=None, idx_plot_list=np.array(config["idx_plot_list"][data_slice['db_name'][0]]), config=config, hrir_gt=data_slice['HRIR'][0,:,:,:], hrir_pred=prediction['HRIR'][0,:,:,:], figdir=f"{config['artifacts_dir']}/figure/HRIR/", fname=f"{config['artifacts_dir']}/figure/HRIR/HRIR_{data_slice['db_name'][0]}-sub-{sub_id+1}_{mode}")

                #=== plot LSD_bm ====
                fig_dir_lsd_bm = f'{config["artifacts_dir"]}/figure/LSD/'
                os.makedirs(fig_dir_lsd_bm, exist_ok=True)

                emphasize_mes_pos = True if "idx_mes_pos" in prediction else False
                    
                for ch in range(2):
                    ch_str_l = ['left', 'right']
                    ch_str = ch_str_l[ch]
                    plotazimzeni(pos=data_slice['SrcPos'][0,:,:].cpu(), c=prediction['lsd_bm'][0,:,ch].cpu(), fname=f'{fig_dir_lsd_bm}lsd_{db_name}-sub-{sub_id+1}_{ch_str}_{mode}',title=f'{db_name} sub:{sub_id+1}, {ch_str}',cblabel=f'LSD (dB)',cmap='gist_heat',figsize=(10.5,5),dpi=300,emphasize_mes_pos=emphasize_mes_pos, idx_mes_pos=prediction["idx_mes_pos"],vmin=0, vmax=10)

        if flg_save and sub_id == config['sub_index'][db_name][mode_str][-1] - config['sub_index'][db_name][mode_str][0]:
            for data_kind in prediction_all:
                # ic(data_kind)
                pt_dir = f'{config["artifacts_dir"]}/{data_kind}'
                os.makedirs(pt_dir, exist_ok=True)
                for db_name in prediction_all[data_kind]:
                    th.save(prediction_all[data_kind][db_name], f'{pt_dir}/{data_kind}_{mode}_{db_name}.pt')
                    # ic(th.mean(prediction_all[data_kind][db_name]))

    
    loss = 0
    for k in keys:
        if k in config["loss_weights"] and config["loss_weights"][k] > 0:
            loss += config["loss_weights"][k] * returns[k]
    returns["loss"] = loss.detach().clone()
    for k in returns:
        returns[k] /= len(subject_list)

    loss_str = "    ".join([f"{k}:{returns[k]:.6}" for k in sorted(returns.keys())])
    print(loss_str)
    
    if returns["loss"] < best_loss:
        print("Best Loss.")

    return returns


def test_old(net, flg_save, best_loss, data, mode, use_coeff=False, coeff=None, use_cuda=True):
    device = 'cuda' if use_cuda else 'cpu'
    # net.cpu()
    net.to(device)
    net.eval()
    use_cuda_forward = True

    if mode.startswith('train'):
        mode_str = 'train'
    elif mode.startswith('valid'):
        mode_str = 'valid'
    elif mode.startswith('test'):
        mode_str = 'test'
    else:
        raise NotImplementedError

    num_sub = sum([len(config['sub_index'][db_name][mode_str]) for db_name in config['database']])
    subject_list = np.arange(0, num_sub)
    if config["DNN_for_interp_ITD"]:
        srcpos, itd_gt = data['SrcPos'], data['ITD']
        # srcpos, itd_gt = srcpos.to(device), itd_gt.to(device)
        itd_gt_all = {}
        itd_pred_all = {}
        if config["model"] == "FIAE":
            returns = {}
            # subject_list = np.arange(0,srcpos.shape[-1])
            mae_loss = MAE()
            keys = ["mae_itd"]
            for k in keys:
                returns.setdefault(k,0)
            for index in subject_list:
                db_name = config["Table"][mode_str]["db_name"][index]
                sub_id = config["Table"][mode_str]["sub_idx"][index] - config['sub_index'][db_name][mode_str][0]
                itd_sub = itd_gt[db_name][..., sub_id:sub_id+1]
                srcpos_sub = srcpos[db_name][..., sub_id:sub_id+1]
                srcpos_sub, itd_gt_sub = srcpos_sub.to(device), itd_gt_sub.to(device)
                prediction = net.forward(input=itd_sub, srcpos=srcpos_sub, mode='test', db_name=db_name)

                itd_pred = prediction["output"].detach().clone()
                returns["mae_itd"] += mae_loss(itd_pred,itd_sub)

                if flg_save:
                    fig_dir_itd = f'{config["artifacts_dir"]}/figure/ITD/'
                    os.makedirs(fig_dir_itd, exist_ok=True)

                    plotazimzeni(pos=srcpos[db_name][:,:,sub_id].cpu(),c=itd_pred[:,0].cpu().detach()*1000,fname=f'{fig_dir_itd}itd_{mode}_{db_name}-sub-{sub_id+1}',title=f'{db_name} sub:{sub_id+1}',cblabel=f'ITD (ms)',cmap='bwr',figsize=(10.5,5),dpi=300,emphasize_mes_pos=True, idx_mes_pos=prediction["idx_mes_pos"])

                    plotazimzeni(pos=srcpos[db_name][:,:,sub_id].cpu(),c=th.abs(itd_pred[:,0]-itd_gt[db_name][:,sub_id]).cpu().detach()*1000,fname=f'{fig_dir_itd}itd_ae_{mode}_{db_name}-sub-{sub_id+1}',title=f'{db_name} sub:{sub_id+1}',cblabel=f'AE of ITD (ms)',cmap='gist_heat',figsize=(10.5,5),dpi=300,emphasize_mes_pos=True, idx_mes_pos=prediction["idx_mes_pos"])
                
                if not db_name in itd_gt_all.keys():
                    itd_gt_all[db_name] = th.zeros(itd_gt.shape[0],len(config['sub_index'][db_name][mode_str]))
                if not db_name in itd_pred_all.keys():
                    itd_pred_all[db_name] = th.zeros(itd_pred.shape[0],len(config['sub_index'][db_name][mode_str]))
                
                itd_gt_all[db_name][:,sub_id] = itd_sub[:,0]
                itd_pred_all[db_name][:,sub_id] = itd_pred[:,0]

                if sub_id == config['sub_index'][db_name][mode_str][-1] - config['sub_index'][db_name][mode_str][0]:
                    itd_dir = f'{config["artifacts_dir"]}/ITD'
                    os.makedirs(itd_dir, exist_ok=True)
                    th.save(itd_pred_all[db_name], f'{itd_dir}/itd_pred_{mode}_{db_name}.pt')
                    if mode.startswith('test'):
                        mode_gt = 'test'
                    else:
                        mode_gt = mode
                    th.save(itd_gt_all[db_name], f'{itd_dir}/itd_gt_{mode_gt}_{db_name}.pt')
        
            loss = returns["mae_itd"]
            returns["loss"] = loss.detach().clone()
            # returns_new = {
            #     "loss": loss.detach().clone(),
            # }
            # returns.update(returns_new)
            for k in returns:
                returns[k] /= len(subject_list)
        else:
            raise NotImplementedError
    elif config["DNN_for_interp_HRIR"]:
        srcpos, hrir_gt, hrtf_gt = data['SrcPos'], data['HRIR'], data['HRTF']
        if config["model"] == "FIAE":
            returns = {}
            l2_loss = MSE()
            lsd_loss = LSD()
            lsd_bm_loss = LSD_before_mean()
            keys = ["l2_time", "lsd"]
            for k in keys:
                returns.setdefault(k,0)
            
            hrir_gt_all = {}
            hrir_pred_all = {}
            lsd_bm_all = {}
            for index in subject_list:
                db_name = config["Table"][mode_str]["db_name"][index]
                sub_id = config["Table"][mode_str]["sub_idx"][index] - config['sub_index'][db_name][mode_str][0]
                hrtf_sub = hrtf_gt[db_name][..., sub_id:sub_id+1]
                hrir_sub = hrir_gt[db_name][..., sub_id:sub_id+1]
                srcpos_sub = srcpos[db_name][..., sub_id:sub_id+1]
                # print(f'{db_name} {mode}-{sub_id}') 
                srcpos_sub, hrir_sub, hrtf_sub = srcpos_sub.to(device), hrir_sub.to(device), hrtf_sub.to(device)
                prediction = net.forward(input=hrir_sub, srcpos=srcpos_sub, mode='test', db_name=db_name)
                hrir_pred = prediction["output"].detach().clone()
                hrtf_pred = th.conj(th.fft.fft(hrir_pred, dim=2))[:,:,1:config["fft_length"]//2+1,:]

                returns["l2_time"] += l2_loss(hrir_pred, hrir_sub)
                returns["lsd"] += lsd_loss(hrtf_pred, hrtf_sub)

                if flg_save or sub_id == config['sub_index'][db_name][mode_str][0] - config['sub_index'][db_name][mode_str][0]:
                    plotmaghrtf(srcpos=srcpos[db_name][:,:,sub_id],hrtf_gt=hrtf_gt[db_name][:,:,:,sub_id],hrtf_pred=hrtf_pred,idx_plot_list=np.array(config["idx_plot_list"][db_name]), config=config, figdir=f"{config['artifacts_dir']}/figure/HRTF/", fname=f"{config['artifacts_dir']}/figure/HRTF/HRTF_Mag_{db_name}-sub-{sub_id+1}_{mode}")

                    plothrir(srcpos=srcpos[db_name][:,:,sub_id],hrtf_gt=None,hrtf_pred=None,idx_plot_list=np.array(config["idx_plot_list"][db_name]), config=config, hrir_gt=hrir_sub.squeeze(), hrir_pred=hrir_pred, figdir=f"{config['artifacts_dir']}/figure/HRIR/", fname=f"{config['artifacts_dir']}/figure/HRIR/HRIR_{db_name}-sub-{sub_id+1}_{mode}")

                    hrirs_all_list = ['hrir_gt_all','hrir_pred_all']
                    hrirs_list = ['hrir_sub', 'hrir_pred']
                    for hrirs_all, hrirs in zip(hrirs_all_list, hrirs_list):
                        if not db_name in eval(hrirs_all).keys():
                            eval(hrirs_all)[db_name] = th.zeros(eval(hrirs).shape[0],eval(hrirs).shape[1],eval(hrirs).shape[2],len(config['sub_index'][db_name][mode_str]))
                        if eval(hrirs).dim()==4:
                            eval(hrirs_all)[db_name][:,:,:,sub_id] = eval(hrirs)[:,:,:,0]
                        else:
                            eval(hrirs_all)[db_name][:,:,:,sub_id] = eval(hrirs)

                    if not db_name in lsd_bm_all.keys():
                        lsd_bm_all[db_name] = th.zeros(hrir_pred.shape[0],hrir_pred.shape[1],len(config['sub_index'][db_name][mode_str])) # B,2,S

                    # plot LSD (before mean)
                    fig_dir_lsd_bm = f'{config["artifacts_dir"]}/figure/LSD/'
                    os.makedirs(fig_dir_lsd_bm, exist_ok=True)
                    lsd_bm = lsd_bm_loss(hrtf_pred, hrtf_sub).squeeze().cpu() # (B,2)

                    emphasize_mes_pos = True if "idx_mes_pos" in prediction else False
                        
                    for ch in range(2):
                        ch_str_l = ['left', 'right']
                        ch_str = ch_str_l[ch]
                        plotazimzeni(pos=srcpos[db_name][:,:,sub_id].cpu(),c=lsd_bm[:,ch],fname=f'{fig_dir_lsd_bm}lsd_{db_name}-sub-{sub_id+1}_{ch_str}_{mode}',title=f'{db_name} sub:{sub_id+1}, {ch_str}',cblabel=f'LSD (dB)',cmap='gist_heat',figsize=(10.5,5),dpi=300,emphasize_mes_pos=emphasize_mes_pos, idx_mes_pos=prediction["idx_mes_pos"],vmin=0, vmax=10)
                    # store LSD (before mean)
                    lsd_bm_all[db_name][:,:,sub_id] = lsd_bm

                if sub_id == config['sub_index'][db_name][mode_str][-1] - config['sub_index'][db_name][mode_str][0]:
                    hrir_dir = f'{config["artifacts_dir"]}/HRIR'
                    lsd_bm_dir = f'{config["artifacts_dir"]}/LSD'
                    os.makedirs(hrir_dir, exist_ok=True)
                    os.makedirs(lsd_bm_dir, exist_ok=True)
                    th.save(hrir_pred_all[db_name], f'{hrir_dir}/hrir_pred_{mode}_{db_name}.pt')
                    th.save(hrir_gt_all[db_name], f'{hrir_dir}/hrir_gt_{db_name}.pt')
                    th.save(lsd_bm_all[db_name], f'{lsd_bm_dir}/lsd_{mode}_{db_name}_before_mean.pt')
                
            loss = 0
            for k in keys:
                if k in config["loss_weights"] and config["loss_weights"][k] > 0:
                    loss += config["loss_weights"][k] * returns[k]
            returns["loss"] = loss.detach().clone()
            for k in returns:
                returns[k] /= len(subject_list)
        else:
            raise NotImplementedError
    else:
        srcpos, hrtf_gt = data['SrcPos'], data['HRTF']
        if config["model"] in ["AE", "CAE", "HCAE", "AE_PINN" , "CNN", "FIAE"]:
            
            returns = {}

            # subject_list = np.arange(0,srcpos.shape[-1])
            l2_loss = MSE()
            l2_loss_n = NMSE()
            # l2_loss_angle = L2Loss_angle()
            lsd_loss = LSD()
            lsd_bm_loss = LSD_before_mean()
            reg_loss = RegLoss()
            var_loss = VarLoss()
            cosdist_loss = CosDistIntra()
            cosdistsq = CosDistIntraSquared()
            phase_loss = PhaseLoss()
            ls_diff_1st_loss = LogSpecDiff_1st()
            ls_diff_2nd_loss = LogSpecDiff_2nd()
            ls_diff_4th_loss = LogSpecDiff_4th()
            IS = ItakuraSaito(bottom=config["is_pm_bottom"]) # 'target' or 'data'
            # cossim_loss = CosSimLoss()
            helmholtz_loss_cart = HelmholtzLoss_Cart()

            keys = ["l2_rec", "l2_rec_n", "lsd", "reg",'var_z',"cdintra_z", "cdintrasq_z",'phase_rec','ls_diff_1st','ls_diff_2nd','ls_diff_4th','is_pm','cossim','helmholtz']
            for k in keys:
                returns.setdefault(k,0)
            returns_zi = {}

            hrir_gt_all = {}
            hrir_pred_all = {}
            hrir_pred_min_all = {}
            lsd_bm_all = {}

            for index in subject_list:
                db_name = config["Table"][mode_str]["db_name"][index]
                sub_id = config["Table"][mode_str]["sub_idx"][index] - config['sub_index'][db_name][mode_str][0]
                hrtf_sub = hrtf_gt[db_name][..., sub_id:sub_id+1]
                srcpos_sub = srcpos[db_name][..., sub_id:sub_id+1]
                # print(f'{db_name} {mode}-{sub_id}') 
                srcpos_sub, hrtf_sub = srcpos_sub.to(device), hrtf_sub.to(device)

                if config["model"] == "CNN":
                    hrtf_re, hrtf_im = th.real(hrtf_sub), th.imag(hrtf_sub)
                    prediction = net.forward(hrtf_re, hrtf_im, srcpos_sub)
                else:
                    if config["model"] in ["AE", "FIAE"]:
                        input = th.cat((th.real(hrtf_sub[:,0,:]), th.imag(hrtf_sub[:,0,:]), th.real(hrtf_sub[:,1,:]), th.imag(hrtf_sub[:,1,:])), dim=1)
                    
                    else:
                        input = th.cat((th.real(hrtf_sub[:,0,:]).unsqueeze(1), th.imag(hrtf_sub[:,0,:]).unsqueeze(1), th.real(hrtf_sub[:,1,:]).unsqueeze(1), th.imag(hrtf_sub[:,1,:]).unsqueeze(1)), dim=1)
                        # print(input.shape) # torch.Size([440, 4, 128])
                    srcpos_sub = srcpos[db_name][:,:,sub_id:sub_id+1]
                    if config["model"] == "AE":
                        prediction = net.forward(input=input, use_cuda_forward=True, use_srcpos=True, srcpos=srcpos_sub, mode='test', db_name=db_name)
                    elif config["model"] == "FIAE":
                        prediction = net.forward(input=input, srcpos=srcpos_sub, mode='test', db_name=db_name)
                hrtf_pred = prediction["output"].detach().clone()
                returns["l2_rec"] += l2_loss(hrtf_pred, hrtf_sub)
                returns["l2_rec_n"] += l2_loss_n(hrtf_pred, hrtf_sub)
                returns["lsd"] += lsd_loss(hrtf_pred, hrtf_sub)
                returns["phase_rec"] += phase_loss(hrtf_pred, hrtf_sub)
                returns["ls_diff_1st"] += ls_diff_1st_loss(hrtf_pred, hrtf_sub)
                returns["ls_diff_2nd"] += ls_diff_2nd_loss(hrtf_pred, hrtf_sub)
                returns["ls_diff_4th"] += ls_diff_4th_loss(hrtf_pred, hrtf_sub)
                returns["is_pm"] += IS(hrtf_pred, hrtf_sub)
                if config["model"] == "FIAE":
                    if config["aggregation_mean"]:
                        returns["var_z"] += var_loss(prediction["z_bm"].detach().clone().permute(2,0,1,3))
                
                if not config["model"] == "FIAE":
                    # r_pred = prediction["r"] # fixed in HUTUBS 
                    # theta_pred = prediction["theta"].detach().clone()
                    # phi_pred = prediction["phi"].detach().clone()
                    # vec_pred = prediction["vec"].detach().clone()
                    z_pred = prediction["z"].detach().clone()
                    if config["num_classifier"] > 1:
                        for i in range(config["num_classifier"]):
                            if f"cdintra_z_{i}" in returns_zi:
                                returns_zi[f"cdintra_z_{i}"] += cosdist_loss(prediction[f"z_{i}"]).detach().clone()
                            else:
                                returns_zi[f"cdintra_z_{i}"] = cosdist_loss(prediction[f"z_{i}"]).detach().clone()
                
                    returns["var_z"] += var_loss(z_pred)
                    returns["cdintra_z"] += cosdist_loss(z_pred)
                    returns["cdintrasq_z"] += cosdistsq(z_pred)

                if config["model"] in ["AE", "CAE" , "HCAE", "CNN"]:
                    coeff_pred = prediction["coeff"].detach().clone()
                    returns["reg"] += reg_loss(coeff_pred, SpecialFunc(maxto=config["max_truncation_order"]).n_vec)
                elif config["model"]=="AE_PINN":
                    returns["helmholtz"] += helmholtz_loss_cart(xf=prediction['xf'], yf=prediction['yf'], zf=prediction['zf'], func=prediction['output_4ch'], k=prediction['k'], L=prediction['L'], B=prediction['B']).detach().clone()
                
                if sub_id == 0 and not config["model"] == "FIAE" and not config["model"]=="AE":
                    #== weight-mean visualize====
                    weight_mean = prediction["weight_mean"]
                    vec_cart_gt = sph2cart(srcpos_sub[:,1,0],srcpos_sub[:,2,0],srcpos_sub[:,0,0])
                    # print(srcpos_sub.shape) # torch.Size([440, 3, 1])
                    # print(weight_mean.shape) # torch.Size([9])
                    if config["num_pts"] < 440:
                        t = round(config["num_pts"]**0.5-1)
                        idx_t_des = aprox_t_des(pts=vec_cart_gt/srcpos_sub[0,0,0], t=t, plot=False)
                        vec_cart_tdes = vec_cart_gt[idx_t_des,:]
                    else:
                        vec_cart_tdes = vec_cart_gt
                    plotcolonpos(pos=vec_cart_tdes.to('cpu').detach().numpy().copy(),c=weight_mean.to('cpu').detach().numpy().copy(),config=config,path=f'{config["artifacts_dir"]}',mode=mode)
                    #=========

                if flg_save or sub_id == config['sub_index'][db_name][mode_str][0] - config['sub_index'][db_name][mode_str][0]:
                    # Visualise
                    if config["model"] in ["AE", "CAE", "HCAE", "AE_PINN", "CNN", "FIAE"]:
                        ## plot HRTFs' magnitude
                        plotmaghrtf(srcpos=srcpos[db_name][:,:,sub_id],hrtf_gt=hrtf_gt[db_name][:,:,:,sub_id],hrtf_pred=hrtf_pred,idx_plot_list=np.array(config["idx_plot_list"][db_name]), config=config, figdir=f"{config['artifacts_dir']}/figure/HRTF/", fname=f"{config['artifacts_dir']}/figure/HRTF/HRTF_Mag_{db_name}-sub-{sub_id+1}_{mode}")

                        mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude', top_db = 80) 
                        plot_mag_Angle_vs_Freq(HRTF_mag=mag2db(th.abs(hrtf_pred)), SrcPos=srcpos[db_name][:,:,sub_id], fs=config['max_frequency'], figdir=f"{config['artifacts_dir']}/figure/HRTF_mag_2D/", fname=f"{config['artifacts_dir']}/figure/HRTF_mag_2D/HRTF_Mag_{db_name}-sub-{sub_id+1}_{mode}")
             
                        ## plot HRIRs
                        if config["minphase_recon"]:
                            fs = config['max_frequency']*2
                            f_us = config['fs_upsampling']
                            
                            hrtf_pred = hrtf_pred.permute(3,0,1,2) # (B,2,L,S) -> (S,B,2,L)
                            hrtf_sub = hrtf_sub.permute(3,0,1,2) # (B,2,L,S) -> (S,B,2,L)
            
                            hrir_sub = posTF2IR_dim4(hrtf_sub)
                            if config["use_itd_gt"]:
                                itd_des = hrir2itd(hrir=hrir_sub, fs=fs, f_us=f_us).to(hrir_sub.device)
                            else:
                                raise NotImplementedError()
                            
                            phase_min, hrir_min = minphase_recon(tf=hrtf_pred)
                            itd_ori = hrir2itd(hrir=hrir_min, fs=fs, f_us=f_us).to(hrir_sub.device)
                            hrir_min_itd = assign_itd(hrir_ori=hrir_min, itd_ori=itd_ori, itd_des=itd_des, fs=fs)

                            hrir_pred_min = hrir_min.permute(1,2,3,0)
                            hrir_pred = hrir_min_itd.permute(1,2,3,0) # (B,2,L,S)
                            hrir_sub = hrir_sub.permute(1,2,3,0) # (B,2,L,S)
                            hrtf_pred = hrtf_pred.permute(1,2,3,0) # (B,2,L,S)
                            hrtf_sub = hrtf_sub.permute(1,2,3,0) # (B,2,L,S)

                            _, _ = plothrir(srcpos=srcpos[db_name][:,:,sub_id],hrtf_gt=None,hrtf_pred=None,idx_plot_list=np.array(config["idx_plot_list"][db_name]), config=config, hrir_gt=hrir_sub.squeeze(), hrir_pred=hrir_pred, figdir=f"{config['artifacts_dir']}/figure/HRIR/", fname=f"{config['artifacts_dir']}/figure/HRIR/HRIR_{db_name}-sub-{sub_id+1}_{mode}")
                        else:
                            hrir_sub, hrir_pred = plothrir(srcpos=srcpos[db_name][:,:,sub_id],hrtf_gt=hrtf_gt[db_name][:,:,:,sub_id],hrtf_pred=hrtf_pred,idx_plot_list=np.array(config["idx_plot_list"][db_name]), config=config, figdir=f"{config['artifacts_dir']}/figure/HRIR/", fname=f"{config['artifacts_dir']}/figure/HRIR/HRIR_{db_name}-sub-{sub_id+1}_{mode}")
                        
                        # sum([len(config['sub_index'][db_name][mode_str]) for db_name in config['database']])

                        hrirs_all_list = ['hrir_gt_all','hrir_pred_all','hrir_pred_min_all']
                        hrirs_list = ['hrir_sub', 'hrir_pred', 'hrir_pred_min']
                        for hrirs_all, hrirs in zip(hrirs_all_list, hrirs_list):
                            if not db_name in eval(hrirs_all).keys():
                                eval(hrirs_all)[db_name] = th.zeros(eval(hrirs).shape[0],eval(hrirs).shape[1],eval(hrirs).shape[2],len(config['sub_index'][db_name][mode_str]))
                            if eval(hrirs).dim()==4:
                                eval(hrirs_all)[db_name][:,:,:,sub_id] = eval(hrirs)[:,:,:,0]
                            else:
                                eval(hrirs_all)[db_name][:,:,:,sub_id] = eval(hrirs)
                        # if not db_name in hrir_gt_all.keys():
                        #     hrir_gt_all[db_name] = th.zeros(hrir_pred.shape[0],hrir_pred.shape[1],hrir_pred.shape[2],len(config['sub_index'][db_name][mode_str]))
                        # if not db_name in hrir_pred_all.keys():
                        #     hrir_pred_all[db_name] = th.zeros(hrir_pred.shape[0],hrir_pred.shape[1],hrir_pred.shape[2],len(config['sub_index'][db_name][mode_str]))
                        # if not db_name in hrir_pred_min_all.keys():
                        #     hrir_pred_min_all[db_name] = th.zeros(hrir_pred_min.shape[0],hrir_pred_min.shape[1],hrir_pred_min.shape[2],len(config['sub_index'][db_name][mode_str]))
                        if not db_name in lsd_bm_all.keys():
                            lsd_bm_all[db_name] = th.zeros(hrir_pred.shape[0],hrir_pred.shape[1],len(config['sub_index'][db_name][mode_str])) # B,2,S


                        # for h, hrirs in enumerate(['hrir_sub', 'hrir_pred', 'hrir_pred_min']):
                        #     if eval(hrirs).dim()==4:
                        #         eval(hrirs_all_list[h])[db_name][:,:,:,sub_id] = eval(hrirs)[:,:,:,0]
                        #     else:
                        #         eval(hrirs_all_list[h])[db_name][:,:,:,sub_id] = eval(hrirs)

                        # if hrir_sub.dim()==4:
                        #     hrir_gt_all[db_name][:,:,:,sub_id] = hrir_sub[:,:,:,0]
                        # else:
                        #     hrir_gt_all[db_name][:,:,:,sub_id] = hrir_sub
                        # if hrir_pred.dim()==4:
                        #     hrir_pred_all[db_name][:,:,:,sub_id] = hrir_pred[:,:,:,0]
                        # else:
                        #     hrir_pred_all[db_name][:,:,:,sub_id] = hrir_pred
                        # if hrir_pred_min.dim()==4:
                        #     hrir_pred_min_all[db_name][:,:,:,sub_id] = hrir_pred_min[:,:,:,0]
                        # else:
                        #     hrir_pred_min_all[db_name][:,:,:,sub_id] = hrir_pred_min

                        # plot LSD (before mean)
                        fig_dir_lsd_bm = f'{config["artifacts_dir"]}/figure/LSD/'
                        os.makedirs(fig_dir_lsd_bm, exist_ok=True)
                        lsd_bm = lsd_bm_loss(hrtf_pred, hrtf_sub).squeeze().cpu() # (B,2)

                        emphasize_mes_pos = True if "idx_mes_pos" in prediction else False
                            
                        for ch in range(2):
                            ch_str_l = ['left', 'right']
                            ch_str = ch_str_l[ch]
                            plotazimzeni(pos=srcpos[db_name][:,:,sub_id].cpu(),c=lsd_bm[:,ch],fname=f'{fig_dir_lsd_bm}lsd_{db_name}-sub-{sub_id+1}_{ch_str}_{mode}',title=f'{db_name} sub:{sub_id+1}, {ch_str}',cblabel=f'LSD (dB)',cmap='gist_heat',figsize=(10.5,5),dpi=300,emphasize_mes_pos=emphasize_mes_pos, idx_mes_pos=prediction["idx_mes_pos"],vmin=0, vmax=10)
                        # store LSD (before mean)
                        lsd_bm_all[db_name][:,:,sub_id] = lsd_bm

                        if sub_id == config['sub_index'][db_name][mode_str][-1] - config['sub_index'][db_name][mode_str][0]:
                            hrir_dir = f'{config["artifacts_dir"]}/HRIR'
                            lsd_bm_dir = f'{config["artifacts_dir"]}/LSD'
                            os.makedirs(hrir_dir, exist_ok=True)
                            os.makedirs(lsd_bm_dir, exist_ok=True)
                            th.save(hrir_pred_all[db_name], f'{hrir_dir}/hrir_pred_{mode}_{db_name}.pt')
                            th.save(hrir_pred_min_all[db_name], f'{hrir_dir}/hrir_pred_min_{mode}_{db_name}.pt')
                            th.save(hrir_gt_all[db_name], f'{hrir_dir}/hrir_gt_{db_name}.pt')
                            th.save(lsd_bm_all[db_name], f'{lsd_bm_dir}/lsd_{mode}_{db_name}_before_mean.pt')
                        if config["use_own_hrtf_test"] and len(subject_list)==1:
                            srcpos_sub_np =  srcpos_sub.detach().clone().squeeze() # B,3 # [rad, azim, zeni]
                            #     radius, azimuth in [0,2*pi), zenith in [0,pi]
                            # ->  azimuth in [0,360), elevation in [-90,90], radius
                            srcpos_sub_np[:,1] = srcpos_sub_np[:,1]/np.pi*180 # azimuth in [0,360)
                            srcpos_sub_np[:,2] = 90 - srcpos_sub_np[:,2]/np.pi*180  # elevation in [-90,90]
                            srcpos_sub_np = th.cat((srcpos_sub_np[:,1].unsqueeze(1), srcpos_sub_np[:,2].unsqueeze(1), srcpos_sub_np[:,0].unsqueeze(1)),dim=1)
                            srcpos_sub_np =  srcpos_sub_np.to('cpu').detach().numpy().copy()
                            hrir_pred_np = hrir_pred.squeeze().to('cpu').detach().to(th.double).numpy().copy()

                            mat_dir = config["artifacts_dir"] + "/mat"
                            os.makedirs(mat_dir, exist_ok=True)
                            scipy.io.savemat(f'{mat_dir}/hrir_pred_{mode}.mat', {'IR':hrir_pred_np, 'SourcePosition':srcpos_sub_np, 'SamplingRate':config['max_frequency']*2})

                            hrir_gt_np = hrir_sub.squeeze().to('cpu').detach().to(th.double).numpy().copy()
                            scipy.io.savemat(f'{mat_dir}/hrir_gt.mat', {'IR':hrir_gt_np, 'SourcePosition':srcpos_sub_np, 'SamplingRate':config['max_frequency']*2})

                            scipy.io.savemat(f'{mat_dir}/lsd_{mode}_before_mean.mat', {'LSD':lsd_bm.squeeze().to('cpu').detach().to(th.double).numpy().copy()})

            if config["loss_weights"]["l2_rec"] == 0 and config["loss_weights"]["lsd"] == 0:
                loss = returns["lsd"]
            else:
                loss = returns["l2_rec"] * config["loss_weights"]["l2_rec"] +  returns["lsd"] * config["loss_weights"]["lsd"]

            
            returns_new = {
                "loss": loss.detach().clone(),
            }
            returns.update(returns_new)
            returns.update(returns_zi)
            
            for k in returns:
                returns[k] /= len(subject_list)
        
        elif config["model"] == "Lemaire05":
            returns = {}
            subject_list = np.arange(0,srcpos.shape[-1])
            l2_loss = MSE()
            l2_loss_angle = L2Loss_angle()
            lsd_loss = LSD()
            l2_rec = 0
            lsd_rec = 0
            for sub_id in subject_list:
                hrtf_sub = hrtf_gt[db_name][:,:,:,sub_id:sub_id+1]
                srcpos_sub = srcpos[db_name][:,:,sub_id:sub_id+1]

                hrtf_gt_l = hrtf_sub[:,0,:,:]
                hrtf_gt_r = hrtf_sub[:,1,:,:]
                mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude', top_db = 80) 
                hrtf_logmag_l = mag2db(th.abs(hrtf_gt_l))
                hrtf_logmag_r = mag2db(th.abs(hrtf_gt_r))
                input = th.cat((hrtf_logmag_l,hrtf_logmag_r), dim=1)
                # print(input.shape)
                prediction = net.forward(input=input, srcpos=srcpos_sub)
                hrtf_pred = prediction["output"].detach().clone()
                l2_rec += l2_loss(hrtf_pred, hrtf_sub)
                lsd_rec += lsd_loss(hrtf_pred, hrtf_sub)

                if flg_save:
                    # Visualise
                    plotmaghrtf(srcpos=srcpos[db_name][:,:,sub_id],hrtf_gt=hrtf_gt[db_name][:,:,:,sub_id],hrtf_pred=hrtf_pred,idx_plot_list=np.array(config["idx_plot_list"][db_name]), config=config, figdir=f"{config['artifacts_dir']}/figure/HRTF/", fname=f"{config['artifacts_dir']}/figure/HRTF/HRTF_Mag_{db_name}-sub-{sub_id+1}_{mode}")
                    hrir_gt, hrir_pred = plothrir(srcpos=srcpos[db_name][:,:,sub_id],hrtf_gt=hrtf_gt[db_name][:,:,:,sub_id],hrtf_pred=hrtf_pred,idx_plot_list=np.array(config["idx_plot_list"][db_name]), config=config, figdir=f"{config['artifacts_dir']}/figure/HRIR/", fname=f"{config['artifacts_dir']}/figure/HRIR/HRIR_{db_name}-sub-{sub_id+1}_{mode}")
                    if sub_id == 0:
                        hrir_gt_all = th.zeros(hrir_pred.shape[0],hrir_pred.shape[1],hrir_pred.shape[2],len(subject_list))
                        hrir_pred_all = th.zeros(hrir_pred.shape[0],hrir_pred.shape[1],hrir_pred.shape[2],len(subject_list))
                    hrir_gt_all[:,:,:,sub_id] = hrir_gt
                    hrir_pred_all[:,:,:,sub_id] = hrir_pred
                    if sub_id == subject_list[-1]:
                        hrir_dir = f'{config["artifacts_dir"]}/HRIR'
                        os.makedirs(hrir_dir, exist_ok=True)
                        th.save(hrir_pred_all, f'{hrir_dir}/hrir_pred_{mode}.pt')
                        th.save(hrir_gt_all, f'{hrir_dir}/hrir_gt.pt')
            loss = lsd_rec
            returns_new = {
                "l2_rec": l2_rec.detach().clone(), 
                "lsd": lsd_rec.detach().clone(),
                "loss": loss.detach().clone(),
            }
            returns.update(returns_new)
            for k in returns:
                returns[k] /= len(subject_list)
        
        else:
            bs = 128
            num_b = math.ceil(srcpos.shape[0] / bs)
            if use_coeff:
                coeff=th.tile(coeff[0,:,:,:].view(1,2,-1,coeff.shape[-1]),(srcpos.shape[0],1,1,1))
                # print("coeff[0,0,:,:]=")
                # print(coeff[0,0,:,:])
            helmholtz = 0
            for itr_b in range(num_b):
                itr_start = bs * itr_b
                itr_end = min(bs * (itr_b+1), srcpos.shape[0])
                # print(itr_start)
                # print(itr_end)
                if use_coeff:
                    coeff_b = coeff[itr_start:itr_end].cuda()
                else:
                    coeff_b = None
                prediction = net.forward(srcpos[itr_start:itr_end], use_cuda_forward = use_cuda_forward, use_coeff = use_coeff, coeff=coeff_b)
                if 'hrtf_pred' not in locals():
                    # print(srcpos[itr_start:itr_end].device)
                    hrtf_pred = prediction["output"].detach().clone()
                else:
                    hrtf_pred = th.cat((hrtf_pred, prediction["output"].detach().clone()), dim=0)
                # print(hrtf_pred.device)
                if config["model"] == "PINN":
                    helmholtz += HelmholtzLoss()(net, net.f_th, srcpos[itr_start:itr_end]).squeeze().detach()

                # print(hrtf_gt[0,:10])
                # print(hrtf_pred[0,:10])
            l2 = MSE()(hrtf_pred, hrtf_gt)
            reg = RegLoss()(prediction["coeff"],SpecialFunc(maxto=config["max_truncation_order"]).n_vec)
            lsd = LSD()(hrtf_pred, hrtf_gt)
            loss = l2 * config["loss_weights"]["l2"] + reg * config["loss_weights"]["reg"] + lsd * config["loss_weights"]["lsd"]
            # loss = lsd
            print(f"l2:  {l2:.4}")
            print(f"Reg:  {reg:.4}")
            print(f"LSD:  {lsd:.4}")
            returns = {
                    "l2": l2.detach().clone(), 
                    "reg": reg.detach().clone(), 
                    "lsd": lsd.detach().clone(), 
            }
            if config["model"] == "PINN":
                loss += helmholtz * config["loss_weights"]["helmholtz"]
                print(f"Helmholtz:  {helmholtz:.4}")
                returns["helmholtz"] = helmholtz.detach().clone()
    loss_str = "    ".join([f"{k}:{returns[k]:.4}" for k in sorted(returns.keys())])
    print(loss_str)

    
    if returns["loss"] < best_loss:
        flg_save = True
        print("Best Loss.")
    if flg_save: # Visualise
        print("Saving figures...")
        if config["DNN_for_interp_ITD"]:
            fig_dir_itd = f'{config["artifacts_dir"]}/figure/ITD/'
            os.makedirs(fig_dir_itd, exist_ok=True)

            sub_id = 0
            plotazimzeni(pos=srcpos[db_name][:,:,sub_id].cpu(),c=itd_pred_all[db_name][:,sub_id].cpu().detach()*1000,fname=f'{fig_dir_itd}itd_{mode}_{db_name}-sub-{sub_id+1}',title=f'{db_name} sub:{sub_id+1}',cblabel=f'ITD (ms)',cmap='bwr',figsize=(10.5,5),dpi=300,emphasize_mes_pos=True, idx_mes_pos=prediction["idx_mes_pos"])

            plotazimzeni(pos=srcpos[db_name][:,:,sub_id].cpu(),c=th.abs(itd_pred_all[db_name][:,sub_id]-itd_gt_all[db_name][:,sub_id]).cpu().detach()*1000,fname=f'{fig_dir_itd}itd_ae_{mode}_{db_name}-sub-{sub_id+1}',title=f'{db_name} sub:{sub_id+1}',cblabel=f'AE of ITD (ms)',cmap='gist_heat',figsize=(10.5,5),dpi=300,emphasize_mes_pos=True, idx_mes_pos=prediction["idx_mes_pos"])
        else:
            if config["model"] in ["AE", "CAE", "HCAE", "AE_PINN", "CNN", "FIAE"]:
                pass
                # plotmaghrtf(srcpos=srcpos[db_name][:,:,-1],hrtf_gt=hrtf_gt[db_name][:,:,:,-1],hrtf_pred=hrtf_pred,idx_plot_list=np.array(config["idx_plot_list"][db_name]), config=config, figdir=f"{config['artifacts_dir']}/figure/HRTF/", fname=f"{config['artifacts_dir']}/figure/HRTF/HRTF_Mag_{db_name}-sub-{sub_id+1}_{mode}")
                # plotangle(gt=srcpos[db_name][:,1,-1],pred=phi_pred,mode=mode, title='Azimuth(phi)')
                # plotangle(gt=srcpos[db_name][:,2,-1],pred=theta_pred,mode=mode, title='Zenith(theta)')
            elif config["model"] == "Lemaire05":
                plotmaghrtf(srcpos=srcpos[db_name][:,:,-1],hrtf_gt=hrtf_gt[db_name][:,:,:,-1],hrtf_pred=hrtf_pred,idx_plot_list=np.array(config["idx_plot_list"][db_name]), config=config, figdir=f"{config['artifacts_dir']}/figure/HRTF/", fname=f"{config['artifacts_dir']}/figure/HRTF/HRTF_Mag_{db_name}-sub-{sub_id+1}_{mode}")
            else:
                plotmaghrtf(srcpos=srcpos,hrtf_gt=hrtf_gt,hrtf_pred=hrtf_pred, config=config, figdir=f"{config['artifacts_dir']}/figure/HRTF/", fname=f"{config['artifacts_dir']}/figure/HRTF/HRTF_Mag_{db_name}-sub-{sub_id+1}_{mode}") 
    return returns

def plotangle(gt,pred,mode, title):
    plt.figure(figsize=(6,6))
    pred = pred.to('cpu').detach().numpy().copy() * 180/np.pi
    gt = gt.to('cpu').detach().numpy().copy() * 180/np.pi
    plt.scatter(gt,pred,marker='x')
    plt.xlabel("Target Angle [deg]")
    plt.ylabel("Predicted Angle [deg]")
    x = np.linspace(0,360,10)
    plt.plot(x,x,color='black',linestyle=':')
    plt.plot(x,x+45,color='black',linestyle=':')
    plt.plot(x,x-45,color='black',linestyle=':')
    plt.plot(x,x-360+45,color='black',linestyle=':')
    plt.plot(x,x+360-45,color='black',linestyle=':')
    if title.startswith('Azimuth'):
        lim = [0,360]
    elif title.startswith('Zenith'):
        lim = [0,180]
    plt.xlim(lim)
    plt.ylim(lim)
    plt.title(title)
    figure_dir = config["artifacts_dir"] + "/figure/angle/"
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(figure_dir + "Angle_" + title + "_" +mode+".png", dpi=300)
    # plt.savefig(figure_dir + "Angle_" + title + "_" +mode+".jpg", dpi=300)
    plt.close()

def objective_AE(trial):
    config["epochs"] = 250
    config_loss_weights_new = {
        'cross_entropy': 10**(-1*trial.suggest_int('lw_ce', 0, 2, step=1, log=False)),
        'l2_c': 10**(-1*trial.suggest_int('lw_l2_c', 2, 4, step=1, log=False)),
        'l2_rec': 10**(-1*trial.suggest_int('lw_l2_rec', 2, 4, step=1, log=False)),
        'l2_reg': 10**(-1*trial.suggest_int('lw_l2_reg', 4, 7, step=1, log=False)),
        'lsd': 10**(-1*trial.suggest_int('lw_l2_lsd', 0, 2, step=1, log=False)),
        'lsd_c': 10**(-1*trial.suggest_int('lw_l2_lsd_c', 0, 2, step=1, log=False)),
        'lsd_diff': 10**(-1*trial.suggest_int('lw_lsd_diff', 0, 2, step=1, log=False)),
        'reg': 10**(-1*trial.suggest_int('lw_reg', 5, 7, step=1, log=False)),
        'var_z': 10**(-1*trial.suggest_int('lw_var_z', 0, 2, step=1, log=False))
    }
    config["loss_weights"].update(config_loss_weights_new)

    config['metric_margin'] = trial.suggest_uniform('metric_margin', 0.0, 1.0)

    std_dim_str = trial.suggest_categorical('std_dim', ['13','23','123'])
    if std_dim_str == '13':
        config["std_dim"] = [1,3]
    elif std_dim_str == '23':
        config["std_dim"] = [2,3]
    else:
        config["std_dim"] = [1,2,3]
    c_dim = config["std_dim"]
    coeff_train_std = coeff_train
    eps = 1e-10*th.ones(coeff_train_std.shape)
    coeff_mag_log10 = th.log10(th.max(th.abs(coeff_train_std),eps))
    coeff_std = th.std(coeff_mag_log10, dim=c_dim,unbiased=True)
    coeff_mean = th.mean(coeff_mag_log10, dim=c_dim)
    
    dt_now = datetime.datetime.now() + datetime.timedelta(hours=9)
    timestamp = dt_now.strftime('_%m%d_%H%M%S%f')
    config["timestamp"] = timestamp
    net = HRTFApproxNetwork_AE(config=config, c_std = coeff_std, c_mean = coeff_mean)

    trainer = Trainer(config, net, traindataset)
    study.trials_dataframe().to_csv(config["artifacts_dir"] + "/study_history.csv")
    
    results = train_optuna(trainer=trainer)
    os.remove(config["artifacts_dir"]+'/hrtf_approx_network.newbob' + timestamp +'.net')
    
    return results["best_loss_valid"]

def objective_AE_old(trial):
    config["epochs"] = 30
    config["channel_En"] = 2**(trial.suggest_int('channel_En', 6, 9, step=1, log=False))
    config["channel_En_z"] = 2**(trial.suggest_int('channel_En_z', 6, 9, step=1, log=False))
    config["channel_En_v"] = 0
    config["channel_De_z"] = 2**(trial.suggest_int('channel_De_z', 6, 9, step=1, log=False))
    config["dim_z"] = 2**(trial.suggest_int('dim_z', 2, 7, step=1, log=False))
    config["hlayers_En"] = trial.suggest_int('hlayers_En', 0, 6, step=2, log=False)
    config["hlayers_En_z"] = trial.suggest_int('hlayers_En_z', 0, 6, step=2, log=False)
    config["hlayers_En_v"] = 0
    config["hlayers_De_z"] = trial.suggest_int('hlayers_De_z', 0, 6, step=2, log=False)
    config["droprate"] = 0.1*trial.suggest_int('droprate', 0, 3, step=1, log=False)
    config["learning_rate"] = 10**(-1*trial.suggest_int('learning_rate', 2, 5, step=1, log=False))
    config["loss_weights"]["reg"] = 10**(-1*trial.suggest_int('reg', 2, 7, step=1, log=False))
    config["loss_weights"]["lsd"] = 10**(-1*trial.suggest_int('lsd', 0, 4, step=1, log=False))
    config["loss_weights"]["l2_rec"] = 10**(-1*trial.suggest_int('l2_rec', 0, 4, step=1, log=False))
    config["loss_weights"]["cossim"] = 0
    config["loss_weights"]["lsd_c"] = 10**(-1*trial.suggest_int('l2_c', 0, 4, step=1, log=False))
    config["loss_weights"]["l2_c"] = 10**(-1*trial.suggest_int('l2_c', 0, 4, step=1, log=False))
    config["loss_weights"]["huber_c"] = 10**(-1*trial.suggest_int('l2_c', 0, 4, step=1, log=False))
    config["loss_weights"]["var_z"] = 10**(-1*trial.suggest_int('var_z', 0, 4, step=1, log=False))

    dt_now = datetime.datetime.now() + datetime.timedelta(hours=9)
    timestamp = dt_now.strftime('_%m%d_%H%M%S%f')
    config["timestamp"] = timestamp
    net = HRTFApproxNetwork_AE(config = config)
    trainer = Trainer(config, net, traindataset)
    results = train_optuna(trainer=trainer)
    os.remove(config["artifacts_dir"]+'/hrtf_approx_network.newbob' + timestamp +'.net')
    return results["best_loss_valid"]

def objective_CAE(trial):
    config["epochs"] = 30

    config["channel_En"] = 2**(trial.suggest_int('channel_En', 6, 9, step=1, log=False))
    # config["channel_En_z"] = 2**(trial.suggest_int('channel_En_z', 5, 9, step=1, log=False))
    config["channel_En_z"] = config["channel_En"]
    # config["channel_En_v"] = 2**(trial.suggest_int('channel_En_v', 5, 7, step=1, log=False))
    config["channel_En_v"] = config["channel_En"]
    # config["channel_De_z"] = 2**(trial.suggest_int('channel_De_z', 5, 9, step=1, log=False))
    config["channel_De_z"] = config["channel_En"]

    config["hlayers_En"] = trial.suggest_int('hlayers_En', 0, 4, step=1, log=False)

    # config["hlayers_En_z"] = trial.suggest_int('hlayers_En_z', 0, 2, step=1, log=False)
    config["hlayers_En_z"] = 0

    # config["hlayers_En_v"] = trial.suggest_int('hlayers_En_v', 0, 6, step=2, log=False)
    config["hlayers_En_v"] = 0
    config["kernel_size"] = trial.suggest_int('kernel_size', 2, 8, step=2, log=False)

    # config["droprate"] = 0.1*trial.suggest_int('droprate', 0, 10, step=1, log=False)
    config["droprate"] = 0.0

    config["learning_rate"] = 10**(-1*trial.suggest_int('learning_rate', 2, 5, step=1, log=False))
    config["loss_weights"]["reg"] = 10**(-1*trial.suggest_int('reg', 2, 7, step=1, log=False))
    config["loss_weights"]["lsd"] = 10**(-1*trial.suggest_int('lsd', 0, 4, step=1, log=False))
    config["loss_weights"]["l2_rec"] = 10**(-1*trial.suggest_int('l2_rec', 0, 4, step=1, log=False))
    # config["loss_weights"]["cossim"] = 10**(-1*trial.suggest_int('cossim', 0, 4, step=1, log=False))
    config["loss_weights"]["cossim"] = 0.0
    config["loss_weights"]["l2_c"] = 10**(-1*trial.suggest_int('l2_c', 0, 4, step=1, log=False))
    config["loss_weights"]["var_z"] = 10**(-1*trial.suggest_int('var_z', 0, 4, step=1, log=False))

    dt_now = datetime.datetime.now() + datetime.timedelta(hours=9)
    timestamp = dt_now.strftime('_%m%d_%H%M_%S_') + dt_now.strftime('%f')[:2]
    config["timestamp"] = timestamp
    net = HRTFApproxNetwork_CAE(config = config)
    print(net)
    trainer = Trainer(config, traindataset)
    results = train_optuna(net=net, trainer=trainer)
    os.remove(config["artifacts_dir"]+'/hrtf_approx_network.newbob' + timestamp +'.net')
    return results["best_loss_valid"]

def objective_AE_PINN(trial):
    config["epochs"] = 30

    # config["channel_En"] = 2**(trial.suggest_int('channel_En', 6, 8, step=1, log=False))
    # config["channel_En_z"] = 2**(trial.suggest_int('channel_En_z', 6, 8, step=1, log=False))
    # config["channel_En_v"] = 2**(trial.suggest_int('channel_En_v', 6, 7, step=1, log=False))
    
    config["hlayers_En"] = trial.suggest_int('hlayers_En', 0, 2, step=1, log=False)
    config["hlayers_En_z"] = trial.suggest_int('hlayers_En_z', 0, 4, step=2, log=False)
    config["hlayers_En_v"] = trial.suggest_int('hlayers_En_v', 0, 4, step=2, log=False)

    config["dim_z"] = 2**(trial.suggest_int('dim_z', 3, 6, step=1, log=False))
    
    config["droprate"] = 0.1*trial.suggest_int('droprate', 0, 3, step=1, log=False)
    config["learning_rate"] = 10**(-1*trial.suggest_int('learning_rate', 2, 5, step=1, log=False))
    
    config["loss_weights"]["lsd"] = 10**(-1*trial.suggest_int('lsd', 0, 4, step=1, log=False))
    config["loss_weights"]["l2_rec"] = 10**(-1*trial.suggest_int('l2_rec', 0, 4, step=1, log=False))
    config["loss_weights"]["cossim"] = 10**(-1*trial.suggest_int('cossim', 0, 4, step=1, log=False))
    config["loss_weights"]["var_z"] = 10**(-1*trial.suggest_int('var_z', 0, 4, step=1, log=False))
    config["loss_weights"]["helmholtz"] = 10**(-1*trial.suggest_int('helmholtz', 0, 8, step=1, log=False))

    dt_now = datetime.datetime.now() + datetime.timedelta(hours=9)
    timestamp = dt_now.strftime('_%m%d_%H%M_%S_') + dt_now.strftime('%f')[:2]
    config["timestamp"] = timestamp
    net = HRTFApproxNetwork_AE_PINN(config = config)
    print(net)
    trainer = Trainer(config, net, traindataset)
    results = train_optuna(trainer=trainer)
    os.remove(config["artifacts_dir"]+'/hrtf_approx_network.newbob' + timestamp +'.net')
    return results["best_loss_valid"]

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
    # print(config["model"])
    if args.lininv:
        net_pre = HRTFApproxNetwork(config=config)
    elif config["model"] == "SWFE":
        net = HRTFApproxNetwork(config=config,
                        maxto = config["max_truncation_order"],
                        fft_length = config["fft_length"],
                        max_f = config["max_frequency"],
                        channel = config["channel"],
                        layers = config["layers"],
                        droprate = config["droprate"],
                        use_freq_as_input = config["use_freq_as_input"]
                        )
    elif config["model"] == "PINN":
        net = HRTFApproxNetwork_PINN(maxto = config["max_truncation_order"],
                        fft_length = config["fft_length"],
                        max_f = config["max_frequency"],
                        channel = config["channel"],
                        layers = config["layers"],
                        droprate = config["droprate"],
                        )
    elif config["model"] == "AE":
        net_pre = HRTFApproxNetwork(config=config)
        # print(config)
        if not config["lininv_only"]:
            net = HRTFApproxNetwork_AE(config=config)
    elif config["model"] == "CAE":
        net_pre = HRTFApproxNetwork(config=config)
        net = HRTFApproxNetwork_CAE(config=config)
    elif config["model"] == "HCAE":
        net_pre = HRTFApproxNetwork(config=config)
        net = HRTFApproxNetwork_HyperCAE(config=config)
    elif config["model"] == "AE_PINN":
        net = HRTFApproxNetwork_AE_PINN(config=config)
    elif config["model"] == "CNN":
        net_pre = HRTFApproxNetwork(config=config)
        net = HRTFApproxNetwork_CNN(config=config)
    elif config["model"] == "Lemaire05":
        net = HRTFApproxNetwork_Lemaire05(config=config)
    elif config["model"] == "FIAE":
        net = HRTFApproxNetwork_FIAE(config=config)
    
    # print(config["model"])
    # sys.exit()
        
    
    print("=====================")
    print("config: ")
    pprint.pprint(config)
    print("=====================")

    if args.load and args.model_file != '': # test or continue train
        net.load_from_file(args.model_file)
    
    if not config["activation_function"].startswith('nn.ReLU'):
        replace_activation_function(net, act_func_new=eval(config["activation_function"]))

    if not args.lininv and not args.optuna and not config["lininv_only"]:
        print(net)
        print("=====================")

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

    dataset = HRTFDataset(config=config,filter_length=round(config["fft_length"]/2),
                                    max_f = config["max_frequency"],
                                    sofa_path=args.dataset_directory,
                                    debug = args.debug)
    traindataset_all = dataset.trainitem()
    validdataset = dataset.validitem()
    testdataset = dataset.testitem()
    config["Table"] = dataset.Table
    config["Val_Standardize"] = dataset.Val_Standardize

    with open('LAP_idx_mes_pos.pkl', 'rb') as f:
        data_idx_mes_pos = pickle.load(f)
    config['lap_smp_idx'] = data_idx_mes_pos

    if config["model"] in ["AE", "CAE", "HCAE", "AE_PINN", "CNN", "Lemaire05", "FIAE"]:
        traindataset = dataset
        ptins = "debug_" if args.debug else ""
        if config["model"] in ["AE", "CAE", "HCAE", "CNN"]:
            # reg_w = config["loss_weights"]["reg"]
            # reg_w = 1e-7
            reg_w = config["reg_w"]
            green = '' if config["green"] else '_nogreen'
            if config["lininv_only"] and config["balanced"]:
                maxto = config["t_des"]
            else:
                maxto = config["max_truncation_order"]

            if config["underdet"] or config["balanced"]:
                num_pts = f'_{round((config["t_des"]+1)**2)}pts'
            else:
                num_pts = ''
            ptname = f'coeff_{ptins}{config["max_frequency"]}_{round(config["fft_length"]/2)}_{maxto}_{reg_w:.0e}{num_pts}{green}.pt'
            
            # ptpath = config["artifacts_dir"] + "/" + ptname
            ptpath = "outputs/" + ptname
            if os.path.isfile(ptpath) and not args.lininv and not config["lininv_experiment"]:
                print("Load Coeff. from " + ptpath)
                coeff_train = th.load(ptpath)
                # print(th.max(th.log10(th.abs(coeff_train)))) # tensor(2.8599)
            else:
                if config["lininv_experiment"]:
                    print("================")
                    print("Experiment mode.")
                    print("================")
                    t_list = np.arange(1,18+1)
                    # t_list = np.arange(2,3)
                    

                    for t in t_list:
                        if t == 1: # dammy
                            num_pts = f'_{440}pts'
                            mode_l = [2]
                        else:
                            config["t_des"] = t
                            num_pts = f'_{round((config["t_des"]+1)**2)}pts'
                            mode_l = [0,1]
                        bs = 1
                        for mode in mode_l:
                            print("=============================")
                            if mode == 0:
                                maxto = config["t_des"]
                            else:
                                maxto = config["max_truncation_order"]
                            if mode==0:
                                config["balanced"] = True
                                config["underdet"] = False
                            elif mode==1:
                                config["balanced"] = False
                                config["underdet"] = True
                            else:
                                config["balanced"] = False
                                config["underdet"] = False
                            print(f"num_pts: {num_pts}, maxto: {maxto}")
                            print("=============================")
                    
                            ptname = f'coeff_{ptins}{config["max_frequency"]}_{round(config["fft_length"]/2)}_{maxto}_{reg_w:.0e}{num_pts}_all.pt'
                            ptname_hrtf = f'hrtf_{ptins}{config["max_frequency"]}_{round(config["fft_length"]/2)}_{maxto}_{reg_w:.0e}{num_pts}_all.pt'
                            ptname_loss = f'loss_{ptins}{config["max_frequency"]}_{round(config["fft_length"]/2)}_{maxto}_{reg_w:.0e}{num_pts}_all.pt'
                            ptpath = config["artifacts_dir"] + "/" + ptname
                            ptpath_hrtf = config["artifacts_dir"] + "/" + ptname_hrtf
                            ptpath_loss = config["artifacts_dir"] + "/" + ptname_loss

                            net_pre = HRTFApproxNetwork(config=config)
                            trainer = Trainer(config, net_pre, traindataset)
                            coeff_all = th.zeros((2, (maxto+1)**2, 128, 94),dtype=th.complex64) 
                            hrtf_all = th.zeros((440, 2, 128, 94),dtype=th.complex64) 
                            loss_all = th.zeros((4,94))

                            # traindata, validdata, testdata
                            sub = 1
                            while sub <= 94:
                                print("------")
                                t_start = time.time()
                                print(f"subject:{sub}")

                                if sub <= 77:
                                    srcpos, hrtf_gt = traindataset_all[0][:,:,sub-1], traindataset_all[1][:,:,:,sub-1]
                                elif sub <= 87:
                                    srcpos, hrtf_gt = validdataset[0][:,:,sub-77-1], validdataset[1][:,:,:,sub-77-1]
                                else:
                                    srcpos, hrtf_gt = testdataset[0][:,:,sub-87-1], testdataset[1][:,:,:,sub-87-1]

                                srcpos, hrtf_gt = srcpos.squeeze(), hrtf_gt.squeeze()
                                returns = net_pre.forward(srcpos, use_cuda_forward=False, use_lininv=True, hrtf=hrtf_gt)
                                hrtf_pred, coeff = returns["output"], returns["coeff"]
                                plotmaghrtf(srcpos=srcpos,hrtf_gt=hrtf_gt,hrtf_pred=hrtf_pred,mode=f'train_sub-{sub}_{maxto}{num_pts}', 
                                idx_plot_list=np.array([202,211,220,229]), config=config, figdir=f"{config['artifacts_dir']}/figure/HRTF/", fname=f"{config['artifacts_dir']}/figure/HRTF/HRTF_Mag_train_sub-{sub}_{maxto}{num_pts}") 
                                
                                hrtf_all[:,:,:,sub-1] = hrtf_pred
                                coeff_all[:,:,:,sub-1] = coeff[0,:,:,:]
                                l2 = MSE()(hrtf_pred, hrtf_gt)
                                lsd = LSD()(hrtf_pred, hrtf_gt)
                                reg = RegLoss()(coeff[0,:,:,:].unsqueeze(-1), SpecialFunc(maxto=maxto).n_vec)
                                loss = l2 * config["loss_weights"]["l2"] + reg * config["loss_weights"]["reg"] + lsd * config["loss_weights"]["lsd"]
                                loss_all[:,sub-1] = th.tensor([l2,lsd,reg,loss])
                                print(f"l2:  {l2:.4}")
                                print(f"Reg:  {reg:.4}")
                                print(f"LSD:  {lsd:.4}")
                                print(f"Loss (total):  {loss:.4}")
                                
                                t_end = time.time()
                                time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
                                print(f"Linear Inverse Problem: " + "        " + time_str)

                                sub = sub+1

                            print("----------------")
                            print("loss_mean(train)")
                            print("# [l2, LSD, Reg, Loss(total)]")
                            print(th.mean(loss_all[:,:77], dim=1))

                            print("----------------")
                            print("loss_mean(valid)")
                            print("# [l2, LSD, Reg, Loss(total)]")
                            print(th.mean(loss_all[:,77:87], dim=1))

                            print("----------------")
                            print("loss_mean(test)")
                            print("# [l2, LSD, Reg, Loss(total)]")
                            print(th.mean(loss_all[:,87:], dim=1))
                            print("----------------")
                            
                            
                            th.save(coeff_all, ptpath)
                            th.save(hrtf_all, ptpath_hrtf)
                            th.save(loss_all, ptpath_loss)

                else:
                    print("Obtain Coeff. by solving Linear Inverse Problem.")
                    bs = config["batch_size"]
                    trainer = Trainer(config, net_pre, traindataset)
                    sub = 1
                
                    for data in trainer.dataloader:
                        print("------")
                        t_start = time.time()
                        print(f"subject:{sub}")
                        srcpos, hrtf_gt = data
                        srcpos = srcpos.squeeze()
                        hrtf_gt = hrtf_gt.squeeze()
                        returns = net_pre.forward(srcpos, use_cuda_forward=False, use_lininv=True, hrtf=hrtf_gt)
                        hrtf_pred, coeff = returns["output"], returns["coeff"]
                        if sub==1:
                            coeff_train = th.zeros((coeff.shape[1], coeff.shape[2], coeff.shape[3], 77),dtype=th.complex64) # 77 = round(96*0.8)
                            # print(coeff.shape) # torch.Size([440, 2, 1296, 128])
                        # if sub < 3:
                        if True:
                            plotmaghrtf(srcpos=srcpos,hrtf_gt=hrtf_gt,hrtf_pred=hrtf_pred,mode='train_sub-'+str(sub)+'_'+str(config["max_truncation_order"]), 
                            idx_plot_list=np.array([202,211,220,229]), config=config # measured
                            # idx_plot_list=np.array([12,445,877,1309]) # simulated
                            ) 
                        
                        coeff_train[:,:,:,sub-1] = coeff[0,:,:,:]
                        l2 = MSE()(hrtf_pred, hrtf_gt)
                        lsd = LSD()(hrtf_pred, hrtf_gt)
                        reg = RegLoss()(coeff[0,:,:,:].unsqueeze(-1), SpecialFunc(maxto=config["max_truncation_order"]).n_vec)
                        loss = l2 * config["loss_weights"]["l2"] + reg * config["loss_weights"]["reg"] + lsd * config["loss_weights"]["lsd"]
                        print(f"l2:  {l2:.4}")
                        print(f"Reg:  {reg:.4}")
                        print(f"LSD:  {lsd:.4}")
                        print(f"Loss (total):  {loss:.4}")
                        
                        t_end = time.time()
                        time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
                        print(f"Linear Inverse Problem: " + "        " + time_str)
                        
                        # ## TensorBoard #
                        # writer.add_scalar(f"sub:{sub} / l2", l2, maxto) #
                        # writer.add_scalar(f"sub:{sub} / lsd", lsd, maxto) #
                        # writer.add_scalar(f"sub:{sub} / reg", reg, maxto) #
                        # writer.add_scalar(f"sub:{sub} / loss", loss, maxto) #

                        sub = sub+1
                    th.save(coeff_train, ptpath)
                # print(coeff.shape) # torch.Size([440, 2, 441, 128])
                # print(coeff_train.shape) # torch.Size([2, 441, 128, 77])
                # print(th.max(th.abs(coeff_train)))  # tensor(53.4447)
            print("---------------------")
            if config["lininv_only"]:
                sys.exit()

        if args.optuna:
            print("Optimize HyperParameter by Optuna.")
            study = optuna.create_study(study_name=config["artifacts_dir"],
                            # storage='mysql+pymysql://../optuna_study_m.db',
                            storage='sqlite:///./'+config["artifacts_dir"] +'/optuna_study.db',
                            load_if_exists=True)
            study.trials_dataframe().to_csv(config["artifacts_dir"] + "/study_history.csv")
            study.optimize(func=eval('objective_'+config["model"]),n_trials=None, timeout=2*24*3600, n_jobs=1)
            # if config["model"] == "AE":
            #     study.optimize(func=objective_AE,n_trials=None, timeout=86400, n_jobs=1)
            # elif config["model"] == "CAE":
            #     study.optimize(func=objective_CAE,n_trials=None, timeout=86400, n_jobs=1)
            print("params_{}".format(study.best_params))
            print("value_{}".format(study.best_value))
            study.trials_dataframe().to_csv(config["artifacts_dir"] + "/study_history.csv")
        else:
            if config["model"] == "AE" and not config["lininv_only"]:
                # Standardization
                # c_dim: [0:ch, 1:(m,n), 2:freq, 3:sub]
                c_dim = config["std_dim"]
                if c_dim == None:
                    coeff_std = th.ones(coeff_train.shape[:-1])
                    coeff_mean = th.zeros(coeff_train.shape[:-1])
                else:
                    if args.debug == True:
                        ptname_std = f'coeff_{config["max_frequency"]}_{round(config["fft_length"]/2)}_{config["max_truncation_order"]}_{config["loss_weights"]["reg"]:.0e}.pt'
                        ptpath_std = "outputs/" + ptname_std
                        # ptname_std = "coeff_"+ str(config["max_frequency"]) + "_" + str(round(config["fft_length"]/2)) + "_" + str(config["max_truncation_order"]) + ".pt"
                        # ptpath_std = config["artifacts_dir"] + "/" + ptname_std
                        if os.path.isfile(ptpath_std):
                            print("Load Coeff. from " + ptpath_std + "for standardization.")
                            coeff_train_std = th.load(ptpath_std)
                        else:
                            print(".pt file is required.")
                    else:
                        coeff_train_std = coeff_train
                    eps = 1e-10*th.ones(coeff_train_std.shape)
                    coeff_mag_log10 = th.log10(th.max(th.abs(coeff_train_std),eps))
                    coeff_std = th.std(coeff_mag_log10, dim=c_dim,unbiased=True)
                    coeff_mean = th.mean(coeff_mag_log10, dim=c_dim)
                net = HRTFApproxNetwork_AE(config=config, c_std = coeff_std, c_mean = coeff_mean)
                if args.load: # test or continue train
                    net.load_from_file(args.model_file)
                #     # net.load_state_dict(args.model_file)
            if not args.test:
                print("=====================")
                print(net)
                print("=====================")
                
                print("Train model.")
                print(f"number of trainable parameters: {net.num_trainable_parameters()}")
                print("---------------------")
                
                trainer = Trainer(config, net, traindataset)
                ## TensorBoard
                log_dir=config["artifacts_dir"]+"/logs/"+config["model"]+timestamp
                writer = SummaryWriter(log_dir)
                print("logdir: "+log_dir) 
                print("---------------------")

                train(trainer=trainer, ptins=ptins)
                # plotcz(dir=config["artifacts_dir"],config=config)
            # load
            print("test mode.")
            print("=====================")
            
            if args.test and args.load and args.model_file != '':
                net.load_from_file(args.model_file)
            else:
                net.load_from_file(config["artifacts_dir"]+'/hrtf_approx_network.best.net')
            print(net)
            print("=====================")

            #=== LAP Challenge ==========
            # config["pln_smp"] = False
            # config["pln_smp_paral"] = False
            # num_pts_list_lap = config['lap_smp_num']
            
            # for num_pts_lap in num_pts_list_lap:
            #     print("---------------")
            #     print(f'num_pts={num_pts_lap}')
            #     print("----")
            #     config["num_pts"] = num_pts_lap
            #     loss = test(net, True, -1, testdataset, f'test{config["timestamp"]}_lap-{num_pts_lap}')

            #=== regular polyhedron ==========
            # config["pln_smp"] = False
            # config["pln_smp_paral"] = False
            # num_pts_list_rp = [2522]
               
            # for itr_np, num_pts_rp in enumerate(num_pts_list_rp):
            #    print("---------------")
            #    print(f'num_pts={num_pts_rp}')
            #    print("----")
            #    config["num_pts"] = num_pts_rp
            #    # test(net, True, 1e16, testdataset, 'test'+config["timestamp"])
            #    loss = test(net, True, -1, testdataset, f'test{config["timestamp"]}_rp-{num_pts_rp}')

            #========================================================

            #=== t-design ==========
            config["pln_smp"] = False
            config["pln_smp_paral"] = False
            # t_list = range(2,19)
               
            # for t in t_list:
            #    print("---------------")
            #    print(f't={t}')
            #    print("----")
            #    config["num_pts"] = round((t+1)**2)
            #    # test(net, True, 1e16, testdataset, 'test'+config["timestamp"])
            #    loss = test(net, True, -1, testdataset, 'test'+config["timestamp"]+'_t-'+str(round(config["num_pts"]**0.5-1)))

            print("---------------")
            print(f'All pts (2522)')
            print("----")
            
            config["num_pts"] = 2522 #440
            #loss = test(net, True, -1, testdataset, 'test'+config["timestamp"]+'_440pts')
            loss = test(net, True, -1, testdataset, 'test'+config["timestamp"]+'_2522pts')

            #==== plane_sample ======
            #config["pln_smp"] = True
            #config["pln_smp_paral"] = False
            #pln_axes_list = [[0],[1],[2],[0,1],[1,2],[2,0],[0,1,2]]
            #for i, axes in enumerate(pln_axes_list):
            #    axes_char_f = ''
            #    axes_char_d = ''
            #    for ax in axes:
            #        axes_char_f = f'{axes_char_f}-{ax}'
            #        if axes_char_d == '':
            #            axes_char_d = f'{ax}'
            #        else:
            #            axes_char_d = f'{axes_char_d},{ax}'
            #    print("---------------")
            #    print(f'pln_smp: |pts[{axes_char_d}]|<0.01')
            #    print("----")
            #    config["pln_smp_axes"] = axes
            #    # test(net, True, 1e16, testdataset, 'test'+config["timestamp"])
            #    loss = test(net, True, -1, testdataset, f'test{config["timestamp"]}_plnsmp_axes{axes_char_f}')

            #==== planes_sample_paral ======
            #config["pln_smp"] = False
            #config["pln_smp_paral"] = True
            #pln_axis_list = [0,1,2]
            #pln_values_list = [-0.735, 0, 0.735]
            ## if config["DNN_for_interp_ITD"]:
            ##     loss_test_pln_par = np.zeros((3,1))
            ## else:
            ##     loss_test_pln_par = np.zeros((3,3))
            #for i, axis in enumerate(pln_axis_list):
            #    v_char = ''
            #    for v in pln_values_list:
            #        v_char = f'{v_char}_{v:.3f}' # _-0.735_0.000_0.735
            #    print("---------------")
            #    print(f'pln_smp_par: |pts[{axis}]|≒[{v_char}]')
            #    print("----")
            #    config['pln_smp_paral_axis'] = axis
            #    config['pln_smp_paral_values'] = pln_values_list
            #    # test(net, True, 1e16, testdataset, 'test'+config["timestamp"])
            #    loss = test(net, True, -1, testdataset, f'test{config["timestamp"]}_plnsmp_par_axis-{axis}{v_char}')
            ##     if config["DNN_for_interp_ITD"]:
            ##         loss_test_pln_par[i,0] = loss["mae_itd"]
            ##     else:
            ##         #[l2,lsd,l2_rec_n]
            ##         loss_test_pln_par[i,0] = loss["l2_rec"]
            ##         loss_test_pln_par[i,1] = loss["lsd"]
            ##         loss_test_pln_par[i,2] = loss["l2_rec_n"]
            ## loss_dir = f'{config["artifacts_dir"]}/Loss'
            ## os.makedirs(loss_dir, exist_ok=True)
            ## th.save(loss_test_pln_par,f"{loss_dir}/loss_test_plnsmp_par_{v_char}.pt")
    elif args.lininv:
        print("solve linear inverse provlem.")
        traindataset = dataset
        config["batch_size"] = min([1024*2, len(traindataset)])
        bs = config["batch_size"]
        print(f"number of data points: {bs}")
       
        trainer = Trainer(config, net, traindataset)
        t_start = time.time()
        for data in trainer.dataloader:
            srcpos, hrtf_gt = data
            returns = net.forward(srcpos, use_cuda_forward=False, use_lininv=True, hrtf=hrtf_gt)
            hrtf_pred, coeff = returns["output"], returns["coeff"]
            l2 = MSE()(hrtf_pred, hrtf_gt)
            lsd = LSD()(hrtf_pred, hrtf_gt)
            reg = RegLoss()(coeff, SpecialFunc(maxto=config["max_truncation_order"]).n_vec)
            loss = l2 * config["loss_weights"]["l2"] + reg * config["loss_weights"]["reg"] + lsd * config["loss_weights"]["lsd"]
            print(f"l2:  {l2:.4}")
            print(f"Reg:  {reg:.4}")
            print(f"LSD:  {lsd:.4}")
            print(f"Loss (total):  {loss:.4}")
            
            t_end = time.time()
            time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
            print(f"Linear Inverse Problem: " + "        " + time_str)
            plotmaghrtf(srcpos=srcpos,hrtf_gt=hrtf_gt,hrtf_pred=hrtf_pred,mode='train', config=config)     
            break
        print("---------------------")
        print("Validation")
        t_start = time.time()
        loss_valid =  test(flg_save, 1e+16, validdataset, 'valid', use_coeff=True, coeff = coeff)
        t_end = time.time()
        time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
        print(f"Valid: " + "        " + time_str)
        
    elif flg_train:
        traindataset = dataset
        print(f"number of trainable parameters: {net.num_trainable_parameters()}")
        print("---------------------")
        trainer = Trainer(config, net, traindataset)
        BEST_LOSS = 1e+30
        LAST_SAVED = -1
        ## TensorBoard
        log_dir=config["artifacts_dir"]+"/logs/"+config["model"]+timestamp
        # log_dir = "./logs/direct_0920_1224"
        writer = SummaryWriter(log_dir)
        print("logdir: "+log_dir) 

        for epoch in range(args.start_ep+1, args.start_ep+config["epochs"]+1):
            trainer.cuda()
            trainer.train()
            ## Train
            loss_train = trainer.train_1ep(epoch) 
            ## TensorBoard
            writer.add_scalar("L2 / train",loss_train["l2"],epoch)
            writer.add_scalar("Reg / train",loss_train["reg"],epoch)
            writer.add_scalar("LSD / train",loss_train["lsd"],epoch)
            if config["model"] == "PINN":
                writer.add_scalar("Helmholtz / train",loss_train["helmholtz"],epoch)
            writer.add_scalar("Loss (total) / train",loss_train["loss"],epoch)
            ## Validation
            print("---------------------")
            t_start = time.time()
            loss_valid = test(flg_save, BEST_LOSS, validdataset, 'valid')
            t_end = time.time()
            time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
            print(f"epoch {epoch} (valid)" + "        " + time_str)
            ## TensorBoard
            writer.add_scalar("L2 / valid",loss_valid["l2"],epoch)
            writer.add_scalar("Reg / valid",loss_valid["reg"],epoch)
            writer.add_scalar("LSD / valid",loss_valid["lsd"],epoch)
            if config["model"] == "PINN":
                writer.add_scalar("Helmholtz / valid",loss_valid["helmholtz"],epoch)
            writer.add_scalar("Loss (total) / valid",loss_valid["loss"],epoch)
            ## Update Learning rate
            trainer.optimizer.update_lr(loss_valid["loss"])

            if loss_valid["loss"] < BEST_LOSS:
                BEST_LOSS = loss_valid["loss"]
                LAST_SAVED = epoch
                print("Best Loss. Saving model!")
                # trainer.save(suffix="best_"+f'{epoch:03}'+"ep")
                trainer.save(suffix="best")
            elif config["save_frequency"] > 0 and epoch % config["save_frequency"] == 0:
                print("Saving model!")
                trainer.save(suffix="log_"+f'{epoch:03}'+"ep")
            else:
                print("Not saving model! Last saved: {}".format(LAST_SAVED))
            print("---------------------")
        # Save final model
        trainer.save(suffix="final_"+f'{epoch:03}'+"ep")

    else:
        l2_error = test(flg_save, -1, testdataset, 'test')
    
    print("Done.")