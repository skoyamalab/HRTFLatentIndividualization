import os
import sys
import argparse
import datetime
import time
import pprint

import numpy as np
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt   
import optuna
from icecream import ic
# ic.configureOutput(includeContext=True)

from src.dataset import HRTFDataset #, HRTFTestset
from src.models import HRTFApproxNetwork_FIAE
from src.trainer import Trainer
from src.losses import MAE, LSD, ILD_AE
from src.configs import * # configs
from src.utils import replace_activation_function

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
                        help="resume training at this epoch")
    parser.add_argument("-save_freq","--save_freq", type=int, default=250,
                        help="save model after how many epochs")
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
        loss_train = trainer.train_1ep(epoch) 
        # print("trained")
        if config["lr_update"] == 'train':
            ## Update Learning rate
            trainer.optimizer.update_lr_one(loss_train["loss"])
        ## TensorBoard
        for k, v in loss_train.items():
            writer.add_scalar(k + " / train", v, epoch)

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
        plt.clf()
        plt.close()
    # Save final model
    trainer.save(suffix="final_"+f'{epoch:03}'+"ep")

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

            # elif data_kind == 'HRIR':
            #     if 'ITD' in config["data_kind_interp"]:
            #         t = prediction['ITD'].permute(1,0) # B,S
            #     elif config["use_itd_gt"]:
            #         t = data_slice['ITD'].permute(1,0) # B,S
            #     else:
            #         t = None
            #     prediction['HRIR'] = Data_ITD_2_HRIR(prediction['HRTF_mag'].permute(1,2,3,0), itd=t, data_kind='HRTF_mag', config=config).permute(3,0,1,2)
            elif data_kind == 'lsd_bm':
                prediction['lsd_bm'] = lsd_loss(prediction['HRTF_mag'], data_slice['HRTF_mag'], dim=-1, data_kind='HRTF_mag', mean=False) # S,B,2
            
            if data_kind not in prediction_all:
                prediction_all[data_kind] = {}
            if db_name not in prediction_all[data_kind]:
                if data_kind in data:
                    prediction_all[data_kind][db_name] = th.zeros_like(data[data_kind][db_name])
                else:
                    if data_kind == 'lsd_bm':
                        prediction_all[data_kind][db_name] = th.zeros_like(data['HRTF_mag'][db_name][:,:,0,:])
            prediction_all[data_kind][db_name][..., sub_id:sub_id+1] = prediction[data_kind].permute(th.arange(prediction[data_kind].dim())[1:].tolist() + [0])
        
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

    traindataset = dataset
    ptins = "debug_" if args.debug else ""

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

        #=== t-design ==========
        config["pln_smp"] = False
        config["pln_smp_paral"] = False

        print("---------------")
        print('All pts (2522)')
        print("----")
        
        config["num_pts"] = 2522
        loss = test(net, True, -1, testdataset, 'test'+config["timestamp"]+'_2522pts')

    print("Done.")
