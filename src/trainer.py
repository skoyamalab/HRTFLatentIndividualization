import os
import time 
import sys
# from torch._C import complex64
from torch.nn.functional import cross_entropy
import torchaudio as ta
import tqdm
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.utils import NewbobAdam, SpecialFunc, sph2cart, plotmaghrtf, plothrir, hrir2itd, minphase_recon, assign_itd, posTF2IR_dim4, plotazimzeni, plot_mag_Angle_vs_Freq, Data_ITD_2_HRIR
from src.losses import MAE, MSE, NMSE, L2Loss_angle,VarLoss, CosSimLoss, RegLoss, LSD, HelmholtzLoss, HelmholtzLoss_Cart, HuberLoss, LSDdiffLoss, WeightedLSD, RMS, CosDistIntra, CosDistIntraSquared, PhaseLoss, LogSpecDiff_1st, LogSpecDiff_2nd, LogSpecDiff_4th, NotchPriorWeightedLSD, ItakuraSaito, LogSpecDiff_general, ILD_AE


class Trainer:
    def __init__(self, config, net, dataset):
        '''
        :param config: a dict containing parameters
        :param net: the network to be trained, must be of type src.utils.Net
        :param dataset: the dataset to be trained on
        '''
        self.config = config
        self.dataset = dataset
        gpus = list(range(config["num_gpus"]))
        # self.dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=1)
        self.dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=1)
        self.net = th.nn.DataParallel(net, device_ids=gpus)
        weights = filter(lambda x: x.requires_grad, net.parameters())
        self.optimizer = NewbobAdam(weights,
                                    net,
                                    artifacts_dir=config["artifacts_dir"],
                                    initial_learning_rate=config["learning_rate"],
                                    decay=config["newbob_decay"],
                                    max_decay=config["newbob_max_decay"],
                                    timestamp = config["timestamp"])
        self.mae_loss = MAE()
        self.mse_loss = MSE()
        self.nmse_loss = NMSE()
        self.mse_loss_angle = L2Loss_angle(mask_beginning=config["mask_beginning"])
        self.huber_loss = HuberLoss()
        self.cossim_loss = CosSimLoss()
        self.var_loss = VarLoss()
        self.lsd_loss = LSD()
        self.Weightedlsd_loss = WeightedLSD()
        self.NotchPWlsd_loss = NotchPriorWeightedLSD(config["npwlsd_gamma"],config["npwlsd_epsilon"])
        self.lsddiff_loss = LSDdiffLoss()
        self.ild_loss = ILD_AE()
        self.reg_loss = RegLoss()
        self.helmholtz_loss = HelmholtzLoss()
        self.helmholtz_loss_cart = HelmholtzLoss_Cart()
        self.cross_entropy_loss = th.nn.CrossEntropyLoss()
        self.rms_loss = RMS()
        self.cosdist = CosDistIntra()
        self.cosdistsq = CosDistIntraSquared()
        self.phase_loss = PhaseLoss()
        self.ls_diff_1st_loss = LogSpecDiff_1st()
        self.ls_diff_2nd_loss = LogSpecDiff_2nd()
        self.ls_diff_4th_loss = LogSpecDiff_4th()
        self.ls_diff_gen_loss = LogSpecDiff_general(order=config["ls_diff_gen_order"], activation=config["ls_diff_gen_activation"], activate_coeff=config["ls_diff_gen_activate_coeff"], leak_coeff=config["ls_diff_gen_leak_coeff"])
        self.IS = ItakuraSaito(bottom=config["is_pm_bottom"]) # 'target' or 'data'
        # if self.config["model"] == "AE":
        #     self.coeff = coeff
        self.total_iters = 0
        # switch to training mode
        self.net.train()


    def save(self, suffix=""):
        self.net.module.save(self.config["artifacts_dir"], suffix)

    def train(self):
        for epoch in range(self.config["epochs"]):
            loss_stats = {}
            itr = 0
            for data in self.dataloader:
                loss_new = self.train_iteration(data)
                # logging
                for k, v in loss_new.items():
                    loss_stats[k] = loss_stats[k]+v if k in loss_stats else v
                if itr % round(len(self.dataloader)/5) == 0:
                    print(f'{itr:04}'+ "/" + str(len(self.dataloader)))
                itr += 1
            for k in loss_stats:
                loss_stats[k] /= len(self.dataloader)
            self.optimizer.update_lr(loss_stats["accumulated_loss"])
            loss_str = "    ".join([f"{k}:{v:.4}" for k, v in loss_stats.items()])
            print(f"epoch {epoch+1} " + loss_str)
            # Save model
            if self.config["save_frequency"] > 0 and (epoch + 1) % self.config["save_frequency"] == 0:
                self.save(suffix='epoch-' + str(epoch+1))
                print("Saved model")
        # Save final model
        self.save()

    def train_1ep(self,epoch,coeff=None, mode='train'):
        # print(coeff.shape) # torch.Size([2, 400, 128, 77])
        t_start = time.time()
        loss_stats = {}
        # itr = 0
        # for param_group in self.optimizer.param_groups:
        #     print(f"lr: {param_group['lr']}")
        # print(self.optimizer.state_dict())
        num_data = np.ceil(min(
            sum([len(self.config['sub_index'][db_name]['train']) for db_name in self.config['database']])/self.config["batch_size"], len(self.dataloader)
            ))
        print(f"len(dataloader):{len(self.dataloader)}, num_data:{num_data}")

        if self.config["several_num_pts"]:
            self.num_pts_init = self.config["num_pts"]
            num_pts_list = self.config["num_pts_list"]
        else:
            num_pts_list = [self.config["num_pts"]]
        
        for itr, data in enumerate(self.dataloader):
            # print(f'itr:{itr}')
            for itr_numpts, num_pts in enumerate(num_pts_list):
                self.config["num_pts"] = num_pts
                # print(f'num_pts:{num_pts}')
                if itr > num_data-1:
                    break
                if self.config["model"] in ["AE", "CAE", "HCAE", "CNN"]: 
                    # print(f"sub: {sub+1}")
                    # if self.config["batch_size"] == 1:
                    #     loss_new = self.train_iteration(data, coeff=coeff[:,:,:,itr], sub=itr)
                    # else:
                    bs = self.config["batch_size"]
                    slice = range(itr*bs, itr*bs + min(bs, data["SrcPos"].shape[0]))
                    if self.config["class"] == "sub":
                        label = th.tensor(slice).cuda()
                    else:
                        # cf. 20220303.ipynb
                        if self.config["num_cluster"] == 4:
                            cluster = th.tensor([0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 3, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0,
                                1, 0, 1, 0, 0, 1, 3, 1, 2, 0, 0, 1, 3, 3, 3, 1, 3, 1, 0, 3, 1, 3, 0, 2,
                                3, 1, 0, 1, 1, 0, 1, 0, 3, 1, 3, 1, 3, 0, 3, 1, 3, 3, 0, 1, 1, 3, 1, 3,
                                1, 3, 3, 3, 1])
                        elif self.config["num_cluster"] == 8:
                            cluster = th.tensor([0, 0, 1, 6, 0, 0, 4, 5, 0, 0, 3, 5, 0, 3, 4, 6, 0, 1, 4, 5, 6, 1, 6, 5,
                                4, 5, 6, 0, 0, 6, 3, 5, 7, 0, 0, 1, 1, 3, 3, 6, 1, 6, 1, 0, 1, 3, 0, 2,
                                1, 1, 0, 1, 4, 0, 4, 0, 3, 4, 3, 1, 3, 6, 3, 1, 3, 3, 5, 4, 4, 3, 1, 3,
                                1, 0, 3, 0, 1])
                        elif self.config["num_cluster"] == 16:
                            cluster = th.tensor([ 6,  7, 13, 14,  1,  1,  4, 13,  7,  7, 11, 14,  7,  2, 14, 11,  6,  3,
                                13, 14,  3, 13, 15, 13,  4, 14, 14,  7,  6, 15, 11, 13,  8,  7, 10, 10,
                                12,  9,  0,  3, 12, 15,  3,  7, 13,  2,  5, 15, 11,  5,  5,  3,  4,  5,
                                4,  7, 11,  4,  0,  3,  5, 14, 11,  4,  6,  6, 10, 13,  4,  3,  3, 11,
                                3,  9,  0,  3,  3])
                        elif self.config["num_cluster"] == 32:
                            cluster = th.tensor([27, 27, 31, 29,  0,  0, 10,  2, 27, 27, 28, 17, 14, 15, 11, 21, 20, 31,
                                2, 17, 25, 31,  8, 17, 31, 17,  8, 14, 20,  7, 16,  7, 26, 15,  6,  4,
                                13, 28,  5, 25, 12,  9, 22,  1,  5, 16, 24, 23, 30, 31, 14, 25,  4, 14,
                                10, 15, 30, 10,  5, 13, 27,  8, 16,  3, 18,  5,  4, 19, 19, 13, 24,  5,
                                25, 15, 18, 28, 25])
                        elif self.config["num_cluster"] == 64:
                            cluster = th.tensor([31, 53, 48,  9, 21, 21, 35, 25, 59, 61, 30,  7, 53, 45, 10, 52, 32, 16,
                                47, 23, 13, 48, 24, 20, 18, 46, 38, 12, 43, 62, 44, 34, 33, 53, 27,  3,
                                39, 30, 54, 22, 15, 42, 19, 55, 37,  4, 14, 50, 41, 28, 28, 13, 57, 56,
                                5, 53, 17, 35, 36,  0, 49,  8,  6, 51,  1, 60,  2, 26, 40,  0, 29, 54,
                                41, 58, 63, 20, 11])
                        label = cluster[slice].cuda()
                    
                    loss_new = self.train_iteration(data, coeff=coeff[:,:,:,slice], sub=label, itr=itr, mode=mode)
                elif self.config["model"] == "FIAE":
                    bs = self.config["batch_size"]
                    slice = range(itr*bs, itr*bs + min(bs, data["SrcPos"].shape[0]))
                    label = th.tensor(slice).cuda()
                    loss_new = self.train_iteration(data, sub=label, itr=itr, itr_numpts=itr_numpts, mode=mode)
                elif self.config["model"] == "Lemaire05":
                    bs = self.config["batch_size"]
                    slice = range(itr*bs, itr*bs + min(bs, data["SrcPos"].shape[0]))
                    loss_new = self.train_iteration(data, coeff=None, sub=th.tensor(slice).cuda())
                else:
                    loss_new = self.train_iteration(data, coeff=coeff)
                # logging
                for k, v in loss_new.items():
                    if k == 'coeff':
                        if itr == 0 and itr_numpts==0:
                            # print(loss_new["coeff"].shape)
                            coeff_train = th.zeros((loss_new["coeff"].shape[0], loss_new["coeff"].shape[1], loss_new["coeff"].shape[2], 77),dtype=th.complex64).to(loss_new["coeff"].device)
                        # print(loss_new["coeff"].shape) # torch.Size([2, 361, 128, 4])
                        coeff_train[:,:,:,slice] = loss_new["coeff"] 
                    elif k == 'z':
                        if itr == 0 and itr_numpts==0:
                            # print(loss_new["coeff"].shape)
                            # print(loss_new["z"].shape) # torch.Size([25, 16, 4])
                            z_dim_last = 77 * 2 if self.config["use_lr_aug"] else 77
                            z_train = th.zeros((loss_new["z"].shape[0], loss_new["z"].shape[1],  z_dim_last),dtype=th.float32).to(loss_new["z"].device)
                            # print(z_train.shape) # torch.Size([1, 16, 77])
                        if itr_numpts==0:
                            if self.config["use_lr_aug"]:
                                slice_z = range(itr*bs*2, itr*bs*2 + min(bs*2, 2*data["SrcPos"].shape[0]))
                                # print(slice_z)
                                z_train[:,:,slice_z] = loss_new["z"] 
                            else:
                                z_train[:,:,slice] = loss_new["z"] 
                    elif k == 'coeff_in':
                        if itr == 0 and itr_numpts==0:
                            # print(loss_new["coeff"].shape)
                            coeff_in = th.zeros((loss_new["coeff_in"].shape[0], loss_new["coeff_in"].shape[1], loss_new["coeff_in"].shape[2], 77),dtype=th.complex64).to(loss_new["coeff_in"].device)
                        # print(loss_new["coeff"].shape) # torch.Size([2, 361, 128, 4])
                        coeff_in[:,:,:,slice] = loss_new["coeff_in"] 
                    else:
                        loss_stats[k] = loss_stats[k]+v if k in loss_stats else v
            #===== progress bar ======
            prog_step = 20
            if round(num_data/prog_step) != 0:
                if itr == 0:
                    print('[',end='')
                elif itr == num_data-1:
                    print('#]')
                elif itr % round(num_data/prog_step) == 0:
                    print('#',end='')
                    # print(f'{itr:03}'+ "/" + str(len(self.dataloader)))
            #=========================
        # sys.exit()
        #if self.config["several_num_pts"]:
        #    self.config["num_pts"] = self.num_pts_init

        for k in loss_stats:
            loss_stats[k] /= (num_data*len(num_pts_list))
        t_end = time.time()
        
        # loss_str = "    ".join([f"{k}:{v:.4}" for k, v in loss_stats.items()])
        loss_str = "    ".join([f"{k}:{loss_stats[k]:.4}" for k in sorted(loss_stats.keys())])
        time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
        print(f"epoch {epoch} (train) ")
        print(loss_str + "        " + time_str)
        if self.config["model"] == "CNN":
            return loss_stats, coeff_train, coeff_in, z_train
        elif self.config["model"] == "Lemaire05":
            return loss_stats
        elif self.config["model"] == "FIAE":
            return loss_stats
        else:
            return loss_stats, coeff_train, z_train

    def train_iteration(self, data, sub, coeff=None, itr=1, itr_numpts=0, mode='train'):
        '''
        one optimization step
        :param data: tuple of tensors containing source position, hrtf
        :return: dict containing values for all different losses
        '''
        returns = {}
        loss_dict = {}
        # forward
        self.optimizer.zero_grad()
        if True:
            for data_kind in self.config["data_kind_interp"]:
                data[data_kind] = data[data_kind].cuda()
            
            prediction = self.net.forward(data=data, mode=mode)
            # get rid of idx_mes_pos from foward for gpu parallelization
            prediction["idx_mes_pos"] = range(0, data["SrcPos"].shape[1]) # new

            for data_kind in self.config["data_kind_interp"]:
                if data_kind == 'ITD':
                    loss_dict["mae_itd"] = self.mae_loss(prediction['ITD'], data['ITD']) # S,B
                elif data_kind == 'HRTF_mag':
                    loss_dict["lsd"] = self.lsd_loss(prediction['HRTF_mag'], data['HRTF_mag'], dim=-1, data_kind='HRTF_mag', mean=True) # S,B,2,L -> scaler
                    loss_dict["mae_ild"] = self.ild_loss(prediction['HRTF_mag'], data['HRTF_mag'], dim=-2, data_kind='HRTF_mag', mean=True)  # S,B,2,L -> scaler
 
            # plot 
            if data['sub_idx'][0] < self.config['sub_index'][data['db_name'][0]]['train'][1] and itr_numpts==0:
                
                if 'ITD' in self.config["data_kind_interp"]:
                    #=== plot ITD ====
                    fig_dir_itd = f'{self.config["artifacts_dir"]}/figure/ITD/'
                    os.makedirs(fig_dir_itd, exist_ok=True)
                    for sub_id in range(min(3,len(sub))):
                        plotazimzeni(pos=data['SrcPos'][sub_id,:,:].cpu(), c=prediction['ITD'][sub_id,:].cpu().detach()*1000,fname=f"{fig_dir_itd}ITD_{data['db_name'][0]}-sub-{data['sub_idx'][0]+1}_{mode}",title=f'sub:{sub_id+1}',cblabel=f'ITD (ms)',cmap='bwr',figsize=(10.5,5),dpi=300,emphasize_mes_pos=True, idx_mes_pos=prediction["idx_mes_pos"])

                        plotazimzeni(pos=data['SrcPos'][sub_id,:,:].cpu(), c=th.abs(prediction['ITD'][sub_id,:]-data['ITD'][sub_id,:]).cpu().detach()*1000,fname=f"{fig_dir_itd}ITD_AE_{data['db_name'][0]}-sub-{data['sub_idx'][0]+1}_{mode}",title=f'sub:{sub_id+1}',cblabel=f'AE of ITD (ms)',cmap='gist_heat',figsize=(10.5,5),dpi=300,emphasize_mes_pos=True, idx_mes_pos=prediction["idx_mes_pos"])

                if 'HRTF_mag' in self.config["data_kind_interp"]:
                    #=== plot HRTF_mag ====
                    for sub_id in range(min(3,len(sub))):
                        plotmaghrtf(srcpos=data['SrcPos'][sub_id,:,:], sig_gt=data['HRTF_mag'][sub_id,:,:,:], sig_pred=prediction['HRTF_mag'][sub_id,:,:,:], idx_plot_list=np.array(self.config["idx_plot_list"][data['db_name'][0]]),config=self.config,  figdir=f"{self.config['artifacts_dir']}/figure/HRTF/", fname=f"{self.config['artifacts_dir']}/figure/HRTF/HRTF_Mag_{data['db_name'][0]}-sub-{data['sub_idx'][0]+1}_{mode}", data_kind = 'HRTF_mag')

                        plot_mag_Angle_vs_Freq(HRTF_mag=prediction['HRTF_mag'][sub_id,:,:,:], SrcPos=data['SrcPos'][sub_id,:,:], fs=self.config['max_frequency'], figdir=f"{self.config['artifacts_dir']}/figure/HRTF_mag_2D/", fname=f"{self.config['artifacts_dir']}/figure/HRTF_mag_2D/HRTF_Mag_{data['db_name'][0]}-sub-{data['sub_idx'][0]+1}_{mode}")

                    #=== plot HRIR ====
                    # if 'ITD' in self.config["data_kind_interp"]:
                    #     t = prediction['ITD'].permute(1,0) # B,S
                    # elif self.config["use_itd_gt"]:
                    #     t = data['ITD'].permute(1,0).cuda() # B,S
                    # else:
                    #     t = None
                    # prediction['HRIR'] = Data_ITD_2_HRIR(prediction['HRTF_mag'].permute(1,2,3,0), itd=t, data_kind='HRTF_mag', config=self.config).permute(3,0,1,2)
                    # for sub_id in range(min(3,len(sub))):
                    #     _, _ = plothrir(srcpos=data['SrcPos'][sub_id,:,:], hrtf_gt=None,hrtf_pred=None, idx_plot_list=np.array(self.config["idx_plot_list"][data['db_name'][0]]), config=self.config, hrir_gt=data['HRIR'][sub_id,:,:,:], hrir_pred=prediction['HRIR'][sub_id,:,:,:], figdir=f"{self.config['artifacts_dir']}/figure/HRIR/", fname=f"{self.config['artifacts_dir']}/figure/HRIR/HRIR_{data['db_name'][0]}-sub-{sub_id+1}_{mode}")

            loss = 0
            for k,v in loss_dict.items():
                if k in self.config["loss_weights"]:
                    if self.config["loss_weights"][k] > 0:
                        loss += self.config["loss_weights"][k] * v
                returns[k] = v.detach().clone()
            returns["loss"] = loss.detach().clone()
            # update model parameters
            loss.backward()
            self.optimizer.step()
            self.total_iters += 1

            return returns
        else:


            if self.config["DNN_for_interp_ITD"]:
                srcpos, itd_gt = data["SrcPos"], data["ITD"]
                # print(srcpos.shape) # torch.Size([1, 440, 3])
                # print(itd_gt.shape) # torch.Size([1, 440])
                srcpos = srcpos.permute(1,2,0).cuda()
                itd_gt = itd_gt.permute(1,0).cuda()
                if self.config["model"] == "FIAE":
                    prediction = self.net.forward(input=itd_gt, srcpos=srcpos, mode=mode, db_name=data["db_name"])
                    itd_pred = prediction["output"]
                    loss_dict["mae_itd"] = self.mae_loss(itd_pred, itd_gt)
                    if itr==0 and itr_numpts==0:
                        # for sub_id in sub[:min(3,len(sub))]:
                        fig_dir_itd = f'{self.config["artifacts_dir"]}/figure/ITD/'
                        os.makedirs(fig_dir_itd, exist_ok=True)
                        for sub_id in range(min(3,len(sub))):
                            plotazimzeni(pos=srcpos[:,:,sub_id].cpu(),c=itd_pred[:,sub_id].cpu().detach()*1000,fname=f'{fig_dir_itd}itd_{mode}_sub-{sub_id+1}',title=f'sub:{sub_id+1}',cblabel=f'ITD (ms)',cmap='bwr',figsize=(10.5,5),dpi=300,emphasize_mes_pos=True, idx_mes_pos=prediction["idx_mes_pos"])

                            plotazimzeni(pos=srcpos[:,:,sub_id].cpu(),c=th.abs(itd_pred[:,sub_id]-itd_gt[:,sub_id]).cpu().detach()*1000,fname=f'{fig_dir_itd}itd_ae_{mode}_sub-{sub_id+1}',title=f'sub:{sub_id+1}',cblabel=f'AE of ITD (ms)',cmap='gist_heat',figsize=(10.5,5),dpi=300,emphasize_mes_pos=True, idx_mes_pos=prediction["idx_mes_pos"])

                    loss = 0
                    for k,v in loss_dict.items():
                        # print(k)
                        if k in self.config["loss_weights"]:
                            if self.config["loss_weights"][k] > 0:
                                # print(k)
                                loss += self.config["loss_weights"][k] * v
                        returns[k] = v.detach().clone()
                    returns["loss"] = loss.detach().clone()
                    loss.backward()
                    self.optimizer.step()
                    self.total_iters += 1

                    return returns

                else:
                    raise NotImplementedError

            if self.config["DNN_for_interp_HRIR"]:
                srcpos, hrir_gt, hrtf_gt = data["SrcPos"], data["HRIR"], data["HRTF"]
                srcpos = srcpos.permute(1,2,0).cuda()  # B,3,S
                hrtf_gt = hrtf_gt.permute(1,2,3,0).cuda() # B,2,2L,S
                hrir_gt = hrir_gt.permute(1,2,3,0).cuda() # B,2,2L,S
                if self.config["model"] == "FIAE":
                    prediction = self.net.forward(input=hrir_gt, srcpos=srcpos, mode=mode, db_name=data["db_name"][0])
                    hrir_pred = prediction["output"]
                    hrtf_pred = th.conj(th.fft.fft(hrir_pred, dim=2))[:,:,1:self.config["fft_length"]//2+1,:]

                    #== calc loss ===
                    loss_dict["l2_time"] = self.mse_loss(hrir_pred, hrir_gt)
                    loss_dict["lsd"] = self.lsd_loss(hrtf_pred, hrtf_gt)

                    if data['sub_idx'][0] < self.config['sub_index'][data['db_name'][0]]['train'][3] and itr_numpts==0:
                        for sub_id in range(min(3,len(sub))):
                            plotmaghrtf(srcpos=srcpos[:,:,sub_id],hrtf_gt=hrtf_gt[:,:,:,sub_id],hrtf_pred=hrtf_pred[:,:,:,sub_id], idx_plot_list=np.array(self.config["idx_plot_list"][data['db_name'][0]]),config=self.config,  figdir=f"{self.config['artifacts_dir']}/figure/HRTF/", fname=f"{self.config['artifacts_dir']}/figure/HRTF/HRTF_Mag_{data['db_name'][0]}-sub-{data['sub_idx'][0]+1}_{mode}")

                            _, _ = plothrir(srcpos=srcpos[:,:,sub_id],hrtf_gt=None,hrtf_pred=None,idx_plot_list=np.array(self.config["idx_plot_list"][data['db_name'][0]]), config=self.config, hrir_gt=hrir_gt[:,:,:,sub_id], hrir_pred=hrir_pred[:,:,:,sub_id], figdir=f"{self.config['artifacts_dir']}/figure/HRIR/", fname=f"{self.config['artifacts_dir']}/figure/HRIR/HRIR_{data['db_name'][0]}-sub-{sub_id+1}_{mode}")
                    
                    loss = 0
                    for k,v in loss_dict.items():
                        if k in self.config["loss_weights"]:
                            if self.config["loss_weights"][k] > 0:
                                loss += self.config["loss_weights"][k] * v
                        returns[k] = v.detach().clone()
                    returns["loss"] = loss.detach().clone()
                    # update model parameters
                    loss.backward()
                    self.optimizer.step()
                    self.total_iters += 1

                    return returns
                else:
                    raise NotImplementedError
            else: # interp HRTF
                srcpos, hrtf_gt = data["SrcPos"], data["HRTF"]
                srcpos = srcpos.permute(1,2,0).cuda()
                hrtf_gt = hrtf_gt.permute(1,2,3,0).cuda()
                # print(hrtf_gt.shape) # torch.Size([440, 2, 128, S])
                # print(srcpos.shape)  # torch.Size([440, 3, S])
                    
                if self.config["model"] in ["AE", "CAE", "HCAE", "AE_PINN", "CNN", "Lemaire05", "FIAE"]:
                    # print(hrtf_gt.shape) # torch.Size([440, 2, 128])
                    # print(hrtf_gt.dtype) # torch.complex64
                    hrtf_gt_l = hrtf_gt[:,0,:,:]
                    hrtf_gt_r = hrtf_gt[:,1,:,:]

                    if self.config["model"] == "CNN":
                        hrtf_re, hrtf_im = th.real(hrtf_gt), th.imag(hrtf_gt)
                        prediction = self.net.forward(hrtf_re, hrtf_im, srcpos)
                    elif self.config["model"] == "AE":
                        input = th.cat((th.real(hrtf_gt_l), th.imag(hrtf_gt_l), th.real(hrtf_gt_r), th.imag(hrtf_gt_r)), dim=1)
                        # print(coeff.shape) # torch.Size([2, 400, 128, 16])
                        coeff_in = th.cat((th.real(coeff),th.imag(coeff)),dim=0)
                        # print(coeff_in.shape) # torch.Size([4, 400, 128, 16])
                        # th.autograd.set_detect_anomaly(True)
                        prediction = self.net.forward(input=input, use_cuda_forward=True, use_srcpos=True, srcpos=srcpos, sub=sub, mode=mode, coeff=coeff_in)
                    elif self.config["model"] == "FIAE":
                        # print(hrtf_gt_l.dtype)
                        input = th.cat((th.real(hrtf_gt_l), th.imag(hrtf_gt_l), th.real(hrtf_gt_r), th.imag(hrtf_gt_r)), dim=1)#.to(th.float64)
                        # print(input.dtype) # th.float64
                        prediction = self.net.forward(input=input, srcpos=srcpos, mode=mode, db_name=data["db_name"][0])
                    elif  self.config["model"] == "Lemaire05":
                        mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude',top_db=80)
                        # eps = 0*th.ones(hrtf_gt_l.shape).to(hrtf_gt_l.device)
                        # magdb_l = th.log10(th.max(mag_l, eps))
                        hrtf_logmag_l = mag2db(th.abs(hrtf_gt_l))
                        hrtf_logmag_r = mag2db(th.abs(hrtf_gt_r))
                        input = th.cat((hrtf_logmag_l,hrtf_logmag_r), dim=1)
                        prediction = self.net.forward(input=input, srcpos=srcpos)
                    else:
                        input = th.cat((th.real(hrtf_gt_l).unsqueeze(1), th.imag(hrtf_gt_l).unsqueeze(1), th.real(hrtf_gt_r).unsqueeze(1), th.imag(hrtf_gt_r).unsqueeze(1)), dim=1)
                        prediction = self.net.forward(input=input, use_cuda_forward=True, use_srcpos=True, srcpos=srcpos, sub=sub)

                    # if self.config["batch_size"] == 1:
                    #     input = input.squeeze()
                    #     srcpos = srcpos.squeeze()
                    #     hrtf_gt = hrtf_gt.squeeze()
                    

                    hrtf_pred = prediction["output"]
                    if self.config["model"] == "AE":
                        # #===================
                        # # r_pred = prediction["r"] # fixed in HUTUBS 
                        # vec_pred = prediction["vec"]
                        # theta_pred = prediction["theta"]
                        # phi_pred = prediction["phi"]
                        # loss_dict["l2_t"] = self.mse_loss_angle(theta_pred, srcpos[:,2]) # zenith, l2_loss() でも OK
                        # loss_dict["l2_p"] = self.mse_loss_angle(phi_pred, srcpos[:,1]) # azimuth
                        # loss_dict["cossim"] = self.cossim_loss(vec_pred,sph2cart(srcpos[:,1],srcpos[:,2],srcpos[:,0])) # sph2cart(azi,zen,r)
                        # #===================
                        z_pred = prediction["z"]
                        loss_dict["var_z"] = self.var_loss(z_pred)
                        loss_dict["cdintra_z"] = self.cosdist(z_pred)
                        loss_dict["cdintrasq_z"] = self.cosdistsq(z_pred)

                    loss_dict["l2_rec"] = self.mse_loss(hrtf_pred, hrtf_gt)
                    loss_dict["l2_rec_n"] = self.nmse_loss(hrtf_pred, hrtf_gt)
                    loss_dict["lsd"] = self.lsd_loss(hrtf_pred, hrtf_gt)
                    loss_dict["lsd_npw"] = self.NotchPWlsd_loss(hrtf_pred, hrtf_gt)
                    loss_dict["lsd_diff"] = self.lsddiff_loss(hrtf_pred)
                    loss_dict["ls_diff_1st"] = self.ls_diff_1st_loss(hrtf_pred, hrtf_gt)
                    loss_dict["ls_diff_2nd"] = self.ls_diff_2nd_loss(hrtf_pred, hrtf_gt)
                    loss_dict["ls_diff_4th"] = self.ls_diff_4th_loss(hrtf_pred, hrtf_gt)
                    loss_dict["is_pm"] = self.IS(hrtf_pred, hrtf_gt)
                    loss_dict["phase_rec"] = self.phase_loss(hrtf_pred, hrtf_gt)
                    # sys.exit()
                    loss_dict["l2_reg"] = sum(p.pow(2.0).sum() for p in self.net.parameters())
                    if hasattr(self.net.module, 'Encoder_z'):
                        loss_dict["l2_reg_en"] = sum(p.pow(2.0).sum() for p in self.net.module.Encoder_z.parameters())

                    if self.config["windowfunc"]:
                        hrir_extra = prediction["hrir_extra"]
                        loss_dict["hrir_extra"] = self.rms_loss(hrir_extra)
                        # loss_hrir_extra = self.rms_loss(hrir_extra)
                    
                    if self.config["np_diff_loss"]:
                        fs = self.config['max_frequency']*2
                        f_us = self.config['fs_upsampling']
                        
                        hrtf_pred_s = hrtf_pred.permute(3,0,1,2) # (B,2,L,S) -> (S,B,2,L)
                        hrtf_gt_s = hrtf_gt.permute(3,0,1,2) # (B,2,L,S) -> (S,B,2,L)

                        hrir_gt_s = posTF2IR_dim4(hrtf_gt_s)
                        _, hrir_min = minphase_recon(tf=hrtf_pred_s)

                        window_length = self.config["window_length_np_diff"]
                        window = th.hann_window(window_length=window_length, periodic=True)[window_length//2:].to(hrir_min.device)
                        window = F.pad(window, (0,hrir_min.shape[-1]-window_length//2))
                        hrir_gt_win  = hrir_gt_s * window[None,None,None,:]
                        hrir_min_win = hrir_min  * window[None,None,None,:]
                        hrtf_gt_win  = th.conj(th.fft.fft(hrir_gt_win,  dim=-1))[:,:,:,1:hrir_min.shape[-1]//2+1]
                        hrtf_min_win = th.conj(th.fft.fft(hrir_min_win, dim=-1))[:,:,:,1:hrir_min.shape[-1]//2+1]
                        hrtf_gt_win  = hrtf_gt_win.permute(1,2,3,0) # (B,2,L,S)
                        hrtf_min_win = hrtf_min_win.permute(1,2,3,0) # (B,2,L,S)

                        loss_dict["ls_diff_gen"] = self.ls_diff_gen_loss(hrtf_min_win, hrtf_gt_win)


                    # if itr==0 and itr_numpts==0:
                    if data['sub_idx'][0] < self.config['sub_index'][data['db_name'][0]]['train'][1] and itr_numpts==0:
                        # for sub_id in sub[:min(3,len(sub))]:
                        for sub_id in range(min(3,len(sub))):
                            # print(srcpos.shape) # torch.Size([440, 3, 4])
                            # print(hrtf_gt.shape) # torch.Size([440, 2, 128, 4])
                            # print(hrtf_pred.shape)
                            plotmaghrtf(srcpos=srcpos[:,:,sub_id],hrtf_gt=hrtf_gt[:,:,:,sub_id],hrtf_pred=hrtf_pred[:,:,:,sub_id], idx_plot_list=np.array(self.config["idx_plot_list"][data['db_name'][0]]),config=self.config,  figdir=f"{self.config['artifacts_dir']}/figure/HRTF/", fname=f"{self.config['artifacts_dir']}/figure/HRTF/HRTF_Mag_{data['db_name'][0]}-sub-{data['sub_idx'][0]+1}_{mode}")

                            mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude', top_db = 80) 
                            plot_mag_Angle_vs_Freq(HRTF_mag=mag2db(th.abs(hrtf_pred[...,sub_id])), SrcPos=srcpos[:,:,sub_id], fs=self.config['max_frequency'], figdir=f"{self.config['artifacts_dir']}/figure/HRTF_mag_2D/", fname=f"{self.config['artifacts_dir']}/figure/HRTF_mag_2D/HRTF_Mag_{data['db_name'][0]}-sub-{data['sub_idx'][0]+1}_{mode}")

                            ## plot HRIRs
                            if self.config["minphase_recon"]:
                                fs = self.config['max_frequency']*2
                                f_us = self.config['fs_upsampling']
                                
                                hrtf_pred_s = hrtf_pred.permute(3,0,1,2) # (B,2,L,S) -> (S,B,2,L)
                                hrtf_gt_s = hrtf_gt.permute(3,0,1,2) # (B,2,L,S) -> (S,B,2,L)
                
                                hrir_gt_s = posTF2IR_dim4(hrtf_gt_s)
                                if self.config["use_itd_gt"]:
                                    itd_des = hrir2itd(hrir=hrir_gt_s, fs=fs, f_us=f_us)
                                else:
                                    raise NotImplementedError()
                                phase_min, hrir_min = minphase_recon(tf=hrtf_pred_s)
                                itd_ori = hrir2itd(hrir=hrir_min, fs=fs, f_us=f_us)
                                
                                hrir_min_itd = assign_itd(hrir_ori=hrir_min, itd_ori=itd_ori.to(hrir_min.device), itd_des=itd_des.to(hrir_min.device), fs=fs)
                                hrir_pred_s = hrir_min_itd.permute(1,2,3,0)
                                hrir_gt_s = hrir_gt_s.permute(1,2,3,0)
                                # print(hrir_pred_s.shape)
                                # print(hrir_gt_s.shape)

                                _, _ = plothrir(srcpos=srcpos[:,:,sub_id],hrtf_gt=hrtf_gt[:,:,:,sub_id],hrtf_pred=hrtf_pred,idx_plot_list=np.array(self.config["idx_plot_list"][data['db_name'][0]]), config=self.config, hrir_gt=hrir_gt_s[:,:,:,sub_id], hrir_pred=hrir_pred_s[:,:,:,sub_id], figdir=f"{self.config['artifacts_dir']}/figure/HRIR/", fname=f"{self.config['artifacts_dir']}/figure/HRIR/HRIR_{data['db_name'][0]}-sub-{sub_id+1}_{mode}")
                            else:
                                _, _ = plothrir(srcpos=srcpos[:,:,sub_id],hrtf_gt=hrtf_gt[:,:,:,sub_id],hrtf_pred=hrtf_pred[:,:,:,sub_id],idx_plot_list=np.array(self.config["idx_plot_list"][data['db_name'][0]]), config=self.config, figdir=f"{self.config['artifacts_dir']}/figure/HRIR/", fname=f"{self.config['artifacts_dir']}/figure/HRIR/HRIR_{data['db_name'][0]}-sub-{sub_id+1}_{mode}")

                            
                            if  self.config["model"] == "AE" and self.config["windowfunc"]:
                                plotmaghrtf(srcpos=srcpos[:,:,sub_id],hrtf_gt=hrtf_gt[:,:,:,sub_id],hrtf_pred=prediction["output_nowindow"][:,:,:,sub_id], idx_plot_list=np.array(self.config["idx_plot_list"][data['db_name'][0]]),config=self.config,  figdir=f"{self.config['artifacts_dir']}/figure/HRTF/", fname=f"{self.config['artifacts_dir']}/figure/HRTF/HRTF_Mag_{data['db_name'][0]}-sub-{sub_id+1}_{mode}_nowindow")
                                _, _ = plothrir(srcpos=srcpos[:,:,sub_id],hrtf_gt=hrtf_gt[:,:,:,sub_id],hrtf_pred=prediction["output_nowindow"][:,:,:,sub_id],mode=f'train_{sub_id+1}_nowindow',idx_plot_list=np.array(self.config["idx_plot_list"][data['db_name'][0]]), config=self.config, figdir=f"{self.config['artifacts_dir']}/figure/HRIR/", fname=f"{self.config['artifacts_dir']}/figure/HRIR/HRIR_{data['db_name'][0]}-sub-{sub_id+1}_{mode}")

                    if  self.config["model"] in ["AE", "CAE", "HCAE", "CNN"]:
                        if self.config["Decoder"].startswith("coeff"):
                            coeff_pred = prediction["coeff"]
                            # print(coeff_pred.device)
                            # print(coeff.device)
                            loss_dict["reg"] = self.reg_loss(coeff_pred, SpecialFunc(maxto=self.config["max_truncation_order"]).n_vec)
                            # reg = self.reg_loss(coeff_pred, SpecialFunc(maxto=self.config["max_truncation_order"]).n_vec)

                            # print(coeff_pred.shape) # torch.Size([2, 361, 128, 4])
                            # print(coeff.shape) # torch.Size([2, 361, 128, 4])
                            loss_dict["l2_c"] = self.mse_loss(coeff_pred , coeff)
                            loss_dict["l2_c_n"] = self.nmse_loss(coeff_pred , coeff)
                            loss_dict["huber_c"] = self.huber_loss(coeff_pred , coeff)
                            loss_dict["lsd_c"] = self.lsd_loss(coeff_pred , coeff)
                            loss_dict["lsd_cw"] = self.Weightedlsd_loss(coeff_pred , coeff)
                            # loss_dict["phase_c"] = self.phase_loss(coeff_pred , coeff)
                            # print(loss_dict["phase_c"])
                            # sys.exit()
                        if self.config["num_classifier"] > 1:
                            for i in range(self.config["num_classifier"]):
                                exec(f'z_{i} = prediction[f"z_{i}"]')
                                # loss_dict[f"var_z_{i}"] = self.var_loss(eval(f'z_{i}'))
                                loss_dict[f"cdintra_z_{i}"] = self.cosdist(eval(f'z_{i}'))
                                loss_dict[f"cdintrasq_z_{i}"] = self.cosdistsq(eval(f'z_{i}'))

                        if self.config["use_metric"]:
                            if self.config["num_classifier"] == 1:
                                out_metric = prediction['metric'] 
                                label = prediction['label'] 
                                loss_dict["cross_entropy"] = self.cross_entropy_loss(out_metric, label)

                                accuracy = th.sum(th.argmax(out_metric,dim=1)==label)/len(label)
                                returns["accuracy"] = accuracy
                            elif self.config["num_classifier"] > 1:
                                for i in range(self.config["num_classifier"]):
                                    exec(f'z_{i} = prediction[f"z_{i}"]')
                                    loss_dict[f"cross_entropy_{i}"] = self.cross_entropy_loss(prediction[f'metric_{i}'], prediction[f'label_{i}'])
                                    returns[f"accuracy_{i}"] = th.sum(th.argmax(prediction[f'metric_{i}'],dim=1)==prediction[f'label_{i}'])/len(prediction[f'label_{i}'])
                        
                        helm = 0
                        returns["z"] = prediction["z"].detach().clone()
                        returns["coeff"] = prediction["coeff"].detach().clone()
                        if self.config["model"]=="CNN":
                            returns["coeff_in"]=prediction["coeff_in"].detach().clone()

                        
                    elif self.config["model"] == "AE_PINN":
                        reg = 0
                        l2_c = 0
                        # helm = self.helmholtz_loss(rf=prediction['rf'], thf=prediction['thf'], phf=prediction['phf'], p=prediction['output_4ch'], k=prediction['k'], L=prediction['L'], B=prediction['B'])
                        helm = self.helmholtz_loss_cart(xf=prediction['xf'], yf=prediction['yf'], zf=prediction['zf'], func=prediction['output_4ch'], k=prediction['k'], L=prediction['L'], B=prediction['B'])
                        loss_dict["helmholtz"] = helm
                        # returns_new = {
                        # "helmholtz": helm.detach().clone()
                        # }
                    # returns.update(returns_new)

                    if self.config["model"] == "FIAE":
                        if self.config["weight_en_sph_harm"]:
                            loss_dict["l2_w_en_sh"] = self.mse_loss(prediction["weight_en_1"][0,0,:,:],prediction["RSH_en"])
                        if self.config["weight_de_sph_harm"]:
                            loss_dict["l2_w_de_sh"] = self.mse_loss(prediction["weight_de_2"][0,0,:,:],prediction["RSH_de"])
                            # loss_dict["l2_w_de_sh"] = self.mse_loss(prediction["weight_de_2"][0,0,:,:],prediction["RSH_en"])
                        if self.config["aggregation_mean"] and self.config["loss_weights"]["var_z"] > 0:
                            loss_dict["var_z"] = self.var_loss(prediction["z_bm"].permute(2,0,1,3))
                        if self.config["aggregation_mean"] and self.config["loss_weights"]["cdintra_z"] > 0:
                            loss_dict["cdintra_z"] = self.cosdist(prediction["z_bm"].permute(2,0,1,3))
                    # sys.exit()

        
                    loss = 0
                    for k,v in loss_dict.items():
                        # print(k)
                        if k in self.config["loss_weights"]:
                            if self.config["loss_weights"][k] > 0:
                                # print(k)
                                loss += self.config["loss_weights"][k] * v
                        returns[k] = v.detach().clone()

                else:
                    prediction = self.net.forward(srcpos, True)
                    # print(hrtf_gt.shape) # B,2,L
                    l2 = self.mse_loss(prediction["output"], hrtf_gt)
                    lsd = self.lsd_loss(prediction["output"], hrtf_gt)
                    if self.config["Decoder"].startswith("coeff"):
                        reg = self.reg_loss(prediction["coeff"], SpecialFunc(maxto=self.config["max_truncation_order"]).n_vec)
                    loss = l2 * self.config["loss_weights"]["l2"] + reg * self.config["loss_weights"]["reg"] + lsd * self.config["loss_weights"]["lsd"]
                    returns = {
                        "l2": l2.detach().clone(), 
                        "reg": reg.detach().clone(),
                        "lsd": lsd.detach().clone(),
                    }
                    
                    if self.config["model"] == "PINN":
                        helmholtz = self.helmholtz_loss(self.net, self.net.module(srcpos)["f"], srcpos) 
                        # print(helmholtz.shape)
                        loss += helmholtz * self.config["loss_weights"]["helmholtz"]
                        returns["helmholtz"] = helmholtz.detach().clone()

                returns["loss"] = loss.detach().clone()
                # update model parameters
                loss.backward()
                # for k,v in prediction.items():
                #     print(f"{k}:{v.grad}")
                # print(f"srcpos:{srcpos.grad}")
                # sys.exit()

                self.optimizer.step()
                self.total_iters += 1
                
                return returns