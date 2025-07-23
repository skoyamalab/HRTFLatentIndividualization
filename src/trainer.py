import time
import numpy as np
import torch as th
from torch.utils.data import DataLoader
from src.utils import NewbobAdam
from src.losses import MAE, LSD, ILD_AE

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
        self.lsd_loss = LSD()
        self.ild_loss = ILD_AE()
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
                bs = self.config["batch_size"]
                slice = range(itr*bs, itr*bs + min(bs, data["SrcPos"].shape[0]))
                label = th.tensor(slice).cuda()
                loss_new = self.train_iteration(data, sub=label, itr=itr, itr_numpts=itr_numpts, mode=mode)
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

        for k in loss_stats:
            loss_stats[k] /= (num_data*len(num_pts_list))
        t_end = time.time()
        
        loss_str = "    ".join([f"{k}:{loss_stats[k]:.4}" for k in sorted(loss_stats.keys())])
        time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
        print(f"epoch {epoch} (train) ")
        print(loss_str + "        " + time_str)
        return loss_stats

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