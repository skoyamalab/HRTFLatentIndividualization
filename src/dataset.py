import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta
import numpy as np
import pandas as pd
import sofa
from src.utils import cart2sph, sph2cart
from icecream import ic
# ic.configureOutput(includeContext=True)
# ic.disable()

class HRTFDataset:
    '''
    sofa_path: (str) path to .sofa file
    max_f: (int) max_frequency
    filter_length: (int) filter_length (positive freq. bins)
    '''
    def __init__(self,
                 config,
                 filter_length = 65,
                 max_f = 16000,
                 sofa_path = "data/mit_kemar_normal_pinna.sofa", 
                 debug = False
                 ):
        super().__init__()
        self.sofa_path = sofa_path
        self.debug = debug
        self.config = config
        mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude', top_db = 80) 
        if sofa_path == "data/mit_kemar_normal_pinna.sofa":
            SOFA = sofa.Database.open(sofa_path)
            srcpos = th.tensor(SOFA.Source.Position.get_values()) # azimuth, zenith, radius
            srcpos[:,0] = srcpos[:,0] % 360 # azimuth: [0,360]
            srcpos[:,1] = srcpos[:,1] #% 180 # zenith: [0,180]
            self.srcpos = th.cat((srcpos[:,2].unsqueeze(dim=1), th.deg2rad(srcpos[:,:2])), dim=1) # radius, azimuth in [0,2*pi], zenith in [0,pi]

            self.sr_ori = SOFA.Data.SamplingRate.get_values()[0]
            self.HRIR = th.tensor(SOFA.Data.IR.get_values())
            self.fft_length_ori = self.HRIR.shape[-1]
            
            # iFFT
            HRTF_pm = th.fft.ifft(self.HRIR, dim=-1)
            # Estract DC & positive frequency
            HRTF = HRTF_pm[:,:,:round(self.fft_length_ori/2)+1]

            # Upsampling
            pf_us = 1
            upsampler = nn.Upsample(size=round(self.sr_ori/2/pf_us), mode='linear', align_corners=True)
            HRTF_re_us = upsampler(th.real(HRTF))
            HRTF_im_us = upsampler(th.imag(HRTF))
            HRTF_us = th.complex(HRTF_re_us,HRTF_im_us)

            # Downsampling
            HRTF_ds = HRTF_us[:,:,::round(max_f/(filter_length-1)/pf_us)]
            self.HRTF = HRTF_ds[:,:,:filter_length]
        elif sofa_path.startswith("data/EvaluationGrid_"):
            r_str_list = ['05','10','15','20','25']
            # r_str_list = ['05'] # for debug

            for r_str in r_str_list:
                path = "data/EvaluationGrid_" + r_str + ".sofa"
                SOFA = sofa.Database.open(path)
                srcpos_cart = th.tensor(SOFA.Source.Position.get_values()) # x,y,z
                srcpos_sph = cart2sph(srcpos_cart[:,0], srcpos_cart[:,1], srcpos_cart[:,2]) # radius, azimuth in [0,2*pi], zenith in [0,pi]

                HRTF_re = th.from_numpy(SOFA.Data.Real.get_values().astype(np.float32)).clone()
                HRTF_im = th.from_numpy(SOFA.Data.Imag.get_values().astype(np.float32)).clone() 
                HRTF = th.complex(HRTF_re, HRTF_im)
                HRTF = F.pad(HRTF,[1,0])
                # Downsampling
                pf_ori = SOFA.N.get_values()[0] # 125 Hz by default
                HRTF_ds = HRTF[:,:,::round(max_f/(filter_length)/pf_ori)]
                # print(max_f/(filter_length)/pf_ori)
                # print(round(max_f/(filter_length)/pf_ori))
                
                if r_str == '05':
                    self.srcpos = srcpos_sph
                    self.HRTF = HRTF_ds
                else:
                    self.srcpos = th.cat((self.srcpos, srcpos_sph), dim=0)
                    self.HRTF = th.cat((self.HRTF, HRTF_ds), dim=0)                 
            
            self.HRTF = self.HRTF[:,:,1:]
            # self.srcpos = self.srcpos.cuda()
            # print(self.srcpos.shape) # torch.Size([35410, 3])
            # self.HRTF = self.HRTF.cuda()
            # print(self.HRTF.shape) # torch.Size([35410, 2, 64])

            # shuffle
            th.manual_seed(0)
            th.cuda.manual_seed(0)
            len_all = self.HRTF.shape[0]
            idx_rand = th.randperm(len_all)
            self.HRTF = self.HRTF[idx_rand]
            self.srcpos = self.srcpos[idx_rand]

            # set rate of {train/valid/test}
            train_data_rate = 0.8
            valid_data_rate = 0.1
            # test_data_rate = 0.1

            # # devide data
            # self.HRTF_train = self.HRTF[:round(len_all*train_data_rate)]
            # self.srcpos_train = self.srcpos[:round(len_all*train_data_rate)]
            # for comparison to linear inverse problem
            # self.HRTF_train = self.HRTF[:64]
            # self.srcpos_train = self.srcpos[:64]
            self.HRTF_train = self.HRTF[:2048]
            self.srcpos_train = self.srcpos[:2048]
            # print(self.HRTF_train.shape) # torch.Size([28328, 2, 65])
            # print(self.srcpos_train.shape) # torch.Size([28328, 3])

            self.HRTF_valid = self.HRTF[round(len_all*train_data_rate):round(len_all*train_data_rate)+round(len_all*valid_data_rate)]
            self.srcpos_valid = self.srcpos[round(len_all*train_data_rate):round(len_all*train_data_rate)+round(len_all*valid_data_rate)]
            # print(self.HRTF_valid.shape) # torch.Size([3541, 2, 65])
            # print(self.srcpos_valid.shape) # torch.Size([3541, 3])

            self.HRTF_test = self.HRTF[round(len_all*train_data_rate)+round(len_all*valid_data_rate):]
            self.srcpos_test = self.srcpos[round(len_all*train_data_rate)+round(len_all*valid_data_rate):]
        else:
            # self.Data[data_for][data_kind][db_name]
            self.Data = {}
            for data_for in ["all", "train", "valid", "test"]:
                self.Data.setdefault(data_for, {})
                for data_kind in ['SrcPos', 'SrcPos_Cart', 'HRTF', 'HRTF_mag', 'HRIR']: #,'ITD']
                    self.Data[data_for].setdefault(data_kind, {})

            for db_name in config["database"]: # ['HUTUBS', 'RIEC']
                # ic(db_name)
                if db_name == "CIPIC":
                    id_file = '../CIPIC/anthropometry/id.csv'
                    id_data = pd.read_csv(id_file, header=None)
                    cipic_ids = id_data.iloc[:, 0].tolist()
                for sub_id in self.config["sub_index"][db_name]['all']:
                    if db_name == 'SONICOM':
                        path = f"../SONICOM/P0001-P0200/P0{sub_id+1:03}/HRTF/HRTF/48kHz/P0{sub_id+1:03}_FreeFieldCompMinPhase_48kHz.sofa"
                    elif db_name == 'HUTUBS':
                        path = f"../HUTUBS/HRIRs/pp{sub_id+1}_HRIRs_measured.sofa"
                        # path = "../HUTUBS/HRIRs/pp{sub_id+1}_HRIRs_simulated.sofa"
                    elif db_name == 'RIEC':
                        path = f"../RIEC/data/RIEC_hrir_subject_{sub_id+1:03}.sofa"
                    elif db_name == 'SADIE2':
                        path = f"../SADIE2/H{sub_id+3}/H{sub_id+3}_HRIR_SOFA/H{sub_id+3}_48K_24bit_256tap_FIR_SOFA.sofa"
                        print(path)
                    elif db_name == 'CHEDAR': ####################
                        path = f"../CHEDAR/chedar_{sub_id+1:04}_UV2m.sofa"
                    elif db_name == 'Own':
                        # path = f"./data/{self.config['own_hrtf_sub_name'][sub_id]}.sofa"
                        # path = f"./data/{self.config['own_hrtf_sub_name'][sub_id]}_up.sofa"
                        path = f"./data/full_name/{self.config['own_hrtf_sub_name'][sub_id]}_up.sofa"
                    elif db_name == 'LAPChallenge':
                        path = f"../LAPChallenge/task2/LAP Task 2 Sparse HRTFs/LAPtask2_all_{sub_id+1}.sofa"
                    elif db_name == 'MIT':
                        path = f"./data/mit_kemar_normal_pinna.sofa"
                    elif db_name == 'CIPIC':
                        path = f"../CIPIC/subject_{cipic_ids[sub_id]:03}.sofa"
                    else:
                        raise NotImplementedError

                    returns = self.sofa2data(path,max_f,filter_length,self.config["green"],db_name)
                    for data_kind in self.Data["all"]:
                        if data_kind == 'ITD':
                            pass
                        else:
                            if sub_id == 0:
                                self.Data["all"][data_kind][db_name] = th.zeros_like(returns[data_kind]).unsqueeze(-1).tile(th.cat((th.ones(returns[data_kind].dim(), dtype=int), th.tensor([len(self.config["sub_index"][db_name]['all'])])),0).tolist())
                                if data_kind in ['SrcPos', 'SrcPos_Cart']:
                                    self.Data["all"][data_kind][db_name] = self.Data["all"][data_kind][db_name].to(th.float32)
                            self.Data["all"][data_kind][db_name][..., sub_id] = returns[data_kind]
                
                # self.Data["all"]["ITD"][db_name] = hrir2itd(hrir=self.Data["all"]["HRIR"][db_name].permute(3,0,1,2), fs = self.config['max_frequency']*2, f_us = self.config['fs_upsampling']).permute(1,0).cpu() # B x S

                for data_for in ['train','valid','test']:
                    for data_kind in self.Data['all']:
                        self.Data[data_for][data_kind][db_name] = self.Data["all"][data_kind][db_name][..., self.config["sub_index"][db_name][data_for]]
            self.Val_Standardize = {}
            for data_kind in self.Data["train"]:
                self.Val_Standardize[data_kind] = {}
                for db_name in self.config['database']:
                    if db_name == 'Own':
                        self.Val_Standardize[data_kind]['Own'] = self.Val_Standardize[data_kind]['RIEC'].copy()
                    if db_name == 'LAPChallenge':
                        self.Val_Standardize[data_kind]['LAPChallenge'] = self.Val_Standardize[data_kind]['SONICOM'].copy()
                    if len(config["sub_index"][db_name]['train']) > 0:
                        # if data_kind == 'HRTF':
                        #     val = mag2db(th.abs(self.Data['train'][data_kind][db_name]))/20
                        # else:
                        val = self.Data["train"][data_kind][db_name]
                        ic(val.shape)

                        # print(f'[{data_kind}-{db_name}] mean:{th.mean(val):.4}, std:{th.std(val):.4}') 
                        # TODO: this changed
                        self.Val_Standardize[data_kind][db_name] = {
                            "mean": th.mean(val, dim=-1),
                            "std":  th.std(val, dim=-1)
                        }
                        # ic(self.Val_Standardize[data_kind][db_name]["mean"].shape)
                        # ic(self.Val_Standardize[data_kind][db_name]["std"].shape)
                # self.Val_Standardize[data_kind]['Own'] = self.Val_Standardize[data_kind]['RIEC'].copy()
            
            self.Table = {}
            for data_for in ['train','valid','test']:
                self.Table[data_for] = {
                    "db_name": [],
                    "sub_idx": []
                }
                for db_name in self.config["database"]:
                    self.Table[data_for]["db_name"].extend([db_name for _ in self.config["sub_index"][db_name][data_for]])
                    self.Table[data_for]["sub_idx"].extend(list(self.config["sub_index"][db_name][data_for]))

    #------------------------------------

    def __len__(self):
        '''
        :return: number of training subjects in dataset
        '''
        return sum([len(self.config['sub_index'][db_name]['train']) for db_name in self.config['database']])

    def __getitem__(self, index): # for train data
        '''
        :return: dict consisting of
            SrcPos as      B x 3 tensor
            HRTF as        B x 2 x L tensor
            HRTF_mag as    B x 2 x L tensor
            HRIR as        B x 2 x 2L tensor
            ITD  as        B tensor
            db_name as     list of str
            sub_idx as     list of? int
        '''
        db_name = self.Table['train']["db_name"][index]
        sub_idx = self.Table['train']["sub_idx"][index]
        returns = {data_kind: self.Data['train'][data_kind][db_name][..., sub_idx] for data_kind in self.Data['train']}
        returns['db_name'] = db_name
        returns['sub_idx'] = sub_idx
        return returns
    
    def trainitem(self): # for validation data
        '''
        :return: dict
                return[data_kind][db_name]: tensor
        '''
        return self.Data['train']
    
    def validitem(self): # for validation data
        '''
        :return: dict
                return[data_kind][db_name]: tensor
        '''
        return self.Data['valid']
    
    def testitem(self): # for test data
        '''
        :return: dict
                return[data_kind][db_name]: tensor
        '''
        return self.Data['test']
    
    def sofa2data(self,path,max_f,filter_length,multiple_green_func=False,db_name='HUTUBS'):
        mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude', top_db = 80) 
        SOFA = sofa.Database.open(path)
        srcpos_ori = th.tensor(SOFA.Source.Position.get_values()) # azimuth in [0,360),elevation in [-90,90], radius in {1.47}
        srcpos_sph = srcpos_ori
        srcpos_sph[:,0] = srcpos_sph[:, 0] % 360 # azimuth in [0,360)
        srcpos_sph[:,1] = 90 - srcpos_sph[:, 1] # elevation in [-90,90] -> zenith in [180,0] # "%180" は[0,180) にしてしまうのでダメ
        srcpos_sph[:,:2] = srcpos_sph[:,:2] / 180 * np.pi # azimuth in [0,2*pi), zenith in [0,pi]
        if db_name == 'RIEC':
            srcpos_sph[:,2] = 1.5
        srcpos_sph = th.cat((srcpos_sph[:,2].unsqueeze(1), srcpos_sph[:,:2]), dim=1) # radius, azimuth in [0,2*pi), zenith in [0,pi]

        self.sr_ori = SOFA.Data.SamplingRate.get_values()[0] # 44100
        HRIR = th.tensor(SOFA.Data.IR.get_values()) # 440 x 2 x 256
        self.fft_length_ori = HRIR.shape[-1] # 256

        # Downsampling
        downsampler = ta.transforms.Resample(self.sr_ori,2*max_f, dtype=th.float32)
        HRIR_us = downsampler(HRIR.to(th.float32))

        # time alignment
        max_idx_front = round(th.mean(th.argmax(HRIR_us[self.config["idx_plot_list"][db_name][0],:,:], dim=-1).to(th.float32)).item())
        if max_idx_front > 2*max_f*1e-3:
            HRIR_us = HRIR_us[:,:,round(max_idx_front - 2*max_f*1e-3):]
        else:
            HRIR_us = F.pad(input=HRIR_us,pad=(round(2*max_f*1e-3 - max_idx_front),0))
        if 2*filter_length > HRIR_us.shape[-1]:
            HRIR_us = F.pad(input=HRIR_us,pad=(0,2*filter_length-HRIR_us.shape[-1]))
        else:
            # HRIR_us = HRIR_us[:,:,HRIR_us.shape[-1]-2*filter_length:]
            HRIR_us = HRIR_us[:,:,:2*filter_length]

        # FFT & conj(Mesh2HRTFに定義を揃える)
        HRTF_pm = th.conj(th.fft.fft(HRIR_us, dim=-1))
        # Extract positive frequency
        HRTF = HRTF_pm[:,:,1:filter_length+1]
        
        if multiple_green_func:
            r = srcpos_sph[0,0]
            self.freq = th.arange(1,filter_length+1) * max_f/filter_length
            self.k = self.freq * 2 * np.pi / 343.18
            self.green = th.exp(1j* self.k * r) / (4*np.pi*r)
            # * green function
            HRTF = HRTF * self.green[None,None,:]
        # ic(srcpos_sph.shape)
        returns = {
            'SrcPos': srcpos_sph,
            'SrcPos_Cart': sph2cart(srcpos_sph[:,1], srcpos_sph[:,2], srcpos_sph[:,0]),
            'HRTF': HRTF,
            'HRTF_mag': mag2db(th.abs(HRTF)),
            'HRIR': HRIR_us,
        }
        return returns

class HRTFTestset:
    '''
    sofa_path: (str) path to .sofa file
    max_f: (int) max_frequency
    filter_length: (int) filter_length (DC + positive freq. bins)
    '''
    def __init__(self,
                 filter_length = 65,
                 max_f = 16000,
                 sofa_path = "data/mit_kemar_normal_pinna.sofa", 
                 ):
        super().__init__()
        SOFA = sofa.Database.open(sofa_path)
        srcpos = th.tensor(SOFA.Source.Position.get_values()) # azimuth, zenith, radius
        srcpos[:,0] = srcpos[:,0] % 360 # azimuth: [0,360]
        srcpos[:,1] = srcpos[:,1] #% 180 # zenith: [0,180]
        self.srcpos = th.cat((srcpos[:,2].unsqueeze(dim=1), th.deg2rad(srcpos[:,:2])), dim=1) # radius, azimuth in [0,2*pi], zenith in [0,pi]

        self.sr_ori = SOFA.Data.SamplingRate.get_values()[0]
        self.HRIR = th.tensor(SOFA.Data.IR.get_values())
        self.fft_length_ori = self.HRIR.shape[-1]
        
        # iFFT
        HRTF_pm = th.fft.ifft(self.HRIR, dim=-1)
        # Estract DC & positive frequency
        HRTF = HRTF_pm[:,:,:round(self.fft_length_ori/2)+1]

        # Upsampling
        pf_us = 1
        upsampler = nn.Upsample(size=round(self.sr_ori/2/pf_us), mode='linear', align_corners=True)
        HRTF_re_us = upsampler(th.real(HRTF))
        HRTF_im_us = upsampler(th.imag(HRTF))
        HRTF_us = th.complex(HRTF_re_us,HRTF_im_us)

        # Downsampling
        HRTF_ds = HRTF_us[:,:,::round(max_f/(filter_length-1)/pf_us)]
        self.HRTF = HRTF_ds[:,:,:filter_length]

    def item(self):
        '''
        :return: srcpos as B x 3 tensor
                 hrtf as B x 2 x filter_length tensor
        '''
        return self.srcpos, self.HRTF