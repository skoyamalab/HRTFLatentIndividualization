import torch as th
import torch.nn.functional as F
import torchaudio as ta
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm
import sofa
import numpy as np
import scipy.io
import csv
import pandas as pd
from icecream import ic
ic.configureOutput(includeContext=True)

import matplotlib.pyplot as plt

import sys, os
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Now you can import the module
from utils import sph2cart

from models import HRTFApproxNetwork_FIAE, replace_activation_function
output_dir = "./outputs/out_20240918_FIAE_500239/"

class MultiDataset(Dataset):
    def __init__(self,
                 sub_indices, # dict: dbname dict: ranges
                 phase = 'train',
                 filter_length = 128,
                 max_f = 20000,
                 ):
        super().__init__()

        # db_names = ["CHEDAR", "HUTUBS", "CIPIC"]
        db_names = sub_indices.keys()

        self.data = []

        for db_name in db_names:
            all_anthro = self.anthro2data(db_name)

            # Getting HRTF magnitude data
            if db_name == "CIPIC":
                id_file = '../CIPIC/anthropometry/id.csv'
                id_data = pd.read_csv(id_file, header=None)
                cipic_ids = id_data.iloc[:, 0].tolist()

            for sub_id in sub_indices[db_name][phase]:
                if sub_id in sub_indices[db_name].get('skip', []):
                    continue
                if db_name == "CHEDAR":
                    path = f"../CHEDAR/chedar_{sub_id+1:04}_UV2m.sofa"
                elif db_name == "HUTUBS":
                    path = f"../HUTUBS/HRIRs/pp{sub_id+1}_HRIRs_measured.sofa"
                elif db_name == "CIPIC":
                    path = f"../CIPIC/subject_{cipic_ids[sub_id]:03}.sofa"
                returns = self.sofa2data(path, max_f, filter_length)
                returns["db_name"] = db_name
                returns["AnthroFeatures"] = all_anthro[sub_id]

                # for data_kind in ["HRTF", "HRTF_mag", "HRIR"]:
                #     HRTF_L, HRTF_R = th.split(returns_L[data_kind], 1, dim=1)
                #     # if sub_id == 0 and data_kind == "HRTF_mag":
                #     #     ic(db_name)
                #     #     ic(HRTF_L.shape, HRTF_R.shape)
                #     #     ic(HRTF_L, HRTF_R)
                #     returns_L[data_kind] = HRTF_L
                #     returns_R[data_kind] = HRTF_R

                self.data.append(returns)
                #ic(returns)

        ### Standardize every data_kind in data
        ### only use this on training data
        self.Val_Standardize = {}
        self.Anthro_Standardize = {}
        for db_name in db_names:
            # print(f"\nStandardizing {db_name} data")

            self.Val_Standardize[db_name] = {}

            # Collect all data for the current db_name
            db_data = [d for d in self.data if d["db_name"] == db_name]

            for data_kind in ["SrcPos", "SrcPos_Cart", "HRTF", "HRTF_mag", "HRIR"]:
                # Collect all data for the current data_kind in current db_name
                all_data = [d[data_kind] for d in db_data]
                
                # Convert to a single tensor
                all_data_tensor = th.stack(all_data)
                # ic(all_data_tensor)

                # TODO: old standardization method
                # mean, std = th.mean(all_data_tensor).to('cuda'), th.std(all_data_tensor).to('cuda')
                # print(f'[{data_kind}] mean:{mean:.4}, std:{std:.4}')

                # This is to standardize across the S dimension
                # if data_kind in ["HRTF", "HRTF_mag", "HRIR"]:
                if False:
                    # Assuming all_data_tensor has the shape (S, B, 2, F)
                    S, B, _, F = all_data_tensor.shape

                    # Reshape to (S * 2, B, 1, F)
                    reshaped_tensor = all_data_tensor.view(S * 2, B, 1, F)

                    # Step 2: Calculate the mean and std across the S dimension, then duplicate
                    mean = reshaped_tensor.mean(dim=0).repeat(1, 2, 1).to('cuda')  # Shape: (B, 2, F)
                    std = reshaped_tensor.std(dim=0).repeat(1, 2, 1).to('cuda')    # Shape: (B, 2, F)
                else:
                    # Compute mean and standard deviation
                    mean = th.mean(all_data_tensor, dim=0).to('cuda')
                    std = th.std(all_data_tensor, dim=0).to('cuda')
                # ic(db_name, data_kind, mean.shape, std.shape, mean, std)
                # Store in Val_Standardize
                self.Val_Standardize[db_name][data_kind] = {
                    "mean": mean,
                    "std": std
                }
        
            stacked_data = th.stack([item["AnthroFeatures"] for item in self.data if item["db_name"] == db_name]).view(-1, 18)
            # ic(stacked_data.shape)

            mean = th.mean(stacked_data, dim=0).to('cuda')
            std = th.std(stacked_data, dim=0).to('cuda')
            # ic(mean, std)
            self.Anthro_Standardize[db_name] = {
                "mean": mean,
                "std": std
            }
            # ic(self.Anthro_Standardize[db_name])

    def __len__(self):
        '''
        :return: number of training subjects in dataset
        '''
        return len(self.data)

    def __getitem__(self, index):
        '''
        :return: dict consisting of
            db_name as         str
            SrcPos as          B x 3 tensor
            SrcPosCart as      B x 3 tensor
            HRTF as            B x 2 (LR) x L tensor
            HRTF_mag as        B x 2 (LR) x L tensor
            HRIR as            B x 2 (LR) x 2L tensor
            AnthroFeatures as  2 (LR) x 18 tensor (x1-8, x12, d1-7, t1-2)
                                d and t are different for left and right ear
        '''
        return self.data[index]
    
    def anthro2data(self, db_name):
        l_data, r_data = None, None
        if db_name == "CHEDAR":
            mat_contents = scipy.io.loadmat("../CHEDAR/measurements_array.mat")
            tensor_data = th.tensor(mat_contents['measures_array'])

            # TODO: do we actually need to duplicate these...
            # d and t data are left ear only, but because the head meshes are simulated, we can assume they are the same for both ears
            # chedar dataset has identical LR ears but different HRTFs per ear, will this be an issue?
            x_data = th.cat((tensor_data[:, 10].unsqueeze(1), tensor_data[:, 17].unsqueeze(1), tensor_data[:, 20].unsqueeze(1), tensor_data[:, -7:-2], tensor_data[:, 13].unsqueeze(1)), dim=1)
            d_data = th.cat((tensor_data[:, 0].unsqueeze(1), tensor_data[:, 2:8]), dim=1)
            t_data = th.deg2rad(tensor_data[:, 8:10])

            # selected_data = th.cat((x_data, d_data, d_data, t_data, t_data), dim=1)

            l_data = th.cat((x_data, d_data, t_data), dim=1)
            r_data = th.cat((x_data, d_data, t_data), dim=1)
        elif db_name == "HUTUBS":
            xlsx_contents = pd.read_excel("../HUTUBS/Antrhopometric measures/AntrhopometricMeasures.xlsx")

            x_data = th.cat((th.tensor(xlsx_contents.iloc[:, 1:9].values), th.tensor(xlsx_contents.iloc[:, 10].values).unsqueeze(1)), dim=1)
            L_d_data = th.tensor(xlsx_contents.iloc[:, 14:21].values)
            R_d_data = th.tensor(xlsx_contents.iloc[:, 26:33].values)
            L_t_data = th.deg2rad(th.tensor(xlsx_contents.iloc[:, 24:26].values))
            R_t_data = th.deg2rad(th.tensor(xlsx_contents.iloc[:, -2:].values))

            # selected_data = th.cat((x_data, L_d_data, R_d_data, L_t_data,  R_t_data), dim=1)

            l_data = th.cat((x_data, L_d_data, L_t_data), dim=1)
            r_data = th.cat((x_data, R_d_data, R_t_data), dim=1)
        elif db_name == "CIPIC":
            mat_contents = scipy.io.loadmat("../CIPIC/anthropometry/anthro.mat")

            x_data = th.tensor(mat_contents['X'])
            d_data = th.tensor(mat_contents['D'])
            t_data = th.tensor(mat_contents['theta'])

            # selected_data = th.cat((x_data[:, :8], x_data[:, 11].unsqueeze(1), d_data[:, :7], d_data[:, 8:15], t_data), dim=1)

            l_data = th.cat((x_data[:, :8], x_data[:, 11].unsqueeze(1), d_data[:, :7], t_data[:, :2]), dim=1)
            r_data = th.cat((x_data[:, :8], x_data[:, 11].unsqueeze(1), d_data[:, 8:15],t_data[:, -2:]), dim=1)
        # return l_data.to(th.float32), r_data.to(th.float32)
        # ic(db_name, l_data.shape, r_data.shape, th.stack((l_data, r_data), dim=1).shape)
        return th.stack((l_data, r_data), dim=1).to(th.float32)

    def sofa2data(self,path,max_f,filter_length):
        mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude', top_db = 80) 
        SOFA = sofa.Database.open(path)
        srcpos_ori = th.tensor(SOFA.Source.Position.get_values()) # azimuth in [0,360),elevation in [-90,90], radius in {1.47}
        srcpos_sph = srcpos_ori
        srcpos_sph[:,0] = srcpos_sph[:, 0] % 360 # azimuth in [0,360)
        srcpos_sph[:,1] = 90 - srcpos_sph[:, 1] # elevation in [-90,90] -> zenith in [180,0] # "%180" は[0,180) にしてしまうのでダメ
        srcpos_sph[:,:2] = srcpos_sph[:,:2] / 180 * np.pi # azimuth in [0,2*pi), zenith in [0,pi]
        srcpos_sph = th.cat((srcpos_sph[:,2].unsqueeze(1), srcpos_sph[:,:2]), dim=1) # radius, azimuth in [0,2*pi), zenith in [0,pi]

        self.sr_ori = SOFA.Data.SamplingRate.get_values()[0] # 44100
        HRIR = th.tensor(SOFA.Data.IR.get_values()) # 440 x 2 x 256
        self.fft_length_ori = HRIR.shape[-1] # 256

        # Downsampling
        downsampler = ta.transforms.Resample(self.sr_ori,2*max_f, dtype=th.float32)
        HRIR_us = downsampler(HRIR.to(th.float32))

        # time alignment
        idx_plot_list = [0, 1, 2, 3]
        max_idx_front = round(th.mean(th.argmax(HRIR_us[idx_plot_list[0],:,:], dim=-1).to(th.float32)).item())
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
        
        # ic(srcpos_sph.shape)
        returns = {
            'SrcPos': srcpos_sph.to(th.float32),
            'SrcPos_Cart': sph2cart(srcpos_sph[:,1], srcpos_sph[:,2], srcpos_sph[:,0]).to(th.float32),
            'HRTF': HRTF,
            'HRTF_mag': mag2db(th.abs(HRTF)),
            'HRIR': HRIR_us
        }
        
        # Shape: (B, LR=2, filter_length)
        return returns

sub_indices = {
    'CHEDAR': {
        'train': range(0, 1003), # 1003
        'valid': range(1003, 1128), # 125
        'test':  range(1128, 1253), # 125
        'all':   range(0, 1253)
    },
    # 'HUTUBS': {
    #     'train': range(0, 77), # 77 - 1
    #     'valid': range(77, 87), # 10 - 1
    #     'test':  range(88, 95), # 7 - 1, remove duplication: 95==0, 87==21
    #     'all':   range(0, 96), # 96
    #     'skip': [17, 78, 91] # 17, 78, 91 have missing anthropometric features
    # },
    # 'CIPIC': {
    #     'train': range(0, 35), # 35 - 7
    #     'valid': range(35, 40), # 5
    #     'test':  range(40, 45), # 5 - 1
    #     'all':   range(0, 45), # 45 subjects but only 43 have anthropometric features
    #     'skip': [1, 2, 4, 5, 6, 7, 9, 41] # 1-2, 4-7, 9, 41 have missing anthropometric features
    # },
}

if __name__ == "__main__":
    train_dataset = MultiDataset(sub_indices, phase="train")
    ic(len(train_dataset)) # 1107 * 2 subjects
    valid_dataset = MultiDataset(sub_indices, phase="valid")
    ic(len(valid_dataset)) # 139 * 2 subjects
    test_dataset = MultiDataset(sub_indices, phase="test")
    ic(len(test_dataset)) # 135 * 2 subjects

    # Val_Standardize = train_dataset.Val_Standardize
    # Anthro_Standardize = train_dataset.Anthro_Standardize
    # ic(Val_Standardize, Anthro_Standardize)
