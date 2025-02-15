import torch as th
import torch.nn.functional as F
import torchaudio as ta
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sofa
import numpy as np
import scipy.io
from icecream import ic
ic.configureOutput(includeContext=True)

import sys, os
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Now you can import the module
from utils import sph2cart

class CHEDARDataset(Dataset):
    def __init__(self,
                 db_path,
                 sub_indices, # range
                 filter_length = 65,
                 max_f = 16000,
                 ):
        super().__init__()

        mat_path = f"{db_path}/measurements_array.mat"
        self.anthropomorphic_measurements = self.mat2data(mat_path).to(th.float32)

        self.data = []

        for sub_id in sub_indices:
            path = f"{db_path}/chedar_{sub_id+1:04}_UV2m.sofa"
            returns = self.sofa2data(path, max_f, filter_length)
            returns["AnthroFeatures"] = self.anthropomorphic_measurements[sub_id]
            self.data.append(returns)
            #ic(returns)

        ### Standardize every data_kind in data
        ### only use this on training data
        self.Val_Standardize = {}
        for data_kind in ["SrcPos", "SrcPos_Cart", "HRTF", "HRTF_mag", "HRIR"]:
            # Collect all data for the current data_kind
            all_data = [d[data_kind] for d in self.data]
            
            # Convert to a single tensor
            all_data_tensor = th.stack(all_data)
            # ic(all_data_tensor)

            # if data_kind == "AnthroFeatures":
                # ic(all_data_tensor)
                # ic(all_data_tensor.shape)
            
            # Compute mean and standard deviation
            mean = th.mean(all_data_tensor).to('cuda')
            std = th.std(all_data_tensor).to('cuda')

            print(f'[{data_kind}] mean:{mean:.4}, std:{std:.4}')
            
            # Store in Val_Standardize
            self.Val_Standardize[data_kind] = {
                "mean": mean,
                "std": std
            }
        
        # stacked_data = th.stack(self.anthropomorphic_measurements)
        mean = th.mean(self.anthropomorphic_measurements, dim=0).to('cuda')
        std = th.std(self.anthropomorphic_measurements, dim=0).to('cuda')
        self.Anthro_Standardize = {
            "mean": mean,
            "std": std
        }
        # ic(self.Anthro_Standardize)

    def __len__(self):
        '''
        :return: number of training subjects in dataset
        '''
        return len(self.data)

    def __getitem__(self, index):
        '''
        :return: dict consisting of
            SrcPos as          B x 3 tensor
            SrcPosCart as      B x 3 tensor
            HRTF as            B x 2 x L tensor
            HRTF_mag as        B x 2 x L tensor
            HRIR as            B x 2 x 2L tensor
            AnthroFeatures as  12 tensor
        '''
        return self.data[index]
     
    def mat2data(self, path):
        mat_contents = scipy.io.loadmat(path)
        tensor_data = th.tensor(mat_contents['measures_array'])
        # ic(mat_contents)
        # return tensor_data
        # ic(tensor_data.shape)
        selected_data = th.cat((tensor_data[:, :10], tensor_data[:, -2:]), dim=1) # First 10 and last 2 features only
        return selected_data

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
            'HRIR': HRIR_us,
        }
        
        return returns

if __name__ == "__main__":
    dataset = CHEDARDataset("../CHEDAR", range(0, 1003)) # 1253 subjects
