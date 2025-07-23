import torch as th
import torch.nn.functional as F
import torchaudio as ta
from torch.utils.data import Dataset
import sofa
import numpy as np
import scipy.io
import pandas as pd

from utils import sph2cart

sub_indices = {
    'CIPIC': {
        # 'train': range(0, 35), # 35 - 8 = 27
        # 'valid': range(35, 39), # 4
        # 'test':  range(39, 45), # 6 - 2 = 4

        # 'train': range(0, 34), # 35 - 8 = 26
        # 'valid': range(34, 38), # 4

        'train': range(0, 38), # 38 - 8 = 30
        'test':  range(38, 45), # 7 - 2 = 5
        'all':   range(0, 45), # 45 subjects, but only 35 have full anthropometric features
        'skip': [1, 2, 4, 5, 6, 7, 9, 11, 41, 44] # 1-2, 4-7, 9, 11, 41, 44 have missing anthropometric features
    },
    'HUTUBS': {
        # 'train': range(0, 87), # 77 - 1 = 76
        # 'valid': range(77, 87), # 10 - 1 = 9

        'train': range(0, 87), # 87 - 2 = 85
        'test':  range(88, 95), # 7 - 1 = 6, remove duplication: 95==0, 87==21
        'all':   range(0, 96), # 96 subjects, but only 91 have anthropometric features and are not duplicates
        'skip': [17, 78, 87, 91, 95] # 17, 78, 91 have missing anthropometric features, 95==0, 87==21
    },
}

class MultiDataset(Dataset):
    def __init__(self,
                 db_names,
                 phase,
                 filter_length = 128,
                 max_f = 20000,
                 anthro_dim = 23,
                 ):
        super().__init__()
        self.db_names = db_names
        self.anthro_dim = anthro_dim

        self.data = []
        for db_name in self.db_names:
            all_anthro = self.anthro2data(db_name)

            # Getting HRTF magnitude data from CIPIC
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

                self.data.append(returns)

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
                               d and t are different for left and right ear (WITH CHEDAR)
            AnthroFeatures as  2 (LR) x 23 tensor (x[1-9, 12, 14, 16-17] d1-8, t1-2)
                               d and t are different for left and right ear (WITHOUT CHEDAR)
        '''
        return self.data[index]

    # Standardize per dataset
    def getValStandardize(self, device, indices=None):
        if indices is None:
            indices = range(len(self.data))
        Val_Standardize = {}

        for db_name in self.db_names:
            Val_Standardize[db_name] = {}

            # Collect all data for the current db_name
            db_data = [self.data[i] for i in indices if self.data[i]["db_name"] == db_name]

            for data_kind in ["SrcPos", "SrcPos_Cart", "HRTF", "HRTF_mag", "HRIR"]:
                # Collect all data for the current data_kind in current db_name
                stacked_data = th.stack([d[data_kind] for d in db_data])

                # This is to standardize across the S dimension
                # In the future, try also stacking each ear
                Val_Standardize[db_name][data_kind] = {
                    "mean": th.mean(stacked_data, dim=0).to(device),
                    "std": th.std(stacked_data, dim=0).to(device)
                }
        return Val_Standardize

    # Standardize across all datasets
    def getAnthroStandardize(self, device, indices=None):
        if indices is None:
            indices = range(len(self.data))
        stacked_data = th.stack([self.data[i]["AnthroFeatures"] for i in indices]).view(-1, self.anthro_dim)
        return {
            "mean": th.mean(stacked_data, dim=0).to(device),
            "std": th.std(stacked_data, dim=0).to(device)
        }
    
    def anthro2data(self, db_name):
        l_data, r_data = None, None
        if db_name == "CHEDAR":
            mat_contents = scipy.io.loadmat("../CHEDAR/measurements_array.mat")
            tensor_data = th.tensor(mat_contents['measures_array'])

            x_data = th.cat((tensor_data[:, 10].unsqueeze(1), tensor_data[:, 17].unsqueeze(1), tensor_data[:, 20].unsqueeze(1), tensor_data[:, -7:-2], tensor_data[:, 13].unsqueeze(1)), dim=1)
            d_data = th.cat((tensor_data[:, 0].unsqueeze(1), tensor_data[:, 2:8]), dim=1)
            t_data = th.deg2rad(tensor_data[:, 8:10])

            l_data = th.cat((x_data, d_data, t_data), dim=1)
            r_data = th.cat((x_data, d_data, t_data), dim=1)
        elif db_name == "HUTUBS":
            xlsx_contents = pd.read_excel("../HUTUBS/Antrhopometric measures/AntrhopometricMeasures.xlsx")

            # With CHEDAR, only x1-8, x12, d1-7, t1-2
            # x_data = th.cat((th.tensor(xlsx_contents.iloc[:, 1:9].values), th.tensor(xlsx_contents.iloc[:, 10].values).unsqueeze(1)), dim=1)
            # L_d_data = th.tensor(xlsx_contents.iloc[:, 14:21].values)
            # R_d_data = th.tensor(xlsx_contents.iloc[:, 26:33].values)
            # L_t_data = th.deg2rad(th.tensor(xlsx_contents.iloc[:, 24:26].values))
            # R_t_data = th.deg2rad(th.tensor(xlsx_contents.iloc[:, -2:].values))

            # Without CHEDAR, only x[1-9, 12, 14, 16-17], d1-8, t1-2
            x_data = th.tensor(xlsx_contents.iloc[:, 1:14].values)
            L_d_data = th.tensor(xlsx_contents.iloc[:, 14:22].values)
            R_d_data = th.tensor(xlsx_contents.iloc[:, 26:34].values)
            L_t_data = th.deg2rad(th.tensor(xlsx_contents.iloc[:, 24:26].values))
            R_t_data = th.deg2rad(th.tensor(xlsx_contents.iloc[:, -2:].values))

            l_data = th.cat((x_data, L_d_data, L_t_data), dim=1)
            r_data = th.cat((x_data, R_d_data, R_t_data), dim=1)
        elif db_name == "CIPIC":
            mat_contents = scipy.io.loadmat("../CIPIC/anthropometry/anthro.mat")

            x_data = th.tensor(mat_contents['X'])
            d_data = th.tensor(mat_contents['D'])
            t_data = th.tensor(mat_contents['theta'])

            # With CHEDAR, only x1-8, x12, d1-7, t1-2
            # l_data = th.cat((x_data[:, [*range(8), 11]], d_data[:, :7], t_data[:, :2]), dim=1)
            # r_data = th.cat((x_data[:, [*range(8), 11]], d_data[:, 8:15],t_data[:, -2:]), dim=1)

            # Without CHEDAR, only x[1-9, 12, 14, 16-17], d1-8, t1-2
            l_data = th.cat((x_data[:, [*range(9), 11, 13, 15, 16]], d_data[:, :8], t_data[:, :2]), dim=1)
            r_data = th.cat((x_data[:, [*range(9), 11, 13, 15, 16]], d_data[:, 8:16],t_data[:, -2:]), dim=1)
        
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

if __name__ == "__main__":
    train_dataset = MultiDataset(['CIPIC', 'HUTUBS'], phase='train')

    # Val_Standardize = train_dataset.Val_Standardize
    # Anthro_Standardize = train_dataset.Anthro_Standardize
    # ic(Val_Standardize, Anthro_Standardize)
