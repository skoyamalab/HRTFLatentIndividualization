import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os

def plot_mag_Angle_vs_Freq(HRTF_mag, SrcPos, vmin=-40, vmax=+30, fs=16000, figdir='', fname='', flg_save=True):
    '''
    HRTF_mag: B, 2, L
    SrcPos: B, 3
    '''
    HRTF_mag, SrcPos = HRTF_mag.cpu().detach(), SrcPos.cpu().detach()
    SrcPos_Cart = sph2cart(SrcPos[:,1], SrcPos[:,2], SrcPos[:,0])
    idx_all = th.arange(0, SrcPos_Cart.shape[0])
    for plane in ['Median','Horizontal']:
        if plane == 'Median':
            #== median plane (|y|<1e2), Zenith vs Freq ===
            axis_zero, axis_split, axis_sort, axis_ticks  = [1,0,2,2]
            titles = ['Front', 'Back']
        elif plane == 'Horizontal':
            #== horizontal plane (|z|<1e2), Azimuth vs Freq ===
            axis_zero, axis_split, axis_sort, axis_ticks  = [2,1,0,1]
            titles = ['Left', 'Right']
        else:
            raise ValueError
        for lr in [0]:
            fig, axs = plt.subplots(figsize=(6, 3))
            fig.tight_layout()
            fig.subplots_adjust(left=0.15, bottom=0.17)
            # print(axs)
            idx_split = [None, None]
            for s, sign in enumerate([-1, +1]):
                idx_split[s] = idx_all[((th.abs(SrcPos_Cart[:,axis_zero]) < 1e-3) * ((SrcPos_Cart[:,axis_split]+1e-3)*sign >= 0))]
                idx_split[s] = idx_split[s][th.argsort(SrcPos_Cart[idx_split[s], axis_sort], descending=(sign == -1))]
            
            idx = th.cat((idx_split[1], idx_split[0]), dim=0)

            axs.imshow(HRTF_mag[idx.to(th.int).tolist(), lr, :], vmin=vmin, vmax=vmax, aspect='auto')
           
            desired_degrees = [0, 45, 90, 135]
            ytiks_list = [i for i in range(len(idx)) if th.round(np.degrees(SrcPos[idx[i], axis_ticks])).to(th.int) in desired_degrees]
            # print(th.round(np.degrees(SrcPos[idx[:], axis_ticks])).to(th.int))
            axs.set_yticks(ytiks_list)
            # axs.set_yticklabels([f'{int(np.degrees(SrcPos[i, axis_ticks]))}' for i in idx[ytiks_list.to(int)]])
            axs.set_yticklabels([f'{th.round(np.degrees(SrcPos[idx[i], axis_ticks])).to(th.int)}' for i in ytiks_list])
            axs.set_ylim([0, len(idx) - 1])
            axs.set_ylabel(['Radius (m)', 'Azimuth (deg)', 'Zenith (deg)'][axis_ticks], fontsize=12, labelpad=15)
            axs.text(-13, 5, f'{titles[0]} ←', va='center', ha='right', fontsize=10, rotation=90)
            axs.text(-13, 43, f'→ {titles[1]}', va='center', ha='right', fontsize=10, rotation=90)

            # xtiks_list = [0,31,63,95,127]
            xtiks_list = th.linspace(0,HRTF_mag.shape[-1]-1,11)
            xtiks_labels = [f'{int(x)}' for x in np.linspace(0, fs//1e3, 11)]
            axs.set_xticks(xtiks_list)
            # axs[s].set_xticklabels([f'{(xtiks+1)*fs/1e3/HRTF_mag.shape[-1]:}' for xtiks in xtiks_list])
            axs.set_xticklabels(xtiks_labels)
            axs.set_xlim([0, HRTF_mag.shape[-1]-1])
            axs.set_xlabel('Frequency (kHz)', fontsize=12)

            # axs.set_title(f'{plane} plane, {titles[s]}')
            fig.colorbar(axs.images[0], ax=axs, label='Magnitude (dB)')

            if flg_save:
                if fname == '' or figdir == '':
                    raise ValueError
                os.makedirs(figdir, exist_ok=True)
                # print(f'{fname}_{["left","right"][lr]}_{plane}.png')
                fig.savefig(f'{fname}_{["left","right"][lr]}_{plane}.pdf', dpi=300)
                plt.close()

class Net(nn.Module):
    def __init__(self, model_name="network", use_cuda=True, device='cuda:0'):
        super().__init__()
        self.use_cuda = use_cuda
        self.device = device
        if th.cuda.is_available(): # 210710 added
            self.use_cuda = True      # 210710 added
        self.model_name = model_name

    def save(self, model_dir, suffix=''):
        '''
        save the network to model_dir/model_name.suffix.net
        :param model_dir: directory to save the model to
        :param suffix: suffix to append after model name
        '''
        if self.use_cuda:
            self.cpu()

        if suffix == "":
            fname = f"{model_dir}/{self.model_name}.net"
        else:
            fname = f"{model_dir}/{self.model_name}.{suffix}.net"

        th.save(self.state_dict(), fname)
        if self.use_cuda:
            self.to(self.device)

    def load_from_file(self, model_file):
        '''
        load network parameters from model_file
        :param model_file: file containing the model parameters
        '''
        if self.use_cuda:
            self.cpu()

        states = th.load(model_file)
        self.load_state_dict(states)

        if self.use_cuda:
            self.to(self.device)
        print(f"Loaded: {model_file}")

    def load(self, model_dir, suffix=''):
        '''
        load network parameters from model_dir/model_name.suffix.net
        :param model_dir: directory to load the model from
        :param suffix: suffix to append after model name
        '''
        if suffix == "":
            fname = f"{model_dir}/{self.model_name}.net"
        else:
            fname = f"{model_dir}/{self.model_name}.{suffix}.net"
        self.load_from_file(fname)

    def num_trainable_parameters(self):
        '''
        :return: the number of trainable parameters in the model
        '''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class NewbobAdam(th.optim.Adam):

    def __init__(self,
                 weights,
                 net,
                 artifacts_dir,
                 initial_learning_rate=0.001,
                 decay=0.5,
                 max_decay=0.01,
                 timestamp = ""
                 ):
        '''
        Newbob learning rate scheduler
        :param weights: weights to optimize
        :param net: the network, must be an instance of type src.utils.Net
        :param artifacts_dir: (str) directory to save/restore models to/from
        :param initial_learning_rate: (float) initial learning rate
        :param decay: (float) value to decrease learning rate by when loss doesn't improve further
        :param max_decay: (float) maximum decay of learning rate
        '''
        super().__init__(weights, lr=initial_learning_rate, eps=1e-8)
        self.last_epoch_loss = np.inf
        self.second_last_epoch_loss = np.inf
        self.total_decay = 1
        self.net = net
        self.decay = decay
        self.max_decay = max_decay
        self.artifacts_dir = artifacts_dir
        self.timestamp = timestamp
        # store initial state as backup
        if decay < 1.0:
            net.save(artifacts_dir, suffix="newbob"+self.timestamp)
    
    def update_lr_two(self, loss):
        '''
        update the learning rate based on the current loss value and historic loss values
        :param loss: the loss after the current iteration
        '''
        if self.last_epoch_loss > self.second_last_epoch_loss and loss > self.last_epoch_loss and self.decay < 1.0 and self.total_decay > self.max_decay:
            self.total_decay = self.total_decay * self.decay
            print(f"NewbobAdam: Decay learning rate (loss degraded from { self.second_last_epoch_loss} to {loss})."
                  f"Total decay: {self.total_decay}")
            # restore previous network state
            self.net.load(self.artifacts_dir, suffix="newbob"+self.timestamp)
            # decrease learning rate
            for param_group in self.param_groups:
                param_group['lr'] = param_group['lr'] * self.decay
        else:
            self.second_last_epoch_loss = self.last_epoch_loss
            self.last_epoch_loss = loss
        # save last snapshot to restore it in case of lr decrease
        if self.decay < 1.0 and self.total_decay > self.max_decay:
            self.net.save(self.artifacts_dir, suffix="newbob"+self.timestamp)

    def update_lr_one(self, loss):
        '''
        update the learning rate based on the current loss value and historic loss values
        :param loss: the loss after the current iteration
        '''
        if loss > self.last_epoch_loss and self.decay < 1.0 and self.total_decay > self.max_decay:
            self.total_decay = self.total_decay * self.decay
            print(f"NewbobAdam: Decay learning rate (loss degraded from {self.last_epoch_loss} to {loss})."
                  f"Total decay: {self.total_decay}")
            # restore previous network state
            # self.net.load(self.artifacts_dir, suffix="newbob"+self.timestamp)
            # decrease learning rate
            for param_group in self.param_groups:
                param_group['lr'] = param_group['lr'] * self.decay
        self.last_epoch_loss = loss
        # else:
        #     self.last_epoch_loss = loss
        # save last snapshot to restore it in case of lr decrease
        # if self.decay < 1.0 and self.total_decay > self.max_decay:
        #     self.net.save(self.artifacts_dir, suffix="newbob"+self.timestamp)
    
    def update_lr_step(self, gamma=0.1):
        '''
        update the learning rate based on the current loss value and historic loss values
        :param loss: the loss after the current iteration
        '''
        for param_group in self.param_groups:
            param_group['lr'] = param_group['lr'] * gamma
        self.total_decay = self.total_decay * gamma
        print(f"NewbobAdam: Decay learning rate."
              f"Total decay: {self.total_decay}")

def cart2sph(x, y, z):
    """Conversion from Cartesian to spherical coordinates

    Parameters
    ------
    x, y, z : Position in Cartesian coordinates

    Returns
    ------
    totch.tensor contains
    phi, theta, r: Azimuth angle, zenith angle, distance
    phi in [0,2pi), theta in [0,pi)
    """
    r_xy = th.sqrt(x**2 + y**2)
    phi = th.atan2(y, x)
    theta = th.atan2(r_xy, z)
    r = th.sqrt(x**2 + y**2 + z**2)
    phi = phi%(2*np.pi) # [0,2pi)
    theta = theta%(np.pi) # [0,pi)
    # print(r.shape) # (*,)
    out = th.hstack((r.unsqueeze(1), phi.unsqueeze(1), theta.unsqueeze(1)))
    # print(out.shape) # (*,3)
    return out

def sph2cart(phi, theta, r):
    """Conversion from spherical to Cartesian coordinates

    Parameters
    ------
    phi, theta, r: Azimuth angle, zenith angle, distance

    Returns
    ------
    x, y, z : Position in Cartesian coordinates
    """
    x = r * th.sin(theta) * th.cos(phi)
    y = r * th.sin(theta) * th.sin(phi)
    z = r * th.cos(theta)
    return th.hstack((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)))

def sph_harm_th(m, n, phi, theta):
    """Spherical harmonic function
    m, n: degrees and orders
    phi (in [0, 2*pi]): azimuth angle
    theta (in [0, pi]): zenith angle
    """
    return special.sph_harm(m, n, phi.detach().cpu(), theta.detach().cpu())

def sph_harm_nmvec(order, rep=None):
    """Vectors of spherical harmonic orders and degrees
    Returns (order+1)**2 size vectors of n and m
    n = [0, 1, 1, 1, ..., order, ..., order]^T
    m = [0, -1, 0, 1, ..., -order, ..., order]^T

    Parameters
    ------
    order: Maximum order
    rep: Same vectors are copied as [n, .., n] and [m, ..., m]

    Returns
    ------
    n, m: Vectors of orders and degrees
    """
    n = np.array([0])
    m = np.array([0])
    for nn_ in np.arange(1, order+1):
        nn_vec = np.tile([nn_], 2*nn_+1)
        n = np.append(n, nn_vec)
        mm = np.arange(-nn_, nn_+1)
        m = np.append(m, mm)
    if rep is not None:
        n = np.tile(n[:, None], (1, rep))
        m = np.tile(m[:, None], (1, rep))
    return n, m

def spherical_hn(n, k, z):
    """nth-order sphericah Henkel function of kth kind
    Returns h_n^(k)(z)
    """
    if k == 1:
        return special.spherical_jn(n, z) + 1j * special.spherical_yn(n, z)
    elif k == 2:
        return special.spherical_jn(n, z) - 1j * special.spherical_yn(n, z)
    else:
        raise ValueError()

def calcMaxOrder_th(e=np.e, f=16e3, c=343, R = 0.45, maxto = 35):
    k = 2*np.pi*f/c
    ul = th.tensor([maxto])
    arr = th.cat((th.ceil(e*k*R/2), th.tile(ul, (k.shape[0],))), dim=0)
    return th.min(arr, dim=0)[0]

class SpecialFunc():
    def __init__(self, r=th.tensor([1.0]), phi=th.tensor([0.0]), theta=th.tensor([0.0]), maxto=15, fft_length=128, max_f=16000, c=343.18):
        self.max_f = max_f
        self.c = c
        self.r = r #.to('cpu').detach().numpy().copy()

        self.phi = phi #np.array(phi.to('cpu').detach().numpy().copy())
        self.theta = theta #np.array(theta.to('cpu').detach().numpy().copy())
        self.maxto = maxto

        self.filter_length = round(fft_length/2)# + 1)
        self.f_arr = (th.linspace(0, self.max_f, self.filter_length+1))[1:] # frequency bin, (filterlengh,)
        self.N_arr = calcMaxOrder_th(f=self.f_arr,maxto=self.maxto)    # truncation order, (filterlengh,)
        self.n_vec, self.m_vec = sph_harm_nmvec(order = self.maxto) # ((N+1)**2,)
        
    def SphHankel(self):
        wave_num_for_Hankel = 2*np.pi*self.f_arr/self.c
        if wave_num_for_Hankel[0] == 0:
            wave_num_for_Hankel[0] = 1
        
        B = self.r.shape[0]
        n_tile = th.tile(th.arange(self.maxto+1), (B,1)) 
        
        z_tile = self.r[:,None].cpu()*th.tile(wave_num_for_Hankel, (B,1))
        # print(z_tile.shape) # B, filterlength
        out = spherical_hn(n=n_tile[:,:,None], k=1, z=z_tile[:,None,:])
        # out = spherical_hn(n=n_tile[:,:,None], k=2, z=z_tile[:,None,:])
        out = np.nan_to_num(out, copy=False)
        # print(z_tile)
        # print(np.min(z_tile))
        # print(out.shape) # (B, maxto, filter_length)
        # out[:,:,0] = 0
        return th.from_numpy(out.astype(np.complex64)).clone()
    
    def SphBessel(self, s=th.tensor([0.2])): # see Zhang+10
        wave_num_for_Hankel = 2*np.pi*self.f_arr/self.c
        if wave_num_for_Hankel[0] == 0:
            wave_num_for_Hankel[0] = 1
        n_tile = self.n_vec
        z_tile = s.cpu() * wave_num_for_Hankel
        out = spherical_hn(n=n_tile[:,None], k=1, z=z_tile[None,:])
        out = np.real(out) # spherical Bessel
        out = np.nan_to_num(out, copy=False)
        return th.from_numpy(out.astype(np.complex64)).clone()

    def SphHarm(self):
        out = sph_harm_th(n=self.n_vec[None,:], m=self.m_vec[None,:], phi=self.phi[:,None], theta=self.theta[:,None])
        # print(out.shape) # B, (maxto+1)**2
        return out #th.from_numpy(out.astype(np.complex64)).clone()
    
    def RealSphHarm(self):
        SH = sph_harm_th(n=self.n_vec[None,:], m=self.m_vec[None,:], phi=self.phi[:,None], theta=self.theta[:,None])
        m_vec_th = th.from_numpy(self.m_vec).to(SH.device)
        out = th.real(SH) * (m_vec_th>=0)[None,:] - th.imag(SH)* (m_vec_th<0)[None,:]
        # print(out.shape) # B, (maxto+1)**2
        return out #th.from_numpy(out.astype(np.complex64)).clone()
    
    def RandomInput(self):
        num_elements = round(2*2*np.sum((self.N_arr + 1)**2))
        out_nn = th.rand(num_elements)
        coeff = th.zeros(2, (self.maxto+1)**2, self.filter_length)
        idx_now = 0
        for i,n in enumerate(self.n_vec):
            idx = (np.where(self.N_arr>=n))[0][0]
            L = self.filter_length - idx
            pad = (idx, 0) 
            re_l = F.pad(out_nn[idx_now    :idx_now+  L],pad)
            # print(re_l.shape) # torch.Size([filter_length])
            im_l = F.pad(out_nn[idx_now+  L:idx_now+2*L],pad)
            coeff[0,i,:] = th.complex(real=re_l, imag=im_l)
            re_r = F.pad(out_nn[idx_now+2*L:idx_now+3*L],pad)
            im_r = F.pad(out_nn[idx_now+3*L:idx_now+4*L],pad)
            coeff[1,i,:] = th.complex(real=re_r, imag=im_r)
            idx_now = idx_now+4*L
        return coeff



def replace_activation_function(model, act_func_new, act_func_old=nn.ReLU):
    for child_name, child in model.named_children():
        if isinstance(child, act_func_old):
            setattr(model, child_name, act_func_new)
        else:
            replace_activation_function(child, act_func_new, act_func_old)