import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import torchaudio as ta   
import torch.nn.functional as F
import os
import scipy.io
from scipy.spatial import KDTree
import sys

def plotmaghrtf(srcpos,sig_gt,sig_pred,config,idx_plot_list=np.arange(5), figdir='', fname='', data_kind='HRTF_mag'):
    '''
    sig: (B,2,L)
    srcpos: (B,3)
    '''
    if data_kind == 'HRTF':
        mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude', top_db = 80)  
        hrtf_mag_gt = mag2db(th.abs(sig_gt))
        hrtf_mag_pred = mag2db(th.abs(sig_pred))
    elif data_kind == 'HRTF_mag':
        hrtf_mag_gt = sig_gt
        hrtf_mag_pred = sig_pred
    f_bin = np.linspace(0,config["max_frequency"],round(config["fft_length"]/2)+1)[1:]
    plt.figure(figsize=(12,round(6*len(idx_plot_list))))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    for itr, idx_plot in enumerate(idx_plot_list):
        plt.subplot(len(idx_plot_list),1, itr+1)

        plt.plot(f_bin, hrtf_mag_gt[idx_plot,0,:].to('cpu').detach().numpy().copy(), label="Left (Ground Truth)", color='b',linestyle=':')
        plt.plot(f_bin, hrtf_mag_gt[idx_plot,1,:].to('cpu').detach().numpy().copy(), label="Right (Ground Truth)", color='r',linestyle=':')
        plt.plot(f_bin, hrtf_mag_pred[idx_plot,0,:].to('cpu').detach().numpy().copy(), label="Left (Predicted)", color='b')
        plt.plot(f_bin, hrtf_mag_pred[idx_plot,1,:].to('cpu').detach().numpy().copy(), label="Right (Predicted)", color='r')
        plt.grid()
        plt.legend()
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")
        ylim = [-80,0] if config["green"] else [-50,30]
        plt.ylim(ylim)
        plt.xlim([0,config["max_frequency"]])
        srcpos_np = srcpos.to("cpu").detach().numpy().copy()
        # print(srcpos_np.shape)
        plt.title(f"HRTF (radius={srcpos_np[idx_plot,0]:.2f} m, azimuth={srcpos_np[idx_plot,1]/np.pi*180:.1f} deg, zenith={srcpos_np[idx_plot,2]/np.pi*180:.1f} deg)")

    if figdir == '':
        figdir = config["artifacts_dir"] + "/figure/HRTF/"
    if fname == '':
        raise NotImplementedError
    os.makedirs(figdir, exist_ok=True)
    plt.savefig(f"{fname}.png", dpi=300)
    # plt.savefig(figure_dir + "HRTF_Mag_"+mode+".jpg", dpi=300)
    plt.close()

def plot_mag_Angle_vs_Freq(HRTF_mag, SrcPos, vmin=-40, vmax=+30, fs=16000, figdir='', fname='', flg_save=True):
    '''
    HRTF_mag: B,2,L
    SrcPos: B,3
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
        for lr in [0,1]:
            fig, axs = plt.subplots(1, 2, gridspec_kw={
                                    'width_ratios': [1,1],
                                    'height_ratios': [1]}, figsize=(16,4))
            for s, sign in enumerate([+1,-1]):
                idx = idx_all[((th.abs(SrcPos_Cart[:,axis_zero]) < 1e-3) * ((SrcPos_Cart[:,axis_split]+1e-3)*sign >= 0))]
                idx = idx[th.argsort(SrcPos_Cart[idx, axis_sort], descending=True)]
                axs[s].imshow(HRTF_mag[idx.to(th.int).tolist(), lr, :], vmin=vmin, vmax=vmax, aspect='auto')
                ytiks_list = th.linspace(0,len(idx)-1,4)
                axs[s].set_yticks(ytiks_list)
                axs[s].set_yticklabels([f'{SrcPos[i, axis_ticks]:.3}' for i in idx[ytiks_list.to(int)]])
                axs[s].set_ylabel(['Radius (m)', 'Azimuth (rad)', 'Zenith (rad)'][axis_ticks])

                xtiks_list = [0,31,63,95,127]
                axs[s].set_xticks(xtiks_list)
                axs[s].set_xticklabels([f'{(xtiks+1)*fs/1e3/HRTF_mag.shape[-1]:}' for xtiks in xtiks_list])
                axs[s].set_xlabel('Frequency (kHz)')

                axs[s].set_title(f'{plane} plane, {titles[s]} ({["x","y","z"][axis_zero]}'+r'$\simeq$'+f'0, {["x","y","z"][axis_split]}'+[r"$\geq$",r"$\leq$"][s]+'0)')
            
            if flg_save:
                if fname == '' or figdir == '':
                    raise ValueError
                os.makedirs(figdir, exist_ok=True)
                # print(f'{fname}_{["left","right"][lr]}_{plane}.png')
                fig.savefig(f'{fname}_{["left","right"][lr]}_{plane}.png', dpi=300)
                plt.close()

def posTF2IR(tf):
    # in/out: (B,2,L,S) tensor
    tf = tf.squeeze()
    # print(tf.dim())
    if tf.dim() == 3:
        zeros = th.zeros([tf.shape[0],tf.shape[1],1]).to(tf.device).to(tf.dtype)
        tf_fc = th.conj(th.flip(tf[:,:,:-1],dims=(-1,)))
    elif tf.dim() == 4:
        tf = tf.permute(3,0,1,2)
        zeros = th.zeros([tf.shape[0],tf.shape[1],tf.shape[2],1]).to(tf.device).to(tf.dtype)
        tf_fc = th.conj(th.flip(tf[:,:,:,:-1],dims=(-1,)))

    # zeros = th.zeros([tf.shape[0],tf.shape[1],1]).to(tf.device).to(tf.dtype)
    # tf_fc = th.conj(th.flip(tf[:,:,:-1],dims=(-1,)))
    tf = th.cat((zeros,tf,tf_fc), dim=-1)
    ir = th.fft.ifft(th.conj(tf), dim=-1)

    # print(f'mean(abs(imag)):{th.mean(th.abs(th.imag(ir)))}')
    # print(f'mean(abs(real)):{th.mean(th.abs(th.real(ir)))}')
    ir = th.real(ir)

    if tf.dim() == 4:
        ir = ir.permute(1,2,3,0)
    return ir

def posTF2IR_dim4(tf):
    # in/out: (S,B,2,L) tensor
    zeros = th.zeros([tf.shape[0],tf.shape[1],tf.shape[2],1]).to(tf.device).to(tf.dtype)
    tf_fc = th.conj(th.flip(tf[:,:,:,:-1],dims=(-1,)))

    tf = th.cat((zeros,tf,tf_fc), dim=-1)
    ir = th.fft.ifft(th.conj(tf), dim=-1)

    ir = th.real(ir)

    return ir

def Data_ITD_2_HRIR(data, itd=None, data_kind='HRIR', config={}):
    '''
    args:
        data: B,2,Lor2L,S
        itd:  B,S
    return:
        hrir: B,2,2L,S
    '''
    if itd == None:
        assert data_kind == 'HRTF'
        return posTF2IR(data)
    else:
        data = data.permute(-1,0,1,2) # S,B,2,Lor2L
        itd  = itd.permute(-1,0)      # S,B
        if data_kind == 'HRIR':
            hrir_ori = data
        elif data_kind == 'HRTF_mag':
            HRTF_mag_linearscale = 10**(data/20)
            _, hrir_ori = minphase_recon(HRTF_mag_linearscale.to(th.complex64), contain_neg_fbin=False)
        elif data_kind == 'HRTF':
            _, hrir_ori = minphase_recon(data, contain_neg_fbin=False)
        itd_ori = hrir2itd(hrir=hrir_ori, fs=config["max_frequency"]*2, f_us=config["fs_upsampling"]).to(data.device)
        hrir_out = assign_itd(hrir_ori=hrir_ori, itd_ori=itd_ori, itd_des=itd, fs=config["max_frequency"]*2)
        # 2022/12/19 fs がいずれも config["max_frequency"] となっていたのを修正

        return hrir_out.permute(1,2,3,0)

def Data_2_HRTF_mag(data, data_kind='HRTF', config={}):
    '''
    args:
        data: B,2,Lor2L,S
    return:
        hrir: B,2,2L,S
    '''
    mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude', top_db = 80) 
    if data_kind == 'HRIR':
        HRTF_mag = mag2db(th.abs(th.conj(th.fft.fft(data, dim=-2))))[:,:,1:config["fft_length"]//2+1,:]
    elif data_kind == 'HRTF_mag':
        HRTF_mag = data
    elif data_kind == 'HRTF':
        HRTF_mag = mag2db(th.abs(data))

    return HRTF_mag

def plothrir(srcpos,hrtf_gt,hrtf_pred,config,idx_plot_list=np.arange(5), hrir_gt=None, hrir_pred=None,figdir='',fname=''):
    t_bin = np.linspace(0, config["fft_length"]/(2*config["max_frequency"]), config["fft_length"])
    if hrir_gt==None:
        hrir_gt = posTF2IR(hrtf_gt)
    if hrir_pred==None:
        hrir_pred =  posTF2IR(hrtf_pred)
    plt.figure(figsize=(12,round(6*len(idx_plot_list))))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    for itr, idx_plot in enumerate(idx_plot_list):
        plt.subplot(len(idx_plot_list),1, itr+1)

        plt.plot(t_bin, hrir_gt[idx_plot,0,:].to('cpu').detach().numpy().copy(), label="Left (Ground Truth)", color='b',linestyle=':')
        plt.plot(t_bin, hrir_gt[idx_plot,1,:].to('cpu').detach().numpy().copy(), label="Right (Ground Truth)", color='r',linestyle=':')
        plt.plot(t_bin, hrir_pred[idx_plot,0,:].to('cpu').detach().numpy().copy(), label="Left (Predicted)", color='b')
        plt.plot(t_bin, hrir_pred[idx_plot,1,:].to('cpu').detach().numpy().copy(), label="Right (Predicted)", color='r')
        plt.grid()
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Magnitude")
        # ylim = [-80,0] if config["green"] else [-50,30]
        # plt.ylim(ylim)
        plt.xlim([t_bin[0], t_bin[-1]])
        srcpos_np = srcpos.to("cpu").detach().numpy().copy()
        # print(srcpos_np.shape)
        plt.title(f"HRIR (radius={srcpos_np[idx_plot,0]:.2f} m, azimuth={srcpos_np[idx_plot,1]/np.pi*180:.1f} deg, zenith={srcpos_np[idx_plot,2]/np.pi*180:.1f} deg)")
        # sys.exit()

    if figdir == '':
        figdir = config["artifacts_dir"] + "/figure/HRIR/"
    if fname == '':
        raise NotImplementedError
    os.makedirs(figdir, exist_ok=True)
    plt.savefig(f"{fname}.png", dpi=300)
    # plt.savefig(figure_dir + "HRTF_Mag_"+mode+".jpg", dpi=300)
    plt.close()

    return hrir_gt, hrir_pred

class Net(nn.Module):

    def __init__(self, model_name="network", use_cuda=True):
        super().__init__()
        self.use_cuda = use_cuda
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
            self.cuda()

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
            self.cuda()
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

def sph_harm(m, n, phi, theta):
    """Spherical harmonic function
    m, n: degrees and orders
    phi (in [0, 2*pi]): azimuth angle
    theta (in [0, pi]): zenith angle
    """
    return special.sph_harm(m, n, phi, theta)

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
    for nn in np.arange(1, order+1):
        nn_vec = np.tile([nn], 2*nn+1)
        n = np.append(n, nn_vec)
        mm = np.arange(-nn, nn+1)
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

def plotSphHankel(kind=2,min_order=0,max_order=2,max_x = 20):
    eps = 1e-12
    x = np.linspace(eps,max_x,1000)
    fig = plt.figure(figsize=(7,7))
    kind_str = "1st" if kind == 1 else "2nd"
    fig.suptitle('Spherical Henkel Function of ' + kind_str + ' kind', fontsize=16)
    ax_r = fig.add_subplot(2,1,1)
    ax_i = fig.add_subplot(2,1,2)
    for order in np.arange(min_order, max_order+1):
        y = spherical_hn(n=order, k=kind, z=x)
        re = y.real
        im = y.imag
        ax_r.plot(x,re,label=f"n={order}")
        ax_i.plot(x,im,label=f"n={order}")
    ax_r.grid()
    ax_i.grid()
    ax_r.set_xlim([0,max_x])
    ax_i.set_xlim([0,max_x])
    ax_i.set_ylim([-2+(kind-1),1+(kind-1)])
    ax_r.legend()
    ax_i.legend()
    ax_r.set_xlabel("x")
    ax_i.set_xlabel("x")
    ax_r.set_ylabel(f"Re(h_n^({kind})(x))")
    ax_i.set_ylabel(f"Im(h_n^({kind})(x))")
     # plt.show()
    fig.savefig("figure/SphHankel_" + kind_str + ".png", dpi=300)
    fig.savefig("figure/SphHankel_" + kind_str + ".jpg", dpi=300)

def calcMaxOrder(e=np.e, f=16e3, c=343, R = 0.45, maxto = 35):
    k = 2*np.pi*f/c
    ul = np.array([maxto])
    # print(k.shape)
    # print(ul.shape)
    arr = np.vstack([np.ceil(e*k*R/2), np.tile(ul, k.shape[0])])
    # print(arr)
    return np.min(arr, axis=0)

def calcMaxOrder_th(e=np.e, f=16e3, c=343, R = 0.45, maxto = 35):
    k = 2*np.pi*f/c
    ul = th.tensor([maxto])
    # print(k.shape)
    # print(ul.shape)
    # print((e*k*R/2).dtype) # torch.float32
    # print(ul)
    arr = th.cat((th.ceil(e*k*R/2), th.tile(ul, (k.shape[0],))), dim=0)
    # print(arr)
    return th.min(arr, dim=0)[0]

class SpecialFunc():
    def __init__(self, r=th.tensor([1.0]), phi=th.tensor([0.0]), theta=th.tensor([0.0]), maxto=15, fft_length=128, max_f=16000, c=343.18):
        self.max_f = max_f
        self.c = c
        self.r = r #.to('cpu').detach().numpy().copy()
        # print(self.r.shape)
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
        # print(wave_num_for_Hankel.shape) # (filter_length,)
        # print(wave_num_for_Hankel[0])
        B = self.r.shape[0]
        n_tile = th.tile(th.arange(self.maxto+1), (B,1)) 
        # print(n_tile)
        # print(n_tile.shape) # B, maxto+1
        # print(self.r.device) # cpu
        # print(wave_num_for_Hankel.device) # cpu
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
        B = self.r.shape[0]
        # n_tile = th.tile(th.arange(self.maxto+1), (B,1)) 
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

class SpecialFunc_np():
    def __init__(self, r=th.tensor([1.0]), phi=th.tensor([0.0]), theta=th.tensor([0.0]), maxto=15, fft_length=128, max_f=16000, c=343.18):
        self.max_f = max_f
        self.c = c
        self.r = r.to('cpu').detach().numpy().copy()
        # print(self.r.shape)
        self.phi = np.array(phi.to('cpu').detach().numpy().copy())
        self.theta = np.array(theta.to('cpu').detach().numpy().copy())
        self.maxto = maxto

        self.filter_length = round(fft_length/2)# + 1)
        self.f_arr = (np.linspace(0, self.max_f, self.filter_length+1))[1:] # frequency bin, (filterlengh,)
        self.N_arr = calcMaxOrder(f=self.f_arr,maxto=self.maxto)    # truncation order, (filterlengh,)
        self.n_vec, self.m_vec = sph_harm_nmvec(order = self.maxto) # ((N+1)**2,)
        
    def SphHankel(self):
        wave_num_for_Hankel = 2*np.pi*self.f_arr/self.c
        if wave_num_for_Hankel[0] == 0:
            wave_num_for_Hankel[0] = 1
        # print(wave_num_for_Hankel.shape) # (filter_length,)
        # print(wave_num_for_Hankel[0])
        B = self.r.shape[0]
        n_tile = np.tile(np.arange(self.maxto+1), (B,1)) 
        # print(n_tile)
        # print(n_tile.shape) # B, maxto+1
        z_tile = self.r[:,None]*np.tile(wave_num_for_Hankel, (B,1))
        # print(z_tile.shape) # B, filterlength
        out = spherical_hn(n=n_tile[:,:,None], k=1, z=z_tile[:,None,:])
        # out = spherical_hn(n=n_tile[:,:,None], k=2, z=z_tile[:,None,:])
        out = np.nan_to_num(out, copy=False)
        # print(z_tile)
        # print(np.min(z_tile))
        # print(out.shape) # (B, maxto, filter_length)
        # out[:,:,0] = 0
        return th.from_numpy(out.astype(np.complex64)).clone()

    def SphHarm(self):
        out = sph_harm(n=self.n_vec[None,:], m=self.m_vec[None,:], phi=self.phi[:,None], theta=self.theta[:,None])
        # print(out.shape) # B, (maxto+1)**2
        return th.from_numpy(out.astype(np.complex64)).clone()
    
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

def transformCoeff(out_nn, n_vec, N_arr, maxto, filter_length, scale=750):
    '''
    args:
        out_nn:  (B, round(2*2*np.sum((self.N_arr + 1)**2)), ) tensor
    return:
        coeff: (B, 2, (maxto+1)**2, filter_length) tensor
    '''
    B = out_nn.shape[0]
    # print(B)
    coeff = th.zeros(B, 2, (maxto+1)**2, filter_length, dtype=th.complex64).cuda()
    idx_now = 0
    for i,n in enumerate(n_vec):
        idx = (np.where(N_arr>=n))[0][0]
        L = filter_length - idx
        pad = (idx, 0) 
        re_l = F.pad(out_nn[:, idx_now    :idx_now+  L],pad)
        # print(re_l.shape) # torch.Size([B,filter_length])
        im_l = F.pad(out_nn[:, idx_now+  L:idx_now+2*L],pad)
        coeff[:,0,i,:] = th.complex(real=re_l, imag=im_l)
        re_r = F.pad(out_nn[:, idx_now+2*L:idx_now+3*L],pad)
        im_r = F.pad(out_nn[:, idx_now+3*L:idx_now+4*L],pad)
        # print(re_r.shape) # B,filterlength
        coeff[:, 1,i,:] = th.complex(real=re_r, imag=im_r)
        idx_now = idx_now+4*L
        # if i == 1000:
        #     print(n)
        #     print(f"idx:{idx}")
        #     print(f"coeff:{coeff[0,0,i,:50]}")

    # print(f"[coeff-Re]  max:{th.max(th.real(coeff)):.6} min:{th.min(th.real(coeff)):.6} mean:{th.mean(th.real(coeff)):.6} std:{th.std(th.real(coeff)):.6}")
    # print(f"[coeff-Im]  max:{th.max(th.imag(coeff)):.6} min:{th.min(th.imag(coeff)):.6} mean:{th.mean(th.imag(coeff)):.6} std:{th.std(th.imag(coeff)):.6}")
    coeff = coeff * scale
    # print(f"[coeff-Re]  max:{th.max(th.real(coeff)):.6} min:{th.min(th.real(coeff)):.6} mean:{th.mean(th.real(coeff)):.6} std:{th.std(th.real(coeff)):.6}")
    # print(f"[coeff-Im]  max:{th.max(th.imag(coeff)):.6} min:{th.min(th.imag(coeff)):.6} mean:{th.mean(th.imag(coeff)):.6} std:{th.std(th.imag(coeff)):.6}")
        
    # outputs/coeff_16000_128_19_1e-07_nogreen.pt
    # [coeff-Re]  max:672.808 min:-712.046 mean:-0.00391887 std:32.5892
    # [coeff-Im]  max:659.001 min:-693.004 mean:0.000903253 std:32.5832
    # [coeff-Mag] max:2.85988 min:-1.24411 mean:0.992069 std:0.843775
    # ---------------
    # outputs/coeff_16000_128_10_1e-07_nogreen.pt
    # [coeff-Re]  max:673.081 min:-712.559 mean:-0.0120666 std:49.2469
    # [coeff-Im]  max:658.833 min:-693.056 mean:0.00678079 std:49.2331
    # [coeff-Mag] max:2.86009 min:-1.24401 mean:1.34253 std:0.762657
    # ---------------
    # outputs/coeff_16000_128_5_1e-07_nogreen.pt
    # [coeff-Re]  max:677.336 min:-716.717 mean:-0.0568869 std:59.3732
    # [coeff-Im]  max:661.087 min:-697.757 mean:0.017926 std:59.3479
    # [coeff-Mag] max:2.86269 min:-1.25045 mean:1.49076 std:0.650746
    # ---------------
    return coeff

def transformCoeff_MagdBPhase(out_nn, n_vec, N_arr, maxto, filter_length, c_std, c_mean, c_dim):
    '''
    args:
        out_nn:  (B, round(2*3*np.sum((self.N_arr + 1)**2)), ) tensor
    return:
        coeff: (B, 2, (maxto+1)**2, filter_length) tensor
    '''
    B = out_nn.shape[0]
    coeff = th.zeros(B, 2, (maxto+1)**2, filter_length, dtype=th.complex64).cuda()
    idx_now = 0
    for i,n in enumerate(n_vec):
        idx = (np.where(N_arr>=n))[0][0]
        L = filter_length - idx
        pad = (idx, 0) 
        magdb_l    = F.pad(out_nn[:, idx_now    :idx_now+  L],pad)
        magdb_r    = F.pad(out_nn[:, idx_now+3*L:idx_now+4*L],pad)
        
        if c_dim == 3:
            magdb_l    = magdb_l * c_std[0,i,:] + c_mean[0,i,:] 
            magdb_r    = magdb_r * c_std[1,i,:] + c_mean[1,i,:]  
        elif c_dim == [1,3]:
            magdb_l    = magdb_l * c_std[0, :] + c_mean[0, :] 
            magdb_r    = magdb_r * c_std[1, :] + c_mean[1, :] 
        elif c_dim == [2,3]:
            magdb_l    = magdb_l * c_std[0, i, None] + c_mean[0, i, None] 
            magdb_r    = magdb_r * c_std[1, i, None] + c_mean[1, i, None]
        elif c_dim == [1,2,3]:
            magdb_l    = magdb_l * c_std[0, None] + c_mean[0, None] 
            magdb_r    = magdb_r * c_std[1, None] + c_mean[1, None]

        phase_re_l = F.pad(out_nn[:, idx_now+  L:idx_now+2*L],pad)
        phase_im_l = F.pad(out_nn[:, idx_now+2*L:idx_now+3*L],pad)
   
        phase_re_r = F.pad(out_nn[:, idx_now+4*L:idx_now+5*L],pad)
        phase_im_r = F.pad(out_nn[:, idx_now+5*L:idx_now+6*L],pad)

        coeff[:,0,i,:] = 10 ** magdb_l * th.exp(1j * th.atan2(phase_im_l, phase_re_l))
        coeff[:,1,i,:] = 10 ** magdb_r * th.exp(1j * th.atan2(phase_im_r, phase_re_r))
        idx_now = idx_now+6*L

    return coeff

def transformCoeff_MagdB(out_nn, n_vec, N_arr, maxto, filter_length, c_std, c_mean):
    '''
    args:
        out_nn:  (B, round(2*np.sum((self.N_arr + 1)**2)), ) tensor
        * phase: random
    return:
        coeff: (B, 2, (maxto+1)**2, filter_length) tensor
    '''
    B = out_nn.shape[0]
    coeff = th.zeros(B, 2, (maxto+1)**2, filter_length, dtype=th.complex64).cuda()
    idx_now = 0
    th.manual_seed(0)
    for i,n in enumerate(n_vec):
        idx = (np.where(N_arr>=n))[0][0]
        L = filter_length - idx
        pad = (idx, 0) 
        phase_rand = th.rand(4*L)
        magdb_l    = F.pad(out_nn[:, idx_now    :idx_now+  L],pad)
        # magdb_l    = magdb_l * c_std[0,i,:] + c_mean[0,i,:] # cdim= 3
        magdb_l    = magdb_l * c_std[0, :] + c_mean[0, :] # cdim=[1,3]
        # magdb_l    = magdb_l * c_std[0, i, None] + c_mean[0, i, None] # cdim=[2,3]
        phase_re_l = F.pad(phase_rand[ :  L],pad)
        phase_im_l = F.pad(phase_rand[L:2*L],pad)

        magdb_r    = F.pad(out_nn[:, idx_now+L:idx_now+2*L],pad)
        # magdb_r    = magdb_r * c_std[1,i,:] + c_mean[1,i,:] # cdim= 3
        magdb_r    = magdb_r * c_std[1, :] + c_mean[1, :] # cdim=[1,3]
        # magdb_r    = magdb_r * c_std[1, i, None] + c_mean[1, i, None] # cdim=[2,3]
        phase_re_r = F.pad(phase_rand[2*L:3*L],pad)
        phase_im_r = F.pad(phase_rand[3*L:4*L],pad)

        coeff[:,0,i,:] = 10 ** magdb_l * th.exp(1j * th.atan2(phase_im_l, phase_re_l))
        coeff[:,1,i,:] = 10 ** magdb_r * th.exp(1j * th.atan2(phase_im_r, phase_re_r))
        idx_now = idx_now+2*L

    return coeff

def transformCoeff_MagdBPhase_CNN(out_nn, maxto, filter_length, c_std, c_mean, c_dim, scale=1):
    '''
    args:
        out_nn: (sub, 6, (maxto+1)**2, filter_length) float tensor
    return:
        coeff: (sub, 2, (maxto+1)**2, filter_length) complex tensor
    '''
    S = out_nn.shape[0]
    coeff = th.zeros(S, 2, (maxto+1)**2, filter_length, dtype=th.complex64).to(out_nn.device)
    magdb  = out_nn[:,0:2,:,:] * scale # S,ch,(m,n),freq
    # print(f'mag_db.mean:{th.mean(magdb)},magdb.std:{th.std(magdb)},magdb.max:{th.max(magdb)},magdb.min:{th.min(magdb)}')
    # # sys.exit()
    # mag_db.mean:0.584204912185669,magdb.std:1.3623549938201904,magdb.max:15.947799682617188,magdb.min:-13.195794105529785
    # dim: [0:ch, 1:(m,n), 2:freq, 3:sub]
    if c_dim == 3: # ch,(m,n),freq
        magdb = magdb * c_std[None,:,:,:] + c_mean[None,:,:,:]  
    elif c_dim == [1,3]: # ch,freq
        magdb = magdb * c_std[None,:,None,:] + c_mean[None,:,None,:] 
    elif c_dim == [2,3]: # ch,(m,n)
        magdb = magdb * c_std[None,:,:,None] + c_mean[None,:,:,None]
    elif c_dim == [1,2,3]: # ch
        magdb = magdb * c_std[None,:,None,None] + c_mean[None,:,None,None]
    # magdb  = out_nn[:,0:2,:,:] # S,ch,(m,n),freq
    # print(f'mag_db.mean:{th.mean(magdb)},magdb.std:{th.std(magdb)},magdb.max:{th.max(magdb)},magdb.min:{th.min(magdb)}')
    # sys.exit()
    # c_dim=3, size=1dim: mag_db.mean:-0.6627160310745239,magdb.std:2.4896860122680664,magdb.max:3.808256149291992,magdb.min:-10.0
    # import sys
    # sys.exit()
    phase_re = out_nn[:,2:4,:,:]
    phase_im = out_nn[:,4:6,:,:]
    coeff = 10 ** magdb * th.exp(1j * th.atan2(phase_im, phase_re))
    # eps = 1e-10*th.ones(magdb.shape).to(magdb.device)
    # coeff = th.max(10 ** magdb, eps) * th.exp(1j * th.atan2(phase_im, phase_re))
    # print(f'max(abs(coeff)):{th.max(th.abs(coeff))}')
    if th.any(th.isnan(coeff)) or th.any(th.isinf(coeff)):
        if th.any(th.isnan(coeff)):
            print('nan is detected in Coeff.')
        if th.any(th.isinf(coeff)):
            print('inf is detected in Coeff.')
        coeff_re = th.nan_to_num(th.real(coeff),nan=0.0,posinf=0.0,neginf=0.0)
        coeff_im = th.nan_to_num(th.imag(coeff),nan=0.0,posinf=0.0,neginf=0.0)
        coeff = th.complex(coeff_re, coeff_im)
    return coeff

def aprox_t_des(pts, t, plot=False, db_name='HUTUBS', print_num_pts=False):
    t_grid_dic = scipy.io.loadmat(f't_des/grid_t{t}d{(t+1)**2}.mat') 
    t_grid = th.tensor(t_grid_dic["Y"]).T
    # print(t_grid.shape) # [(t+1)**2, 3]
    if db_name in ['RIEC', 'Own']:
        row = t_grid[:,2] >= -0.5
        t_grid = t_grid[row]
        if print_num_pts:
            print(f"[aprox_t_des] remove pts s.t. z<-0.5. {(t+1)**2}->{t_grid.shape[0]} pts")

    # pts = vec_cart_gt / 1.47

    kdt = KDTree(pts.cpu())
    dist, idx = kdt.query(t_grid) # dist, index
    idx = sorted(idx)
    idx_prev = idx.copy()
    idx = list(set(idx)) # 重複を除く
    # print(len(idx_prev))
    # print(len(idx))
    if len(idx_prev) > len(idx) and print_num_pts:
        print(f"[aprox_t_des] detected duplication. {len(idx_prev)}->{len(idx)} pts")
        
    # print(f"[t={t}] mean(dist):{np.mean(dist):.3}")
    # print(f"[t={t}] mean(dist^2):{np.mean(dist**2):.3}")
    # print(idx)
    
    #=========================
    if plot:
        t_grid_aprox = pts[idx]
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(t_grid[:,0], t_grid[:,1], t_grid[:,2],label='t-design',marker='+')
        ax.scatter(t_grid_aprox[:,0], t_grid_aprox[:,1], t_grid_aprox[:,2], label='nearest',marker='x')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f'Spherical t-design with d=(t+1)^2 points (t={t})') 
        ax.legend(loc=2, title='legend')

        # Make data
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot the surface of sphere
        ax.plot_surface(x, y, z,color="gray",rcount=100, ccount=100, antialiased=False, alpha=0.05)
        ax.set_box_aspect((1,1,1))
    #=========================
    return idx

def aprox_reg_poly(pts, num_pts, db_name='HUTUBS', print_num_pts=False):
    t_grid_dic = scipy.io.loadmat(f't_des/grid_rp{num_pts}.mat') 
    t_grid = th.tensor(t_grid_dic["Y"]).T
    # print(t_grid.shape) # [(t+1)**2, 3]
    if db_name in ['RIEC', 'Own']:
        row = t_grid[:,2] >= -0.5
        t_grid = t_grid[row]
        if print_num_pts:
            print(f"[aprox_reg_poly] remove pts s.t. z<-0.5. {num_pts}->{t_grid.shape[0]} pts")

    kdt = KDTree(pts.cpu())
    dist, idx = kdt.query(t_grid) # dist, index
    idx = sorted(idx)
    idx_prev = idx.copy()
    idx = list(set(idx)) # 重複を除く
    if len(idx_prev) > len(idx) and print_num_pts:
        print(f"[aprox_reg_poly] detected duplication. {len(idx_prev)}->{len(idx)} pts")

    return idx


def nearest_pts(pts, query):
    kdt = KDTree(pts.cpu())
    _, idx = kdt.query(query.cpu())
    return idx

def plane_sample(pts, axes, thr=0.01, assertion=False):
    # pts: Bx3 tensor
    # axis: list of {0,1,2} e.g. [0,1]
    # idx = th.zeros(0)
    idx = th.zeros(0, device="cpu") # 20240520 Koyama
    num_pts = len(idx)
    # idx_all = th.arange(0, pts.shape[0])
    idx_all = th.arange(0, pts.shape[0], device="cpu") # 20240517 Koyama
    for ax in axes:
        # idx = th.cat((idx, idx_all[(th.abs(pts[:,ax]) < thr)]), dim=0)
        idx = th.cat((idx, idx_all[(th.abs(pts[:,ax]) < thr).cpu()]), dim=0) # 20240521 Koyama
        if ax==0:
            num_pts += 20
        elif ax==1:
            num_pts += 34
        else:
            num_pts += 36
    idx = idx.to(th.int).tolist()
    idx = list(set(idx)) # 重複を除く
    idx = sorted(idx)
    # print(idx)
    if len(axes)==2:
        num_pts -= 2
    elif len(axes)==3:
        num_pts -= 6
    if assertion:
        assert len(idx) == num_pts

    return idx

def parallel_planes_sample(pts, values, axis=2, thr=0.01):
    '''
        pts: Bx3 tensor
        values: list of v e.g. [-0.7, 0.0, 0.7]

        sample from plane pts[axis]==v for v in values
    '''
    # idx = th.zeros(0)
    idx = th.zeros(0, device="cpu") # 20240520 Koyama
    # num_pts = len(idx)
    # idx_all = th.arange(0, pts.shape[0])
    idx_all = th.arange(0, pts.shape[0], device="cpu") # 20240520 Koyama
    for v in values:
        # idx = th.cat((idx, idx_all[(th.abs(pts[:,axis]-v) < thr)]), dim=0)
        idx = th.cat((idx, idx_all[(th.abs(pts[:,axis]-v) < thr).cpu()]), dim=0) # 20240521 Koyama
    idx = idx.to(th.int).tolist()
    idx = list(set(idx)) # 重複を除く
    idx = sorted(idx)

    return idx

def plotCoeff(ptpath_lininv,ptpath_pred,maxto,sub,clim = [-10,10],flag_save=False,fname=None):
    # print("Load Coeff. from " + ptpath_lininv)
    coeff_lininv = th.load(ptpath_lininv)

    # print("Load Coeff. from " + ptpath_pred)
    coeff_pred = th.load(ptpath_pred)

    plt.figure(figsize=(20,8/(19**2)*(maxto+1)**2))
    fig_num = 5
    fig_itr = 1
    cmap = 'bwr'#'RdBu'
    n,m = sph_harm_nmvec(maxto)
    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
    plt.imshow(np.hstack((n.reshape([-1,1]),m.reshape([-1,1]))),cmap='BrBG',interpolation='none',aspect=0.1)
    plt.title(f"[n,m]")
    plt.colorbar()
    # plt.subplot(1,6,2)
    # plt.imshow(m.reshape([-1,1]),cmap=cmap,aspect=0.1)

    # sub_1, left, real
    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
     # print(coeff_lininv.shape)
    plt.imshow(th.real(coeff_lininv[0,:,:,sub]),cmap=cmap,interpolation='none') # ch, (N+1)**2, L, sub
    plt.clim(clim)
    plt.xlim()
    plt.title(f"Left, Real, Sub={sub+1}, lininv")

    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
    # print(coeff_pred.shape)
    plt.imshow(th.real(coeff_pred[0,:,:,sub]),cmap=cmap,interpolation='none') # ch, (N+1)**2, L, sub
    plt.colorbar()
    plt.clim(clim)
    plt.title(f"Left, Real, Sub={sub+1}, pred")

    # sub_1, left, real
    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
     # print(coeff_lininv.shape)
    plt.imshow(th.imag(coeff_lininv[0,:,:,sub]),cmap=cmap,interpolation='none') # ch, (N+1)**2, L, sub
    # plt.colorbar()
    plt.clim(clim)
    plt.title(f"Left, Imag, Sub={sub+1}, lininv")

    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
     # print(coeff_pred.shape)
    plt.imshow(th.imag(coeff_pred[0,:,:,sub]),cmap=cmap,interpolation='none') # ch, (N+1)**2, L, sub
    plt.colorbar()
    plt.clim(clim)
    plt.title(f"Left, Imag, Sub={sub+1}, pred")

    if flag_save:
        plt.savefig(fname)

     # plt.show()


    # plt.plot(f_bin, mag2db(th.abs(hrtf_gt[idx_plot,0,:])).to('cpu').detach().numpy().copy(), label="Left (Ground Truth)", color='b',linestyle=':')
    # plt.plot(f_bin, mag2db(th.abs(hrtf_gt[idx_plot,1,:])).to('cpu').detach().numpy().copy(), label="Right (Ground Truth)", color='r',linestyle=':')
    # plt.plot(f_bin, mag2db(th.abs(hrtf_pred[idx_plot,0,:])).to('cpu').detach().numpy().copy(), label="Left (Predicted)", color='b')
    # plt.plot(f_bin, mag2db(th.abs(hrtf_pred[idx_plot,1,:])).to('cpu').detach().numpy().copy(), label="Right (Predicted)", color='r')
    plt.clf()
    plt.close()

def plotCoeff_MagPhase(ptpath_lininv,ptpath_pred,maxto,sub,cmap = 'Reds',clim = [0,10],flag_save=False,fname=None):
    # print("Load Coeff. from " + ptpath_lininv)
    coeff_lininv = th.load(ptpath_lininv)

    # print("Load Coeff. from " + ptpath_pred)
    coeff_pred = th.load(ptpath_pred)

    # f_bin = np.linspace(0,16000,128+1)[1:]
    # idx_plot_list=np.array([202,211,220,229]) # measured

    plt.figure(figsize=(20,8/(19**2)*(maxto+1)**2))
    fig_num = 5
    fig_itr = 1
    n,m = sph_harm_nmvec(maxto)
    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
    plt.imshow(np.hstack((n.reshape([-1,1]),m.reshape([-1,1]))),cmap='BrBG',interpolation='none',aspect=0.1)
    plt.title(f"[n,m]")
    plt.colorbar()
    # plt.subplot(1,6,2)
    # plt.imshow(m.reshape([-1,1]),cmap=cmap,aspect=0.1)

    # sub_1, left, mag
    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
     # print(coeff_lininv.shape)
    plt.imshow(th.abs(coeff_lininv[0,:,:,sub]),cmap=cmap,interpolation='none') # ch, (N+1)**2, L, sub
    plt.clim(clim)
    plt.xlim()
    plt.title(f"Left, Magnitude, Sub={sub+1}, lininv")

    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
     # print(coeff_pred.shape)
    plt.imshow(th.abs(coeff_pred[0,:,:,sub]),cmap=cmap,interpolation='none') # ch, (N+1)**2, L, sub
    plt.colorbar()
    plt.clim(clim)
    plt.title(f"Left, Magnitude, Sub={sub+1}, pred")

    # sub_1, left, phase
    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
     # print(coeff_lininv.shape)
    plt.imshow(th.angle(coeff_lininv[0,:,:,sub]),cmap='hsv',interpolation='none') # ch, (N+1)**2, L, sub
    # plt.colorbar()
    plt.clim([0,2*np.pi])
    plt.title(f"Left, Phase, Sub={sub+1}, lininv")

    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
     # print(coeff_pred.shape)
    plt.imshow(th.angle(coeff_pred[0,:,:,sub]),cmap='hsv',interpolation='none') # ch, (N+1)**2, L, sub
    plt.colorbar()
    plt.clim([0,2*np.pi])
    plt.title(f"Left, Phase, Sub={sub+1}, pred")

    if flag_save:
        plt.savefig(fname)

     # plt.show()
    plt.clf()
    plt.close()

def plotCoeff_MagdBPhase(ptpath_lininv,ptpath_pred,maxto,sub,cmap = 'Reds',clim = [0,10],flag_save=False,fname=None):
    # print("Load Coeff. from " + ptpath_lininv)
    coeff_lininv = th.load(ptpath_lininv)

    # print("Load Coeff. from " + ptpath_pred)
    coeff_pred = th.load(ptpath_pred)

    plt.figure(figsize=(20,8/(19**2)*(maxto+1)**2))
    fig_num = 5
    fig_itr = 1
    n,m = sph_harm_nmvec(maxto)
    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
    plt.imshow(np.hstack((n.reshape([-1,1]),m.reshape([-1,1]))),cmap='BrBG',interpolation='none',aspect=0.1)
    plt.title(f"[n,m]")
    plt.colorbar()
    # plt.subplot(1,6,2)
    # plt.imshow(m.reshape([-1,1]),cmap=cmap,aspect=0.1)

    # sub_1, left, mag
    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
     # print(coeff_lininv.shape)
    plt.imshow(20*th.log10(th.abs(coeff_lininv[0,:,:,sub])),cmap=cmap,interpolation='none') # ch, (N+1)**2, L, sub
    plt.clim(clim)
    plt.xlim()
    plt.title(f"Left, Magnitude (dB), Sub={sub+1}, lininv")

    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
     # print(coeff_pred.shape)
    plt.imshow(20*th.log10(th.abs(coeff_pred[0,:,:,sub])),cmap=cmap,interpolation='none') # ch, (N+1)**2, L, sub
    plt.colorbar()
    plt.clim(clim)
    plt.title(f"Left, Magnitude (dB), Sub={sub+1}, pred")

    # sub_1, left, phase
    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
     # print(coeff_lininv.shape)
    plt.imshow(th.angle(coeff_lininv[0,:,:,sub]),cmap='hsv',interpolation='none') # ch, (N+1)**2, L, sub
    # plt.colorbar()
    plt.clim([0,2*np.pi])
    plt.title(f"Left, Phase, Sub={sub+1}, lininv")

    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
     # print(coeff_pred.shape)
    plt.imshow(th.angle(coeff_pred[0,:,:,sub]),cmap='hsv',interpolation='none') # ch, (N+1)**2, L, sub
    plt.colorbar()
    plt.clim([0,2*np.pi])
    plt.title(f"Left, Phase, Sub={sub+1}, pred")

    if flag_save:
        plt.savefig(fname)

     # plt.show()
    plt.clf()
    plt.close()

def plotCoeff_VarMag(ptpath_lininv,ptpath_pred,maxto,cmap = 'Reds',clim = [0,10],flag_save=False,fname=None,unit='db'):
    # print("Load Coeff. from " + ptpath_lininv)
    coeff_lininv = th.load(ptpath_lininv)

    # print("Load Coeff. from " + ptpath_pred)
    coeff_pred = th.load(ptpath_pred)

    fig_num = 3
    fig_itr = 1

    plt.figure(figsize=(5*fig_num,8/(19**2)*(maxto+1)**2))
    
    n,m = sph_harm_nmvec(maxto)
    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
    plt.imshow(np.hstack((n.reshape([-1,1]),m.reshape([-1,1]))),cmap='BrBG',interpolation='none',aspect=0.1)
    plt.title(f"[n,m]")
    plt.colorbar()
    # plt.subplot(1,6,2)
    # plt.imshow(m.reshape([-1,1]),cmap=cmap,aspect=0.1)

    # sub_1, left, mag
    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
     # print(coeff_lininv.shape)
    var = th.var(th.abs(coeff_lininv[0,:,:,:]),dim=-1,unbiased=True)
    if unit == 'db':
        var = 20*th.log10(var)
    plt.imshow(var,cmap=cmap,interpolation='none') # ch, (N+1)**2, L, sub
    plt.clim(clim)
    plt.xlim()
    unit_str = ' (dB)' if unit == 'db' else ''
    plt.title(f"Left, Variance of Magnitude{unit_str}, lininv")

    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
     # print(coeff_pred.shape)
    var = th.var(th.abs(coeff_pred[0,:,:,:]),dim=-1,unbiased=True)
    if unit == 'db':
        var = 20*th.log10(var)
    plt.imshow(var,cmap=cmap,interpolation='none') # ch, (N+1)**2, L, sub
    plt.colorbar()
    plt.clim(clim)
    plt.title(f"Left, Variance of Magnitude{unit_str}, pred")

    # # sub_1, left, phase
    # plt.subplot(1,fig_num,fig_itr)
    # fig_itr += 1
    #  # print(coeff_lininv.shape)
    # plt.imshow(th.angle(coeff_lininv[0,:,:,sub]),cmap='hsv',interpolation='none') # ch, (N+1)**2, L, sub
    # # plt.colorbar()
    # plt.clim([0,2*np.pi])
    # plt.title(f"Left, Phase, Sub={sub+1}, lininv")

    # plt.subplot(1,fig_num,fig_itr)
    # fig_itr += 1
    #  # print(coeff_pred.shape)
    # plt.imshow(th.angle(coeff_pred[0,:,:,sub]),cmap='hsv',interpolation='none') # ch, (N+1)**2, L, sub
    # plt.colorbar()
    # plt.clim([0,2*np.pi])
    # plt.title(f"Left, Phase, Sub={sub+1}, pred")

    if flag_save:
        plt.savefig(fname)

    # plt.show()
    plt.clf()
    plt.close()

def plotCoeff_all(dirname,ptpath_lininv,ptpath_pred,config,maxto=18,fname_suffix='.jpg'):
    fname = dirname + 'figure/coeff' + fname_suffix

    clim = [-10,10] if config["green"] else [-100,100]
    plotCoeff(ptpath_lininv,ptpath_pred,maxto=maxto,sub=0,clim=clim,flag_save=True,fname=fname)
    fname = dirname + 'figure/coeff_MagPhase' + fname_suffix
    clim = [0,20] if config["green"] else [0,100]
    plotCoeff_MagPhase(ptpath_lininv,ptpath_pred,maxto=maxto,sub=0,clim=clim,flag_save=True,fname=fname)
    fname = dirname + 'figure/coeff_MagdBPhase' + fname_suffix
    clim = [0,40] if config["green"] else [0,80]
    plotCoeff_MagdBPhase(ptpath_lininv,ptpath_pred,maxto=maxto,sub=0,clim=clim,flag_save=True,fname=fname)
    fname = dirname + 'figure/coeff_VarMag' + fname_suffix
    clim = [0,10] if config["green"] else [0,80]
    plotCoeff_VarMag(ptpath_lininv,ptpath_pred,maxto=maxto, clim=clim,flag_save=True,fname=fname,unit='')
    fname = dirname + 'figure/coeff_VarMagdB' + fname_suffix
    clim = [0,40] if config["green"] else [0,80]
    plotCoeff_VarMag(ptpath_lininv,ptpath_pred,maxto=maxto, clim=clim,flag_save=True,fname=fname,unit='db')

def plotz(ptpath_pred,sub,clim = [-10,10],flag_save=False,fname=None):
    # print("Load z from " + ptpath_pred)
    z_pred = th.load(ptpath_pred)
    # print(z_pred.shape)

    plt.figure(figsize=(30,10))
    fig_num = 3
    fig_itr = 1
    cmap = 'bwr'#'RdBu'
    aspect = z_pred.shape[1] / 100.0

    # sub_1, 
    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
    plt.imshow(z_pred[:,:,sub],cmap=cmap,interpolation='none',aspect=aspect) # B,dim_z,sub
    plt.colorbar()
    plt.clim(clim)
    plt.xlim()
    plt.title(f"z, Sub={sub+1}")

    # sub_2, 
    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
    plt.imshow(z_pred[:,:,sub+1],cmap=cmap,interpolation='none',aspect=aspect) # B,dim_z,sub
    plt.colorbar()
    plt.clim(clim)
    plt.xlim()
    plt.title(f"z, Sub={sub+2}")

    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
     # print(coeff_pred.shape)
    plt.imshow(th.var(z_pred,dim=-1),cmap='Reds',interpolation='none',aspect=aspect) # B,dim_z,sub
    plt.colorbar()
    plt.clim([0,clim[1]/10])
    plt.title(f"Variance of z (dim=-1:sub)")

    if flag_save:
        plt.savefig(fname)

     # plt.show()
    plt.clf()
    plt.close()

def plotcz(dir,config):
    dirname = f'{dir}/'
    debug = ''
    # reg_w = 1e-6
    green = '' if config["green"] else '_nogreen'
    reg_w = config["reg_w"]
    suffix = f'_16000_128_{config["max_truncation_order"]}_{reg_w:.0e}'
    ptpath_lininv= 'outputs/' + 'coeff' + debug + suffix + green + '.pt'
    ptpath_pred  = dirname + 'coeff_pred' + debug +  suffix + '.pt'
    plotCoeff_all(dirname,ptpath_lininv,ptpath_pred,config,config["max_truncation_order"],fname_suffix='.jpg')

    if config["model"] == "CNN":
        ptpath_lininv= dirname + 'coeff_in' + debug +  suffix + green + '.pt'
        ptpath_pred  = ptpath_lininv # dammy
        plotCoeff_all(dirname,ptpath_lininv,ptpath_pred,round(config['num_pts']**0.5-1),fname_suffix='_in.jpg',config=config)


    ptpath_z = dirname + 'z_train' + debug + suffix + '.pt'
    fname = dirname + 'figure/z_train.jpg'
    clim = [-0.5,0.5] if config["z_norm"] else [-3,3]
    plotz(ptpath_z,sub=0,clim=clim,flag_save=True,fname=fname)
    # print("===========================")

def plotcolonpos(pos,c,config,path=None,mode=None):
    fig = plt.figure(figsize=1.5*plt.figaspect(1))
    ax = fig.add_subplot(111,projection='3d')
    # print(pos.shape)
    # print(c.shape)
    s = ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=c, cmap='jet', vmin = 0.5/pos.shape[0], vmax =1.5/pos.shape[0])
    
    for d in ['x','y','z']:
        eval(f'ax.set_{d}label("{d}")')
    fig.colorbar(s)
    # ax.set_clim(0, 0.5)
    # plt.clim([0,0.5])
    elev_l = [-90,-20,0,20,90]
    azim_l = [-120,-90,-60,0,60,90,120,180] 

    dir = f'{path}/figure/weight'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    dir_all = f'{path}/figure/weight_all'
    if not os.path.isdir(dir_all):
        os.makedirs(dir_all)
    if config["num_pts"] < 440:
        t = round(config["num_pts"]**0.5-1)
        prefix = f't{t}'
    else:
        prefix = f'440pts'
    for elev in elev_l:
        for azim in azim_l:
            ax.view_init(elev=elev, azim=azim)
            fig.savefig(f'{dir_all}/{prefix}_e{elev}_a{azim}_{mode}.jpg')
            if elev == 20 and azim == -60:
                fig.savefig(f'{dir}/{prefix}_e{elev}_a{azim}_{mode}.jpg')
    plt.clf()
    plt.close()


def hrir2itd(hrir,fs,f_us=8*48000,thrsh_ms=1000,lpf=True,upsample_via_cpu=True,conv_cpu=True):
    '''
    args:
        hrir: (S,B,2,L') tensor
        fs: scaler, sampling freq. of original hrir
        f_us: scaler, sampling freq. after upsampling
        thrsh_ms: threshold [ms]. (computed ITD is forced to be in [-thrsh_ms, +thrsh_ms] )
        lpf: bool. If True, Low-pass filter is filtered to hrir.
    returns:
        ITD: (S,B) tensor, interaural time difference [s] (-:src@left, +:src@right)
    '''
    if lpf:
        hrir = ta.functional.lowpass_biquad(waveform=hrir, sample_rate=fs, cutoff_freq=1600)
    else:
        pass
    if upsample_via_cpu:
        hrir = hrir.cpu()
    upsampler = ta.transforms.Resample(fs, f_us)
    hrir_us = upsampler(hrir.to(th.float32).contiguous())
    if upsample_via_cpu and not conv_cpu:
        hrir_us = hrir_us.cuda()
    S, B, _, L = hrir_us.shape
    thrsh_idx = round(f_us/thrsh_ms)
    #===============================
    HRIR_l = hrir_us[:,:,0,:]
    HRIR_r = hrir_us[:,:,1,:]
    HRIR_l_pad = F.pad(HRIR_l,(L,L)) # torch.Size([77, 440, 384])
    HRIR_l_pad_in = HRIR_l_pad.reshape(1, S*B, -1)
    HRIR_r_wt = HRIR_r.reshape(S*B, 1, -1)
    crs_cor = F.conv1d(HRIR_l_pad_in, HRIR_r_wt, groups=S*B) ### CHEDAR CRASH
    crs_cor = crs_cor.reshape(S, B, -1)
    idx_beg = L - thrsh_idx
    idx_end = L + thrsh_idx + 1
    idx_max = th.argmax(crs_cor[:,:,idx_beg:idx_end], dim=-1) - thrsh_idx
    ITD = idx_max/f_us
    return ITD

def HilbertTransform(data, detach=False):
    # https://stackoverflow.com/questions/50902981/hilbert-transform-using-cuda
    assert data.dim()==4
    N = data.shape[-1]
    # Allocates memory on GPU with size/dimensions of signal
    if detach:
        transforms = data.clone().detach()
    else:
        transforms = data.clone()
    transforms = th.fft.fft(transforms, axis=-1)
    transforms[:,:,:,1:N//2]      *= -1j      # positive frequency
    transforms[:,:,:,(N+2)//2 + 1: N] *= +1j # negative frequency
    transforms[:,:,:,0] = 0; # DC signal
    if N % 2 == 0:
        transforms[:,:,:,N//2] = 0; # the (-1)**n term
    # Do IFFT on GPU: in place (same memory)
    return th.fft.ifft(transforms, axis=-1)

def minphase_recon(tf, contain_neg_fbin=False):
    '''
    args:
        tf: (S,B,2,L) or (S,B,2,2L) tensor (L: # of freq. bins)
        contain_neg_fbin: bool. If True, mag.shape == (S,B,2,2L)
        conj: bool. 
    return:
        phase_min: (S,B,2,2L) tensor
        ir_min:  (S,B,2,2L) tensor. Impulse response with minimum phase.
    '''
    if contain_neg_fbin:
        tf_pm = tf
    else:
        tf_nf = th.conj(th.flip(tf[:,:,:,:-1],dims=(-1,))) # negatibe freq.
        tf_pm = th.cat((th.ones_like(tf)[:,:,:,0:1], tf, tf_nf), dim=-1) # [1,pos,neg]
        # DC 成分が 0 でなく 1 なのは，対数を取ったときに-infに発散するのを防ぐため←これでOK?
    mag_pm_log = th.log(th.abs(tf_pm)) # magnitude の対数振幅（底: e）
    phase_min =  - HilbertTransform(mag_pm_log)
    ir_min = th.real(th.fft.ifft(th.abs(tf_pm)*th.exp(1j * phase_min), axis=-1))

    return phase_min, ir_min

def assign_itd(hrir_ori,itd_ori,itd_des,fs,shift_s=1e-3):
    '''
    args:
        hrir_ori: (S,B,2,L) tensor. (L: filter length)
        itd_ori: (S,B) tensor. ITD [s] of hrir_ori
        itd_des: (S,B) tensor. desired ITD [s]
        fs: scaler [Hz]. Sampling Frequency.
        shift_lr: scaler [s]. offset when ITD==0.
    return:
        ir_itd_des:  (S,B,2,L) tensor. Impulse response with desired ITD.
    '''
    S, B = itd_ori.shape
    L = hrir_ori.shape[-1]
    shift_idx = shift_s * fs
    ITD_idx_fs_half = (itd_des-itd_ori) * fs / 2
    offset = th.ones(S,B,2).to(ITD_idx_fs_half.device) * shift_idx
    offset[:,:,0] += ITD_idx_fs_half # left
    offset[:,:,1] -= ITD_idx_fs_half # right
    offset = th.round(offset).to(int)

    arange = th.arange(L).reshape(1,1,1,L).tile(S,B,2,1).to(ITD_idx_fs_half.device)
    arange = (arange - offset[:,:,:,None]) % L

    # square window to remove pre-echo
    window_length = int(L - shift_idx)
    window_sq = th.cat((th.ones(window_length), th.zeros(L-window_length))).to(hrir_ori.device)
    hrir_ori_w = hrir_ori * window_sq[None,None,None,:]
    ir_itd_des = th.gather(hrir_ori_w, -1, arange)

    return ir_itd_des

def vhlines(ax, linestyle='-', color='gray', zorder=1, alpha=0.8, lw=0.75):
    ax.axhline(y=np.pi/2, linestyle=linestyle, color=color,zorder=zorder, alpha=alpha, lw=lw)
    ax.axvline(x=np.pi/2, linestyle=linestyle, color=color,zorder=zorder, alpha=alpha, lw=lw)
    ax.axvline(x=np.pi, linestyle=linestyle, color=color,zorder=zorder, alpha=alpha, lw=lw)
    ax.axvline(x=np.pi*3/2, linestyle=linestyle, color=color,zorder=zorder, alpha=alpha, lw=lw)
    ax.text(np.pi/2, np.pi+0.05, "Left", ha='center')
    ax.text(np.pi*3/2, np.pi+0.05, "Right", ha='center')
    ax.text(np.pi, np.pi+0.05, "Back", ha='center')

def plotazimzeni(pos,c,fname,title,cblabel,cmap='gist_heat',figsize=(10.5,5),dpi=300, emphasize_mes_pos=False, idx_mes_pos=None, vmin=None, vmax=None, save=True, suffix='png', clf=True):
    '''
    args:
        pos: (B,*>3) tensor. (:,1):azimuth, (:,2):zenith
        c: (B) tensor.
        fname: str. filename
        title: str. title.
        cblabel: str. label of colorbar.
        cmap: colormap.
        figsie: (*,*) tuple.
        dpi: scalar.
    '''
    fig, ax = plt.subplots(figsize=figsize)
    vhlines(ax)
    if vmin==None:
        vmin=th.min(c)
    if vmax == None:
        vmax=th.max(c)
    mappable = ax.scatter(pos[:,1], pos[:,2], c=c, cmap=cmap, s=60, lw=0.3, ec="gray", zorder=2, vmin=vmin, vmax=vmax)
    fig.colorbar(mappable=mappable,label=cblabel)
    if emphasize_mes_pos:
        ax.scatter(pos[idx_mes_pos,1], pos[idx_mes_pos,2], s=120, lw=0.5, c="None", marker="o", ec="k", zorder=1)
    ds = 0.1
    xlim = [0-ds, 2*np.pi]
    ylim = [0-ds, ds+np.pi]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()
    # ax.set_xlabel('Azimuth (rad)')
    # ax.set_ylabel('Zenith (rad)')

    #== 2022/12/23 add ==
    ax.set_xlabel('Azimuth (deg)')
    ax.set_ylabel('Zenith (deg)')
    ax.set_xticks(np.linspace(0,2*np.pi,12+1))
    # ax.set_xticklabels(['0',r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$', r'$2\pi$'])
    ax.set_xticklabels([f'{int(azim)}' for azim in np.linspace(0,2*180,12+1)])
    ax.set_yticks(np.linspace(0,np.pi,6+1))
    # ax.set_yticklabels(['0',r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])
    ax.set_yticklabels([f'{int(azim)}' for azim in np.linspace(0,180,6+1)])
    #====================

    ax.set_title(title)
    if save:
        # fig.savefig(f'{fname}.jpg', dpi=dpi)
        fig.savefig(f'{fname}.{suffix}', dpi=dpi)

    if clf:
        fig.clf()
        plt.close()

def replace_activation_function(model, act_func_new, act_func_old=nn.ReLU):
    for child_name, child in model.named_children():
        if isinstance(child, act_func_old):
            setattr(model, child_name, act_func_new)
        else:
            replace_activation_function(child, act_func_new, act_func_old)