import numpy as np
from numpy import complex64
import torch as th
import torch.nn.functional as F
import torchaudio as ta
import torch.autograd as autograd         # computation graph
from icecream import ic
ic.configureOutput(includeContext=True)

class Loss(th.nn.Module):
    def __init__(self, mask_beginning=0):
        '''f
        base class for losses that operate on the wave signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__()
        self.mask_beginning = mask_beginning

    def forward(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        # data = data[..., self.mask_beginning:]
        # target = target[..., self.mask_beginning:]
        return self._loss(data, target)

    def _loss(self, data, target):
        pass

class MAE(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: MSE
        '''
        return th.mean(th.abs(data - target))


class MSE(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: MSE
        '''
        return th.mean(th.abs(data - target).pow(2))

class NMSE(Loss):
    def _loss(self, data, target, eps=1e-10):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: NMSE
        '''
        top = th.abs(data - target).pow(2)
        bottom = th.max(th.abs(target).pow(2), eps*th.ones_like(th.abs(target)))
        return th.mean(top/bottom)

class NMSEdB(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        pow2db = ta.transforms.AmplitudeToDB(stype = 'power')
        return th.mean(pow2db((th.abs(data - target).pow(2))/(th.abs(target).pow(2))))

class HuberLoss(Loss):
    def _loss(self, data, target, delta=1.0):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        dist = th.abs(data-target)
        h = dist**2/2 * (dist<=delta) + delta * (dist-delta/2) * (dist>delta)
        return th.mean(h)

class RMS(th.nn.Module):
    def __init__(self):
        '''
        base class for losses that operate on the wave signal
        '''
        super().__init__()
    def forward(self, z):
        return self._loss(z)
    def _loss(self, z):
        loss = th.mean(th.abs(z)**2)**0.5
        return loss

class VarLoss(th.nn.Module):
    def __init__(self, dim=0):
        '''
        base class for losses that operate on the wave signal
        '''
        super().__init__()
        self.dim=dim
    def forward(self, z):
        return self._loss(z)
    def _loss(self, z):
        '''
        :param z: predicted wave signals in a B x dim_z tensor
        :return: a scalar loss value
        '''
        if z.shape[0]==1:
            loss = th.mean(z) # dammy
        else:
            var = th.var(z,dim=self.dim,unbiased=True)
            # print(var.shape) # torch.Size([2, 128, 441]) # [2S or S, L, dimz]
            loss = th.mean(var)
        return loss

class CosDistIntra(th.nn.Module):
    def __init__(self):
        '''
        base class for losses that operate on the wave signal
        '''
        super().__init__()
    def forward(self, z):
        return self._loss(z)
    def _loss(self, z):
        '''
        :param z: (B,dimz,S)
        :return: a scalar loss value
        '''
        # print(z.shape) # torch.Size([361, 128, 32]) 
        c = th.mean(z,dim=0) # centroid
        # print(c.shape) # torch.Size([128, 32])
        cs = F.cosine_similarity(z, c[None,:,:], dim=1)
        # print(cs.shape) # torch.Size([361, 32])
        # print([th.max(cs),th.min(cs)]) # [tensor(0.6838, device='cuda:0', grad_fn=<MaxBackward1>), tensor(0.2459, device='cuda:0', grad_fn=<MinBackward1>)]

        return th.mean((1-cs).pow(2))**0.5

class CosDistIntraSquared(th.nn.Module):
    def __init__(self):
        '''
        base class for losses that operate on the wave signal
        '''
        super().__init__()
    def forward(self, z):
        return self._loss(z)
    def _loss(self, z):
        '''
        :param z: (B,dimz,S)
        :return: a scalar loss value
        '''
        # print(z.shape) # torch.Size([361, 128, 32]) 
        c = th.mean(z,dim=0) # centroid
        # print(c.shape) # torch.Size([128, 32])
        cs = F.cosine_similarity(z, c[None,:,:], dim=1)
        # print(cs.shape) # torch.Size([361, 32])
        # print([th.max(cs),th.min(cs)]) # [tensor(0.6838, device='cuda:0', grad_fn=<MaxBackward1>), tensor(0.2459, device='cuda:0', grad_fn=<MinBackward1>)]

        return th.mean((1-cs).pow(2))

class LSDdiffLoss_old(th.nn.Module):
    def __init__(self):
        '''
        base class for losses that operate on the wave signal
        '''
        super().__init__()
    def forward(self, z):
        return self._loss(z)
    def _loss(self, z):
        '''
        :param z: predicted wave signals in a B x dim_z tensor
        :return: a scalar loss value
        '''
        mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude')
        z_db = mag2db(th.abs(z)) # (440, 2, 128, 77)
        # print(z_db.shape)
        kernel = th.tensor([1,-1],dtype=th.float32)
        kernel = th.tile(kernel, (1,z_db.shape[-1],1)).to(z_db.device)
        diff_l = F.conv1d(z_db[:,0,:,:].permute(0,2,1), kernel)
        diff_r = F.conv1d(z_db[:,1,:,:].permute(0,2,1), kernel)
        return th.mean(diff_l**2 + diff_r**2)

class LSDdiffLoss(th.nn.Module):
    def __init__(self):
        '''
        base class for losses that operate on the wave signal
        '''
        super().__init__()
    def forward(self, z):
        return self._loss(z)
    def _loss(self, z):
        '''
        :param z: predicted wave signals in a B x dim_z tensor
        :return: a scalar loss value
        '''
        mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude')
        z = z.contiguous().permute(0,1,3,2) # 440,2,77,128
        z_db = mag2db(th.abs(z)) 
        z_db = z_db.contiguous().view(-1,1,z_db.shape[-1]) # 440*154, 128
        kernel = th.tensor([1,-1],dtype=th.float32)
        kernel = th.tile(kernel, (1,1,1)).to(z_db.device)
        diff = F.conv1d(z_db, kernel, groups=1)
        return th.mean(diff**2)

class LogSpecDiff_1st(th.nn.Module):
    def __init__(self):
        '''
        base class for losses that operate on the wave signal
        '''
        super().__init__()
    def forward(self, z_pred, z_true):
        return self._loss(z_pred, z_true)
    def _loss(self, z_pred, z_true):
        '''
        '''
        mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude')
        
        z_pred = mag2db(th.abs(z_pred.contiguous().permute(0,1,3,2))) # 440,2,77,128
        z_pred = z_pred.contiguous().view(-1,1,z_pred.shape[-1]) # 440*154, 128

        z_true = mag2db(th.abs(z_true.contiguous().permute(0,1,3,2))) # 440,2,77,128
        z_true = z_true.contiguous().view(-1,1,z_true.shape[-1]) # 440*154, 128

        kernel = th.tensor([-1,1],dtype=th.float32)
        kernel = th.tile(kernel, (1,1,1)).to(z_pred.device)

        diff_pred = F.conv1d(z_pred, kernel, groups=1)
        diff_true = F.conv1d(z_true, kernel, groups=1)
        return th.mean((diff_pred-diff_true)**2)

class LogSpecDiff_2nd(th.nn.Module):
    def __init__(self):
        '''
        base class for losses that operate on the wave signal
        '''
        super().__init__()
    def forward(self, z_pred, z_true):
        return self._loss(z_pred, z_true)
    def _loss(self, z_pred, z_true):
        '''
        '''
        mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude')
        
        z_pred = mag2db(th.abs(z_pred.contiguous().permute(0,1,3,2))) # 440,2,77,128
        z_pred = z_pred.contiguous().view(-1,1,z_pred.shape[-1]) # 440*154, 128

        z_true = mag2db(th.abs(z_true.contiguous().permute(0,1,3,2))) # 440,2,77,128
        z_true = z_true.contiguous().view(-1,1,z_true.shape[-1]) # 440*154, 128

        kernel = th.tensor([-1/2,0,1/2],dtype=th.float32)
        kernel = th.tile(kernel, (1,1,1)).to(z_pred.device)

        diff_pred = F.conv1d(z_pred, kernel, groups=1)
        diff_true = F.conv1d(z_true, kernel, groups=1)
        return th.mean((diff_pred-diff_true)**2)

class LogSpecDiff_4th(th.nn.Module):
    def __init__(self):
        '''
        base class for losses that operate on the wave signal
        '''
        super().__init__()
    def forward(self, z_pred, z_true):
        return self._loss(z_pred, z_true)
    def _loss(self, z_pred, z_true):
        '''
        '''
        mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude')
        
        z_pred = mag2db(th.abs(z_pred.contiguous().permute(0,1,3,2))) # 440,2,77,128
        z_pred = z_pred.contiguous().view(-1,1,z_pred.shape[-1]) # 440*154, 128

        z_true = mag2db(th.abs(z_true.contiguous().permute(0,1,3,2))) # 440,2,77,128
        z_true = z_true.contiguous().view(-1,1,z_true.shape[-1]) # 440*154, 128

        kernel = th.tensor([+1/12,-8/12,+8/12,-1/12],dtype=th.float32)
        kernel = th.tile(kernel, (1,1,1)).to(z_pred.device)

        diff_pred = F.conv1d(z_pred, kernel, groups=1)
        diff_true = F.conv1d(z_true, kernel, groups=1)
        return th.mean((diff_pred-diff_true)**2)

class LogSpecDiff_general(th.nn.Module):
    def __init__(self, order=2, activation=None, activate_coeff=1.0, leak_coeff=0.1):
        '''
        base class for losses that operate on the wave signal
        '''
        super().__init__()
        self.order = order
        self.activation = activation
        self.activate_coeff = activate_coeff
        self.leak_coeff = leak_coeff
    def forward(self, z_pred, z_true):
        return self._loss(z_pred, z_true)
    def _loss(self, z_pred, z_true):
        '''
        '''
        mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude')
        
        z_pred = mag2db(th.abs(z_pred.contiguous().permute(0,1,3,2))) # 440,2,77,128
        z_pred = z_pred.contiguous().view(-1,1,z_pred.shape[-1]) # 440*154, 128

        z_true = mag2db(th.abs(z_true.contiguous().permute(0,1,3,2))) # 440,2,77,128
        z_true = z_true.contiguous().view(-1,1,z_true.shape[-1]) # 440*154, 128

        if self.order == 1:
            kernel = th.tensor([-1,1],dtype=th.float32)
        elif self.order == 2:
            kernel = th.tensor([-1/2,0,1/2],dtype=th.float32)
        elif self.order == 4:
            kernel = th.tensor([+1/12,-8/12,+8/12,-1/12],dtype=th.float32)
        else:
            raise NotImplementedError
        kernel = th.tile(kernel, (1,1,1)).to(z_pred.device)

        diff_pred = F.conv1d(z_pred, kernel, groups=1) * self.activate_coeff
        diff_true = F.conv1d(z_true, kernel, groups=1) * self.activate_coeff
        if self.activation == None:
            pass
        elif self.activation == 'sign':
            diff_pred = th.sign(diff_pred)
            diff_true = th.sign(diff_true)
        elif self.activation == 'tanh':
            diff_pred = th.tanh(diff_pred)
            diff_true = th.tanh(diff_true)
        elif self.activation == 'leakysign':
            diff_pred = th.sign(diff_pred) + self.leak_coeff * diff_pred
            diff_true = th.sign(diff_true) + self.leak_coeff * diff_true
        elif self.activation == 'leakytanh':
            diff_pred = th.tanh(diff_pred) + self.leak_coeff * diff_pred
            diff_true = th.tanh(diff_true) + self.leak_coeff * diff_true
        return th.mean((diff_pred-diff_true)**2)

class l2diffLoss(th.nn.Module):
    def __init__(self):
        '''
        base class for losses that operate on the wave signal
        '''
        super().__init__()
    def forward(self, z):
        return self._loss(z)
    def _loss(self, z):
        '''
        :param z: predicted wave signals in a B x dim_z tensor
        :return: a scalar loss value
        '''
        z = z.permute(0,1,3,2) # 440,2,77,128
        z = z.view(-1,1,z.shape[-1]) # 440*154,1,128
        z_re = th.real(z)
        z_im = th.imag(z) 
        zz = th.cat((z_re,z_im),dim=0) # 2*440*154,1,128
        kernel = th.tensor([1,-1],dtype=th.float32)
        kernel = th.tile(kernel, (1,1,1)).to(z.device)
        # print(kernel.shape)
        diff = F.conv1d(zz, kernel, groups=1)
        return th.mean(diff**2)

class CosSimLoss(th.nn.Module):
    def __init__(self):
        '''
        base class for losses that operate on the wave signal
        '''
        super().__init__()
    def forward(self, x, y):
        return self._loss(x, y)
    def _loss(self, x, y):
        '''
        :param x, y:  B x dim tensor
        :return: a scalar loss value
        '''
        return th.mean((1-F.cosine_similarity(x,y)).pow(2))
        # return th.mean(th.acos(th.clamp(F.cosine_similarity(x,y), min=-1, max=1)).pow(2))

class L2Loss_angle(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        data, target in [0,2pi)
        :return: a scalar loss value
        '''
        d = th.min(data - target, 2*np.pi - (data - target))
        return th.mean(th.abs(d).pow(2))

class RegLoss(th.nn.Module):
    def __init__(self):
        '''
        base class for losses that operate on the wave signal
        '''
        super().__init__()
    def forward(self, coeff, n_vec):
        return self._loss(coeff, n_vec)
    def _loss(self, coeff, n_vec):
        '''
        :param coeff: coefficient # 2,(maxto+1)**2,filter_length, sub
        :param n_vec: 
        :return: a scalar loss value
        '''
        D = th.from_numpy(1+(n_vec*(n_vec+1)).astype(np.float32)).clone().to(coeff.device)
        # print(D.shape) # torch.Size([1296])
        # print(f"D:{D}") # D:tensor([1.0000e+00, 3.0000e+00, 3.0000e+00,  ..., 1.2610e+03, 1.2610e+03,1.2610e+03], device='cuda:0')
        return th.mean(D[None,:,None,None]*(th.abs(coeff).pow(2)))


class LSD(th.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, data, target, dim=2, data_kind='HRTF', mean=True):
        '''
        :param data:   (B,2,L,S) complex (or float) tensor
        :param target: (B,2,L,S) complex (or float) tensor
        :return: a scalar or (B,2,S) tensor
        '''
        if data_kind == 'HRTF':
            mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude')
            data   = mag2db(th.abs(data))
            target = mag2db(th.abs(target))
        elif data_kind == 'HRTF_mag':
            pass
        # return th.sqrt(th.mean((data_db - target_db).pow(2)))
        # ic(data.device)
        # ic(target.device)
        LSD = th.sqrt(th.mean((data - target).pow(2), dim=dim))
        if mean:
            LSD = th.mean(LSD)
        
        # del data, target

        return LSD

class ILD_AE(th.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, data, target, dim=1, data_kind='HRTF', mean=True):
        '''
        :param data:   (B,2,L,S) complex (or float) tensor
        :param target: (B,2,L,S) complex (or float) tensor
        :return: a scalar or (B,2,S) tensor
        '''
        assert data.shape[dim]==2 and target.shape[dim]==2

        if data_kind == 'HRTF':
            mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude')
            data   = mag2db(th.abs(data))
            target = mag2db(th.abs(target))
        elif data_kind == 'HRTF_mag':
            pass
        # return th.sqrt(th.mean((data_db - target_db).pow(2)))
        # ic(data.device)
        # ic(target.device)
        # B,1,L,S
        ILD_AE = th.abs(th.diff(data, dim=dim) - th.diff(target, dim=dim))
        if mean:
            ILD_AE = th.mean(ILD_AE)
        return ILD_AE

class LSD_before_mean(Loss):
    def _loss(self, data, target):
        '''
        :param data:   (B,2,L,S) complex (or float) tensor
        :param target: (B,2,L,S) complex (or float) tensor
        :return: LSD (B,2,S)
        '''
        mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude')
        data_db   = mag2db(th.abs(data))
        target_db = mag2db(th.abs(target))
        # return th.sqrt(th.mean((data_db - target_db).pow(2)))
        return th.sqrt(th.mean((data_db - target_db).pow(2), dim=2))

class ItakuraSaito(th.nn.Module):
    def __init__(self, bottom='data'):
        '''
        y(top) < x(bottom) となることが期待される
        '''
        self.bottom = bottom
        super().__init__()

    def forward(self, data, target):
        return self._loss(data, target)

    def _loss(self, data, target):
        '''
        :param data:   (B,2,L,S) complex (or float) tensor
        :param target: (B,2,L,S) complex (or float) tensor
        :return: a scalar loss value
        y(top) < x(bottom) となることが期待される
        '''
        # mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude')
        # data_db   = mag2db(th.abs(data))
        # target_db = mag2db(th.abs(target))
        if self.bottom == 'data':
            x, y = th.abs(data), th.abs(target)
        elif self.bottom == 'target':
            y, x = th.abs(data), th.abs(target)
        eps = 1e-10 * th.ones_like(x)
        D_IS = y/(x+eps) - th.log(y/(x+eps)) - 1
        # return th.sqrt(th.mean((data_db - target_db).pow(2)))
        return th.mean(D_IS)

class NotchPriorWeightedLSD(Loss):
    def __init__(self, gamma=10, epsilon=2):
        '''
        gamma >=0: if set to 0, equal to normal LSD
        epsilon >0: set smaller to emphasize notch
        '''
        self.gamma = gamma
        self.epsilon = epsilon
        super().__init__()
    def forward(self, data, target):
        return self._loss(data, target)
    def _loss(self, data, target):
        '''
        ノッチ（正確には振幅が小さいビン）を優先したLSD
        :param data:   (B,2,L,S) complex (or float) tensor
        :param target: (B,2,L,S) complex (or float) tensor
        :return: a scalar loss value
        '''
        mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude')
        data_db   = mag2db(th.abs(data))
        target_db = mag2db(th.abs(target))
        # return th.sqrt(th.mean((data_db - target_db).pow(2)))
        # top = (data_db - target_db).pow(2)
        # bottom = 1 + self.gamma * (target_db - th.min(target_db,dim=2) + self.epsilon)
        diff = (data_db - target_db).pow(2)
        mult = 1 + self.gamma / (target_db - th.min(target_db,dim=2)[0][:,:,None,:] + self.epsilon)

        return th.mean(th.sqrt(th.mean(diff*mult, dim=2)))

class WeightedLSD(Loss):
    def _loss(self, data, target):
        '''
        高域が重いLSD
        :param data:   (B,2,L,S) complex (or float) tensor
        :param target: (B,2,L,S) complex (or float) tensor
        :return: a scalar loss value
        '''
        mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude')
        data_db   = mag2db(th.abs(data))
        target_db = mag2db(th.abs(target))
        freq_bins = data.shape[2]
        weight_freq = th.ones(freq_bins).to(data.device)
        weight_freq[round(freq_bins/2):] *= 2
        # return th.sqrt(th.mean(((data_db-target_db)*weight_freq[None,None,:,None]).pow(2)))
        return th.mean(th.sqrt(th.mean(((data_db - target_db)*weight_freq[None,None,:,None]).pow(2), dim=2)))

class PhaseLoss(Loss):
    def _loss(self, data, target, l=2):
        '''
        :param data:   (B,2,L,S) complex (or float) tensor
        :param target: (B,2,L,S) complex (or float) tensor
        :return: a scalar loss value
        '''
        data_arg = th.atan2(th.imag(data),th.real(data))
        target_arg = th.atan2(th.imag(target),th.real(target))
        diff_arg = th.abs(data_arg - target_arg)
        # print(diff_arg.shape)
        # print(th.max(diff_arg))
        # print(th.min(diff_arg))
        diff_arg = np.pi - th.abs(diff_arg - np.pi) # triangle function over [0, 2pi] with peak at pi
        # diff_arg = th.min(diff_arg, 2*np.pi*th.ones_like(diff_arg)-diff_arg)
        # print(diff_arg.shape)
        # print(th.max(diff_arg))
        # print(th.min(diff_arg))
        
        return (th.mean(diff_arg**l))**(1/l)

class HelmholtzLoss(th.nn.Module):
    def __init__(self):
        '''
        base class for losses that operate on the wave signal
        '''
        super().__init__()
    def forward(self, rf, thf, phf,p, k, L, B):
        return self._loss(rf, thf, phf,p, k, L, B)
    def _loss(self, rf, thf, phf,func, k, L, B):
        '''
        rf: flattened radius
        thf: flattened zenith
        phf: flattened azimuth
        func: function (B,ch,L)
        k: wavenumber vector
        L: filter length
        B: batch size
        '''
        Helmholtz_loss = 0
        for ch in range(func.shape[1]):
            p = func[:,ch,:]
            # print(p)
            pf = th.flatten(p.permute(1,0))
            # print(pf)

            dammy_ones = th.ones(pf.shape[0],).cuda()
            p_r = autograd.grad(pf,rf, dammy_ones,retain_graph=True, create_graph=True)[0]
            p_r = p_r.view(L,B).permute(1,0)
            
            p_p = autograd.grad(pf,phf, dammy_ones, retain_graph=True, create_graph=True)[0]
            p_p = p_p.view(L,B).permute(1,0)

            p_t = autograd.grad(pf,thf, dammy_ones, retain_graph=True, create_graph=True)[0]
            p_t = p_t.view(L,B).permute(1,0)

            p_rf = th.flatten(p_r.permute(1,0))
            p_pf = th.flatten(p_p.permute(1,0))
            p_tf = th.flatten(p_t.permute(1,0))
            
            p_rr = autograd.grad((rf**2)*p_rf,rf, dammy_ones,retain_graph=True, create_graph=True)[0]
            p_tt = autograd.grad(th.sin(thf)*p_tf,thf, dammy_ones,retain_graph=True, create_graph=True)[0]
            p_pp = autograd.grad(p_pf,phf, dammy_ones,retain_graph=True, create_graph=True)[0]

            # print(p_rf.shape)
            # print(p_rr.shape)
            # print(rf.shape)
            eps = 1e-10 * th.ones(thf.shape).cuda()
            term_r = p_rr/(rf**2)
            term_t = p_tt/((rf**2)*th.max(th.sin(thf),eps))
            term_p = p_pp/((rf**2)*(th.max(th.sin(thf),eps)**2))

            lap = term_r + term_t + term_p
            print(th.mean(term_r**2))
            print(th.mean(term_t**2))
            print(th.mean(term_p**2))
            # print(f"lap:{lap}")
            k2 = th.flatten(th.tile(k**2,(B,1)).permute(1,0)) * pf
            # print(th.tile(k**2,(B,1)))
            # print(f"pf:{pf}")
            # print(th.flatten(th.tile(k**2,(B,1)).permute(1,0)))

            Helmholtz = lap + k2
            Helmholtz = Helmholtz.view(L,B).permute(1,0)

            Helmholtz_loss += th.mean(Helmholtz**2)

        # returns = {
        #     "helmholtz": Helmholtz,
        #     "helmholtz_loss": Helmholtz_loss,
        #     "lap": lap.view(L,B).permute(1,0),
        #     "k2": k2.view(L,B).permute(1,0),
        #     "term_r": term_r.view(L,B).permute(1,0),
        #     "term_p": term_p.view(L,B).permute(1,0),
        #     "term_t": term_t.view(L,B).permute(1,0),
        # }
        Helmholtz_loss /= func.shape[1]
        return Helmholtz_loss

class HelmholtzLoss_Cart(th.nn.Module):
    def __init__(self):
        '''
        base class for losses that operate on the wave signal
        '''
        super().__init__()
    def forward(self, xf, yf, zf, func, k, L, B):
        return self._loss(xf, yf, zf, func, k, L, B)
    def _loss(self, xf, yf, zf, func, k, L, B):
        '''
        xf: flattened x
        yf: flattened y
        zf: flattened z
        func: function (B,ch,L)
        k: wavenumber vector
        L: filter length
        B: batch size
        '''
        Helmholtz_loss = 0
        for ch in range(func.shape[1]):
            p = func[:,ch,:]
            # print(p)
            pf = th.flatten(p.permute(1,0))
            # print(pf)

            dammy_ones = th.ones(pf.shape[0],).to(p.device)
            p_x = autograd.grad(pf,xf, dammy_ones,retain_graph=True, create_graph=True)[0]
            p_y = autograd.grad(pf,yf, dammy_ones,retain_graph=True, create_graph=True)[0]
            p_z = autograd.grad(pf,zf, dammy_ones,retain_graph=True, create_graph=True)[0]
            p_xx = autograd.grad(p_x,xf, dammy_ones,retain_graph=True, create_graph=True)[0]
            p_yy = autograd.grad(p_y,yf, dammy_ones,retain_graph=True, create_graph=True)[0]
            p_zz = autograd.grad(p_z,zf, dammy_ones,retain_graph=True, create_graph=True)[0]

            lap = p_xx + p_yy + p_zz
            # print(f"lap:{lap}")
            # print(th.mean(pf**2))
            # print(th.mean(p_x**2))
            # print(th.mean(p_xx**2))
            k2 = th.flatten(th.tile(k**2,(B,1)).permute(1,0)) * pf
            # print(th.tile(k**2,(B,1)))
            # print(f"pf:{pf}")
            # print(th.flatten(th.tile(k**2,(B,1)).permute(1,0)))

            Helmholtz = lap + k2
            Helmholtz = Helmholtz.view(L,B).permute(1,0)

            Helmholtz_loss += th.mean(Helmholtz**2)

        Helmholtz_loss /= func.shape[1]
        return Helmholtz_loss

class HelmholtzLoss_old(th.nn.Module):
    def __init__(self):
        '''
        base class for losses that operate on the wave signal
        '''
        super().__init__()
    def forward(self, net, f_th, srcpos):
        return self._loss(net, f_th, srcpos)
    def _loss(self, net, f_th, srcpos, c=343):
        # srcpos_dammy = th.ones([1,3]).cuda()
        B = srcpos.shape[0]
        srcpos = srcpos.clone()
        r = srcpos[:,0]
        r.requires_grad = True
        phi = srcpos[:,1] # azimuth
        phi.requires_grad = True
        theta = srcpos[:,2] # zenith
        theta.requires_grad = True
        p_f = net.forward(th.cat((r.view(-1,1),phi.view(-1,1),theta.view(-1,1)), dim=1),True)["output_f"]
        # print(p.shape) # B,2,L

        k = f_th/c
        Helmholtz_loss = th.zeros(r.shape, dtype=th.float32).cuda()
        grad_out = th.ones(r.shape, dtype=th.float32).cuda()
        
        for ch in range(2):
            for i,kk in enumerate(k):
                lap_tmp = th.zeros([B,2], dtype=th.float32).cuda()
                for j in range(2): # Re/Im
                    itr = i + j*k.shape[0]
                    p_tmp = p_f[:,ch,itr]
                    p_r = autograd.grad(p_tmp,r, grad_out, retain_graph=True, create_graph=True)[0]
                    # print(p_r)
                    p_t = autograd.grad(p_tmp,theta, grad_out, retain_graph=True, create_graph=True)[0]
                    p_p = autograd.grad(p_tmp,phi, grad_out, retain_graph=True, create_graph=True)[0]
                    # print(p_p)
                    p_rr = autograd.grad((r**2)*p_r,r, grad_out, retain_graph=True, create_graph=True)[0]
                    p_tt = autograd.grad(th.sin(theta)*p_t,theta, grad_out, retain_graph=True, create_graph=True)[0]
                    p_pp = autograd.grad(p_p,phi, grad_out, retain_graph=True, create_graph=True)[0]

                    lap_tmp[:,j] = p_rr/(r**2) + p_tt/((r**2)*th.sin(theta)) + p_pp/((r**2)*(th.sin(theta)**2))

                Helmholtz_loss += th.abs(
                    th.complex(lap_tmp[:,0],lap_tmp[:,1]) 
                    + (kk.to(th.complex64)**2) 
                    * th.complex(p_f[:,ch,i],p_f[:,ch,i+k.shape[0]])
                    ) ** 2
                # print(Helmholtz_loss[0])
                # print(i)
        # print(Helmholtz_loss.shape) # B,
        return th.mean(Helmholtz_loss)