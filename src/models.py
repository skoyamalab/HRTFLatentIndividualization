# from termios import B115200
# from turtle import forward
import sys
import math
import numpy as np
import torch as th
import torchaudio as ta
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from torch.distributions.multivariate_normal import MultivariateNormal
#src.
from src.utils import Net, SpecialFunc, calcMaxOrder, transformCoeff, transformCoeff_MagdBPhase, sph_harm_nmvec, cart2sph, sph2cart, aprox_t_des, aprox_reg_poly, nearest_pts, transformCoeff_MagdBPhase_CNN, plane_sample, parallel_planes_sample, replace_activation_function
from src.lebedev import genGrid
from src.arcfacemetrics import ArcMarginProduct
from src.attention import Attention, Pos2Weight
from icecream import ic
ic.configureOutput(includeContext=True)
# ic.disable()

class ResConvSingleKernelBlock(nn.Module):
    def __init__(self, channel=64, droprate=0.0, filter_length=128):
        super().__init__()
        self.layers1 = nn.Sequential(
            nn.Conv1d(channel,channel,kernel_size=1),
            nn.LayerNorm([channel,filter_length]),
            nn.ReLU(), 
            nn.Dropout(droprate),
            nn.Conv1d(channel,channel,kernel_size=1)
        )
        self.layers2 = nn.Sequential(
            nn.LayerNorm([channel,filter_length]),
            nn.ReLU(), 
            nn.Dropout(droprate)
        )
    def forward(self, input):
        out_mid = self.layers1(input)
        in_mid = out_mid + input # skip connection
        out = self.layers2(in_mid)
        return out

class ResConv2dBlock(nn.Module):
    def __init__(self, size, channel=64,kernel_size=(3,3), droprate=0.0):
        super().__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size, stride=1, padding=round((kernel_size[0]-1)/2)),
            nn.LayerNorm(size),
            nn.ReLU(), 
            nn.Dropout(droprate),
            nn.Conv2d(channel,channel,kernel_size, stride=1, padding=round((kernel_size[0]-1)/2))
        )
        self.layers2 = nn.Sequential(
            nn.LayerNorm(size),
            nn.ReLU(), 
            nn.Dropout(droprate)
        )
    def forward(self, input):
        # print(f'input: {input.shape}')
        out_mid = self.layers1(input)
        in_mid = out_mid + input # skip connection
        # print(f'input_mid: {in_mid.shape}')
        out = self.layers2(in_mid)
        return out

class ResConvTranspose2dBlock(nn.Module):
    def __init__(self, size, channel=64,kernel_size=(3,3), droprate=0.0):
        super().__init__()
        self.layers1 = nn.Sequential(
            nn.ConvTranspose2d(channel,channel,kernel_size, stride=1, padding=round((kernel_size[0]-1)/2)),
            nn.LayerNorm(size),
            nn.ReLU(), 
            nn.Dropout(droprate),
            nn.ConvTranspose2d(channel,channel,kernel_size, stride=1, padding=round((kernel_size[0]-1)/2))
        )
        self.layers2 = nn.Sequential(
            nn.LayerNorm(size),
            nn.ReLU(), 
            nn.Dropout(droprate)
        )
    def forward(self, input):
        out_mid = self.layers1(input)
        in_mid = out_mid + input # skip connection
        out = self.layers2(in_mid)
        return out

class ResLinearBlock(nn.Module):
    def __init__(self, channel=64, droprate=0.2, norm=None, bn_c = None):
        super().__init__()
        if norm == 'ln':
            self.layers1 = nn.Sequential(
                nn.Linear(channel,channel),
                nn.LayerNorm(channel),
                nn.ReLU(), 
                nn.Dropout(droprate),
                nn.Linear(channel,channel)
            )
            self.layers2 = nn.Sequential(
                nn.LayerNorm(channel),
                nn.ReLU(), 
                nn.Dropout(droprate)
            )
        elif norm == 'bn':
            self.layers1 = nn.Sequential(
                nn.Linear(channel,channel),
                nn.BatchNorm1d(bn_c),
                nn.ReLU(), 
                nn.Dropout(droprate),
                nn.Linear(channel,channel)
            )
            self.layers2 = nn.Sequential(
                nn.BatchNorm1d(bn_c),
                nn.ReLU(), 
                nn.Dropout(droprate)
            )
        elif norm == None:
            self.layers1 = nn.Sequential(
                nn.Linear(channel,channel),
                nn.ReLU(), 
                nn.Dropout(droprate),
                nn.Linear(channel,channel)
            )
            self.layers2 = nn.Sequential(
                nn.ReLU(), 
                nn.Dropout(droprate)
            )
    def forward(self, input):
        out_mid = self.layers1(input)
        in_mid = out_mid + input # skip connection
        out = self.layers2(in_mid)
        return out

class WSLinear(nn.Linear):
    "Scaled Weight Standardization in [Brock+21a]"
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

        nn.init.xavier_normal_(self.weight)
        # nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def standardize_weight(self, eps):
        mean = th.mean(self.weight, dim=1, keepdims=True)
        var = th.std(self.weight, dim=1, keepdims=True, unbiased=False) ** 2
        fan_in = self.weight.shape[-1]

        scale = th.rsqrt(th.max(var * fan_in, th.tensor(eps).to(var.device))) 
        weight = (self.weight - mean) * scale
        return weight

    def forward(self, input, eps=1e-4):
        weight = self.standardize_weight(eps)
        out = F.linear(input, weight, self.bias)
        # print(out.shape)
        # print(f"[WSLinear] mean:{th.mean(out):.3}, std:{th.mean(th.std(out, dim=-1)):.3}")
        return out

class ScaledMish(nn.Module):
    "[Brock+21b]. 1.592 \simeq 1/th.std(mish(x)), x \sim N(0,1)"
    "mean shift(proposed): 0.240 \simeq mean(mish(x)), x \sim N(0,1)"
    def __init__(self):
        super().__init__()
    def forward(self, input):
        out = (F.mish(input) - 0.240) * 1.592 
        # print(f"[ScaledMish] mean:{th.mean(out):.3}, std:{th.mean(th.std(out, dim=-1)):.3}")
        return out

class ResWSLinearBlock(nn.Module):
    def __init__(self, channel=64, droprate=0.2, alpha=0.2):
        super().__init__()
        self.layers1 = nn.Sequential(
            WSLinear(channel,channel),
            ScaledMish(), 
            nn.Dropout(droprate),
            WSLinear(channel,channel)
        )
        self.layers2 = nn.Sequential(
            ScaledMish(), 
            nn.Dropout(droprate)
        )
        self.alpha = alpha
        self.gain = nn.Parameter(th.ones(()))

    def forward(self, input):
        out_mid = self.layers1(input) * self.alpha
        out = out_mid + input # skip connection
        # out = out / (1 + self.alpha**2) ** 0.5
        out = out / (1 + self.alpha**2) * self.gain
        out = self.layers2(out)
        
        return out

class ResWSLinearBlock2(nn.Module):
    def __init__(self, channel=64, droprate=0.2, alpha=0.2, beta=1.0):
        super().__init__()

        self.layers0 = nn.Sequential(
            ScaledMish(), 
            nn.Dropout(droprate)
        )

        self.layers1 = nn.Sequential(
            WSLinear(channel,channel),
            ScaledMish(), 
            nn.Dropout(droprate),
            WSLinear(channel,channel),
            ScaledMish(), 
            nn.Dropout(droprate),
        )
    
        self.alpha = alpha
        self.beta  = beta
        self.skipinit_gain = nn.Parameter(th.zeros(()))
        # self.skipinit_gain = nn.Parameter(th.ones(()))

    def forward(self, input):
        out_mid = self.layers1(self.layers0(input) / self.beta) 
        out = out_mid * self.alpha * self.skipinit_gain + input 
        return out

class scale(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        out = input * self.alpha
        return out

class Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        mean = th.mean(input)
        std = th.mean(th.std(input, dim=-1))
        output = (input - mean) / std
        return output

class HyperConvSingleKernelBlock(nn.Module):
    def __init__(self, ch_in, ch_out, input_size=3, ch_hidden=32):
        super().__init__()
        self.ch_in  = ch_in
        self.ch_out = ch_out
        self.weight_layers = nn.Sequential(
            nn.Linear(input_size, ch_hidden),
            nn.LayerNorm(ch_hidden),
            nn.ReLU(), 
            nn.Linear(ch_hidden, ch_out * ch_in)
        )
        self.bias_layers = nn.Sequential(
            nn.Linear(input_size, ch_hidden),
            nn.LayerNorm(ch_hidden),
            nn.ReLU(), 
            nn.Linear(ch_hidden, ch_out)
        )

    def forward(self, input):
        x = input["x"]
        z = input["z"]
        B = x.shape[0] # batch size
        weight = self.weight_layers(z) # (B, ch_out * ch_in)
        weight = weight.view(B * self.ch_out, self.ch_in, 1) # (B * ch_out, ch_in)
        bias = self.bias_layers(z) # (B, ch_out)

        x = x.view(1, B * self.ch_in, -1) # (B * ch_out, L)

        out = F.conv1d(x, weight, groups=B)  # (1, B*ch_o, L)
        out = out.view(B, self.ch_out, -1)   # (B, ch_o, L)
        out = out + bias[:,:,None]
        
        return out

class HyperLinear(nn.Module):
    def __init__(self, ch_in, ch_out, input_size=3, ch_hidden=32, num_hidden=1, use_res=True, droprate=0.0):
        super().__init__()
        self.ch_in  = ch_in
        self.ch_out = ch_out

        #==== weight ======
        modules = [
            nn.Linear(input_size, ch_hidden),
            nn.LayerNorm(ch_hidden),
            nn.ReLU(),
            nn.Dropout(droprate)]
        if not use_res:
            for l in range(num_hidden):
                modules.extend([
                    nn.Linear(ch_hidden, ch_hidden),
                    nn.LayerNorm(ch_hidden),
                    nn.ReLU(),
                    nn.Dropout(droprate)
                ])
        else:
            for l in range(round(num_hidden/2)):
                modules.extend([ResLinearBlock(ch_hidden,droprate=0.0)])
        modules.extend([
             nn.Linear(ch_hidden, ch_out * ch_in)
        ])
        self.weight_layers = nn.Sequential(*modules)

        #==== bias ======
        modules = [
            nn.Linear(input_size, ch_hidden),
            nn.LayerNorm(ch_hidden),
            nn.ReLU(),
            nn.Dropout(droprate)]
        if not use_res:
            for l in range(num_hidden):
                modules.extend([
                    nn.Linear(ch_hidden, ch_hidden),
                    nn.LayerNorm(ch_hidden),
                    nn.ReLU(),
                    nn.Dropout(droprate)
                ])
        else:
            for l in range(round(num_hidden/2)):
                modules.extend([ResLinearBlock(ch_hidden,droprate=0.0)])
        modules.extend([
             nn.Linear(ch_hidden, ch_out)
        ])
        self.bias_layers = nn.Sequential(*modules)

    def forward(self, input):
        x = input["x"] # (..., ch_in)
        z = input["z"] # (..., input_size)
        batches = list(x.shape)[:-1] # (...,)
        num_batches = math.prod(batches)

        weight = self.weight_layers(z) # (..., ch_out * ch_in)
        # ic(weight.shape)
        weight = weight.reshape([num_batches, self.ch_out, self.ch_in]) # (num_batches, ch_out, ch_in)
        # ic(weight.shape)
        bias = self.bias_layers(z) # (..., ch_out)
        # ic(bias.shape)

        output = {}
        # ic(x.reshape(num_batches,-1,1).shape)
        wx = th.matmul(weight, x.reshape(num_batches,-1,1)).reshape(batches + [-1])
        # ic(wx.shape)
        output["x"] = wx + bias #F.linear(x, weight, bias)
        output["z"] = z
        
        return output

class HyperLinearBlock(nn.Module):
    def __init__(self, ch_in, ch_out, ch_hidden=32, num_hidden=1, droprate=0.0,use_res=True, input_size=3, post_prcs=True):
        super().__init__()
        self.hyperlinear = HyperLinear(ch_in, ch_out, ch_hidden=ch_hidden, num_hidden=num_hidden, use_res=use_res, input_size=input_size, droprate=droprate)
        if post_prcs:
            self.layers_post = nn.Sequential(
                nn.LayerNorm(ch_out),
                nn.ReLU(),
                nn.Dropout(droprate)
            )
        else:
            self.layers_post = nn.Sequential(
                nn.Identity()
            )

    def forward(self, input):
        # x = input["x"] # (B, ch_in)
        z = input["z"] # (B, input_size=3)
        # print(f'models.py l354 {z.shape}')
        # print(f'models.py l355 {input["x"].shape}')
        xz = self.hyperlinear(input) # x,z]
        # print(f'models.py l357 {xz["x"].shape}')
        # print(f'models.py l358 {xz["z"].shape}')

        output = {}
        output["x"] = self.layers_post(xz["x"])
        output["z"] = z
        
        return output

class ResHyperLinearBlock(nn.Module):
    def __init__(self, channel=64, ch_hyper_hidden=32, num_hyper_hidden=1, droprate=0.0, use_res=True, input_size=3):
        super().__init__()
        
        self.layers1_1 = nn.Sequential(
            HyperLinear(channel,channel, ch_hidden=ch_hyper_hidden, num_hidden=num_hyper_hidden, use_res=use_res, input_size=input_size))
        self.layers1_2 = nn.Sequential(
            nn.LayerNorm(channel),
            nn.ReLU(), 
            nn.Dropout(droprate)
        )
        self.layers1_3 = nn.Sequential(
            HyperLinear(channel,channel, ch_hidden=ch_hyper_hidden, num_hidden=num_hyper_hidden, use_res=use_res, input_size=input_size)
        )

        self.layers2 = nn.Sequential(
            nn.LayerNorm(channel),
            nn.ReLU(), 
            nn.Dropout(droprate)
        )
        
    def forward(self, input):
        x = input["x"]
        z = input["z"]

        out1_1 = self.layers1_1(input)       # x,z
        out1_2 = self.layers1_2(out1_1["x"]) # x
        in1_3 = {
            "x": out1_2,
            "z": z
        }
        out1_3 = self.layers1_3(in1_3)       # x,z
        out = {}
        out["x"] = self.layers2(x + out1_3["x"])
        out["z"] = z
        return out

class HyperLinear_FIAE(nn.Module): # Freq Independent AutoEncoder
    def __init__(self, ch_out, input_size=3, ch_hidden=32, num_hidden=1, use_res=True, droprate=0.0, use_bias=True, pinv=False, identity=True, reg_w=1e-3, freq_in=False, reg_mat_learn='diag', reg_mat_base='duraiswami', relax=True, rel_l=1.0):
        super().__init__()
        if identity and freq_in:
             input_size -= 1
        self.ch_out = ch_out
        self.input_size = input_size
        self.pinv = pinv
        self.identity = identity 
        self.reg_w = reg_w
        self.freq_in = freq_in
        self.reg_mat_learn = reg_mat_learn
        self.reg_mat_base = reg_mat_base
        self.relax = relax
        self.rel_l = rel_l

        #==== weight ======
        if identity:
            modules = [nn.Identity()]
            self.weight_layers = nn.Sequential(*modules)
            if reg_mat_learn == 'None':
                self.dammy = nn.Parameter(th.FloatTensor(1))
        else:
            modules = [
                nn.Linear(input_size, ch_hidden),
                nn.LayerNorm(ch_hidden),
                nn.ReLU(),
                nn.Dropout(droprate)]
            if not use_res:
                for l in range(num_hidden):
                    modules.extend([
                        nn.Linear(ch_hidden, ch_hidden),
                        nn.LayerNorm(ch_hidden),
                        nn.ReLU(),
                        nn.Dropout(droprate)
                    ])
            else:
                for l in range(round(num_hidden/2)):
                    modules.extend([ResLinearBlock(ch_hidden,droprate=0.0)])
            modules.extend([
                nn.Linear(ch_hidden, ch_out)
            ])

            if not pinv:
                # init to avoid nan
                # std_for_init = 1.0 if pinv else 0.001
                for layers in modules:
                    print(layers)
                    if hasattr(layers, "weight"):
                        nn.init.normal_(layers.weight, mean=0.0, std=0.01)
                        print("init_w")
                    if hasattr(layers, "bias"):
                        nn.init.constant_(layers.bias, 0.0)
                        print("init_b")
            self.weight_layers = nn.Sequential(*modules)
        

        self.use_bias = use_bias
        if use_bias:
            #==== bias ======
            if self.freq_in: # f_l -> bias
                modules = [
                    nn.Linear(input_size-3, ch_hidden),
                    nn.LayerNorm(ch_hidden),
                    nn.ReLU(),
                    nn.Dropout(droprate)]
                if not use_res:
                    for l in range(num_hidden):
                        modules.extend([
                            nn.Linear(ch_hidden, ch_hidden),
                            nn.LayerNorm(ch_hidden),
                            nn.ReLU(),
                            nn.Dropout(droprate)
                        ])
                else:
                    for l in range(round(num_hidden/2)):
                        modules.extend([ResLinearBlock(ch_hidden,droprate=0.0)])
                modules.extend([
                    nn.Linear(ch_hidden, ch_out)
                ])
                self.bias_layers = nn.Sequential(*modules)
            else:
                self.bias = nn.Parameter(th.FloatTensor(self.ch_out))

        if pinv and reg_mat_learn == 'diag' or self.reg_mat_learn == 'full':
            if reg_mat_learn == 'diag':
                ch_out_mat = ch_out 
            elif reg_mat_learn == 'full':
                ch_out_mat = ch_out**2
            if self.freq_in:
                modules = [
                nn.Linear(1, ch_hidden),
                nn.LayerNorm(ch_hidden),
                nn.ReLU(),
                nn.Dropout(droprate)]
                if not use_res:
                    for l in range(num_hidden):
                        modules.extend([
                            nn.Linear(ch_hidden, ch_hidden),
                            nn.LayerNorm(ch_hidden),
                            nn.ReLU(),
                            nn.Dropout(droprate)
                        ])
                else:
                    for l in range(round(num_hidden/2)):
                        modules.extend([ResLinearBlock(ch_hidden,droprate=0.0)])
                modules.extend([
                    nn.Linear(ch_hidden, ch_out_mat)
                ])
                for layers in modules:
                    if hasattr(layers, "weight"):
                        nn.init.normal(layers.weight, mean=0.0, std=0.0001)
                self.reg_mat_layers = nn.Sequential(*modules)
                
            else:
                self.reg_mat = nn.Parameter(th.zeros(ch_out_mat))
    
    def forward(self, input):
        output = {}
        x = input["x"] # (S,L,B)
        z = input["z"] # (S,L,B,input_size)
        S, L, B = x.shape
        # print(x.shape) # torch.Size([64, 128, 440])
        # print(z.shape) # torch.Size([64, 128, 440, 3])
        if self.identity and self.freq_in:
            z = z[:,:,:,:-1]
        weight = self.weight_layers(z) # (S,L,B,ch_out)
        # print(f'max|weight_HL_FIAE|:{th.max(th.abs(weight))}') # nan_debug
        # print(weight.shape) # torch.Size([64, 128, 440, 128])
        # print(weight[round(S/2)+1,0,:5,:4])
        output["weight"] = weight
        # print(weight.shape)
        weight = weight.view(S*L, B, self.ch_out)
        # print(th.max(th.abs(weight))) # tensor(1.4501, device='cuda:0', grad_fn=<MaxBackward1>)
        # print(x[0,:3,:5])
        x = x.reshape(S*L, 1, B)
        
        if self.pinv:
            # U, s, Vh = th.linalg.svd(weight)
            # eps = s[:,0] * 1e-4
            # s_inv = th.zeros_like(s) + 1/s * (s>eps[:,None])
            # s_inv_mat = th.diag_embed(s_inv, offset=U.shape[-1]-Vh.shape[-1])
            # s_inv_mat = s_inv_mat[:,:,U.shape[-1]-Vh.shape[-1]:]
            # weight = U @ s_inv_mat @ Vh
            # sys.exit()
            # print(th.bmm(weight.permute(0,2,1), weight).shape)
            # print(x.shape)
            #====== regularization =====
            if self.reg_mat_base == 'duraiswami':
                n_vec,_ = sph_harm_nmvec(round(self.ch_out**0.5-1))
                G = th.sqrt(th.diag(1+th.from_numpy((n_vec*(n_vec+1)).astype(np.float32)).clone()).to(x.device))
            elif self.reg_mat_base == 'identity':
                G = th.diag(th.ones(self.ch_out).to(x.device))
            elif self.reg_mat_base == 'zero':
                G = th.diag(th.zeros(self.ch_out).to(x.device))

            if self.reg_mat_learn == 'diag':
                if self.freq_in:
                    G_l = self.rel_l * self.reg_mat_layers(input["z"][:,:,0,-1:]) # (S,L,(N+1)^2)
                    G_l = th.diag_embed(G_l,offset=0,dim1=-2,dim2=-1) # (S,L,(N+1)^2,(N+1)^2)
                    G_l = G_l.reshape(S*L,self.ch_out,self.ch_out)  # (S*L,(N+1)^2,(N+1)^2)
                    G = G[None,:,:] + G_l
                else:
                    G = G + self.rel_l * th.diag(self.reg_mat)
            elif self.reg_mat_learn == 'full':
                if self.freq_in:
                    G_l = self.rel_l * self.reg_mat_layers(input["z"][:,:,0,-1:]) # (S,L,(N+1)^2*(N+1)^2)
                    G_l = G_l.reshape(S*L,self.ch_out,self.ch_out)  # (S*L,(N+1)^2,(N+1)^2)
                    G = G[None,:,:] + G_l
                else:
                    G = G + self.rel_l * self.reg_mat.reshape(self.ch_out,self.ch_out)
            elif self.reg_mat_learn == 'None':
                pass
            else:
                print("[G: Error] not implemented.")

            if G.dim() == 2:
                G = th.tile(G.unsqueeze(0), (S*L,1,1))
            D = self.reg_w * th.bmm(G.permute(0,2,1), G)
            eps = 1e-4
            D += eps * th.diag(th.ones(self.ch_out).to(x.device))[None,:,:]
            # print(D.shape)
            #============================
            # D:      (S*L, (N+1)^2, (N+1)^2)
            # weight: (S*L, B, (N+1)^2)
            # x:      (S*L, 1, B)
            if self.relax:
                wx = th.linalg.solve(th.bmm(weight.permute(0,2,1), weight)+D ,th.bmm(weight.permute(0,2,1), x.reshape(-1,B,1))).view(S, L, self.ch_out)
                # (W^T W + D)^{-1} W^T x (= W^T (W W^T + D)^{-1} x)
                # print(wx.shape)
            else:
                D_inv = 1 / (D +  1e-10 * th.ones_like(D))
                D_i_W_t = th.bmm(D_inv, weight.permute(0,2,1)) # D^{-1} W^T
                wx = th.linalg.solve(th.bmm(weight, D_i_W_t), x.reshape(-1,B,1))
                # (W D^{-1} W^T)^{-1} x
                wx = th.bmm(D_i_W_t, wx)
                # D^{-1} W^T (W D^{-1} W^T)^{-1} x
                wx = wx.view(S, L, self.ch_out)
        
        else:
            wx =  th.bmm(x, weight).view(S, L, self.ch_out) # (S,L,ch_out)
        if self.identity and self.reg_mat_learn=='None':
            wx = wx + th.zeros_like(wx) * self.dammy
            # print(wx.shape)
            # sys.exit()
            # print(wx[0,:3,:5])
            # sys.exit()
        if self.use_bias:
            if self.freq_in:
                bias = self.bias_layers(z[0:1,0,0,3:]).view(self.ch_out) # (ch_out)
            else:
                bias = self.bias # (ch_out)
            output["x"] = wx + bias[None,None,:]
        else:
            output["x"] = wx
        output["z"] = z
        # sys.exit()
        # print(f'max|wx_HL_FIAE|:{th.max(th.abs(wx))}') # nan_debug
        return output
        # return output["x"]

class HyperLinear_FIAE_Trans(nn.Module): # Freq Independent AutoEncoder
    def __init__(self, ch_in, input_size=3, ch_hidden=32, num_hidden=1, use_res=True, droprate=0.0, use_bias=True, identity=True,  freq_in=False):
        super().__init__()
        if identity and freq_in:
            input_size -= 1
        self.ch_in = ch_in
        self.input_size = input_size
        self.identity = identity
        self.freq_in = freq_in
        
        #==== weight ======
        if identity:
            modules = [nn.Identity()]
            self.weight_layers = nn.Sequential(*modules)
        else:
            modules = [
                nn.Linear(input_size, ch_hidden),
                nn.LayerNorm(ch_hidden),
                nn.ReLU(),
                nn.Dropout(droprate)]
            if not use_res:
                for l in range(num_hidden):
                    modules.extend([
                        nn.Linear(ch_hidden, ch_hidden),
                        nn.LayerNorm(ch_hidden),
                        nn.ReLU(),
                        nn.Dropout(droprate)
                    ])
            else:
                for l in range(round(num_hidden/2)):
                    modules.extend([ResLinearBlock(ch_hidden,droprate=0.0)])
            modules.extend([
                nn.Linear(ch_hidden, ch_in)
            ])
            # init to avoid nan
            for layers in modules:
                if hasattr(layers, "weight"):
                    nn.init.normal(layers.weight, mean=0.0, std=0.001)
            self.weight_layers = nn.Sequential(*modules)
        

        self.use_bias = use_bias
        if use_bias:
            #==== bias ======
            modules = [
                nn.Linear(input_size, ch_hidden),
                nn.LayerNorm(ch_hidden),
                nn.ReLU(),
                nn.Dropout(droprate)]
            if not use_res:
                for l in range(num_hidden):
                    modules.extend([
                        nn.Linear(ch_hidden, ch_hidden),
                        nn.LayerNorm(ch_hidden),
                        nn.ReLU(),
                        nn.Dropout(droprate)
                    ])
            else:
                for l in range(round(num_hidden/2)):
                    modules.extend([ResLinearBlock(ch_hidden,droprate=0.0)])
            modules.extend([
                nn.Linear(ch_hidden, 1)
            ])
            self.bias_layers = nn.Sequential(*modules)

    def forward(self, input):
        output = {}
        x = input["x"] # (S,L,ch_in)
        z = input["z"] # (S,L,B,input_size)
        S, L, B = z.shape[:-1]
        if self.identity and self.freq_in:
            z = z[:,:,:,:-1]
        weight = self.weight_layers(z) # (S,L,B,ch_in)
        output["weight"] = weight
        weight = weight.permute(0,1,3,2) # (S,L,ch_in,B)
        # print(f"[debug] weight:{weight.shape}, x:{x.shape}")
        weight = weight.view(S*L, self.ch_in, B)
        x = x.view(S*L, 1, self.ch_in)
        wx =  th.bmm(x, weight).view(S, L, B) # (S,L,B)
        if self.use_bias:
            bias = self.bias_layers(z).view(S,L,B) 
            output["x"] = wx + bias
        else:
            output["x"] = wx
        output["z"] = z
        
        return output
        # return output["x"]

class ResHCSKB(nn.Module):
    def __init__(self, channel=64, droprate=0.0, filter_length=128, input_size=3):
        super().__init__()
        self.layers1 = nn.Sequential(
            HyperConvSingleKernelBlock(channel,channel,input_size))
        self.layers2 = nn.Sequential(
            nn.LayerNorm(filter_length),
            nn.ReLU(), 
            nn.Dropout(droprate)
        )
        self.layers3 = nn.Sequential(
            HyperConvSingleKernelBlock(channel,channel,input_size))
        
        self.layers_out = nn.Sequential(
            nn.LayerNorm(filter_length),
            nn.ReLU(), 
            nn.Dropout(droprate)
        )
    def forward(self, input):
        x = input["x"]
        z = input["z"]
        out_mid = self.layers1({"x":x,"z":z})
        out_mid = self.layers2(out_mid)
        out_mid = self.layers3({"x":out_mid,"z":z})
        in_mid = out_mid + x # skip connection
        out = self.layers_out(in_mid)
        return {"x":out, "z":z}

class HyperConv1d(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, input_size=3, ch_hidden=32, num_hidden=1, use_res=True, droprate=0.0, use_freq=True):
        super().__init__()
        if use_freq:
            ch_in += 1
        self.ch_in  = ch_in
        self.ch_out = ch_out
        self.kernel_size = kernel_size
        self.use_freq = use_freq

        #==== weight ======
        modules = [
            nn.Linear(input_size, ch_hidden),
            nn.LayerNorm(ch_hidden),
            nn.ReLU(),
            nn.Dropout(droprate)]
        if not use_res:
            for l in range(num_hidden):
                modules.extend([
                    nn.Linear(ch_hidden, ch_hidden),
                    nn.LayerNorm(ch_hidden),
                    nn.ReLU(),
                    nn.Dropout(droprate)
                ])
        else:
            for l in range(round(num_hidden/2)):
                modules.extend([ResLinearBlock(ch_hidden,droprate=0.0)])
        modules.extend([
             nn.Linear(ch_hidden, ch_out * ch_in * kernel_size)
        ])
        self.weight_layers = nn.Sequential(*modules)

        #==== bias ======
        modules = [
            nn.Linear(input_size, ch_hidden),
            nn.LayerNorm(ch_hidden),
            nn.ReLU(),
            nn.Dropout(droprate)]
        if not use_res:
            for l in range(num_hidden):
                modules.extend([
                    nn.Linear(ch_hidden, ch_hidden),
                    nn.LayerNorm(ch_hidden),
                    nn.ReLU(),
                    nn.Dropout(droprate)
                ])
        else:
            for l in range(round(num_hidden/2)):
                modules.extend([ResLinearBlock(ch_hidden,droprate=0.0)])
        modules.extend([
             nn.Linear(ch_hidden, ch_out)
        ])
        self.bias_layers = nn.Sequential(*modules)

    def forward(self, input):
        if self.use_freq:
            f = input["f"] # (B, 1, L)
        x = input["x"] # (B, ch_in or ch_in-1, L)
        z = input["z"] # (B, input_size=3)
        B = x.shape[0] # batch size
        weight = self.weight_layers(z) # (B,ch_out * ch_in * kernel_size)
        weight = weight.view(B, self.ch_out, self.ch_in, self.kernel_size) # (B, ch_out, ch_in, kernel_size)
        bias = self.bias_layers(z) # (B, ch_out)

        # dbg
        # print(f"[HConv1d] x.shape: {x.shape}")
        # print(f"[HConv1d] z.shape: {z.shape}")
        # print(f"[HConv1d] weight.shape: {weight.shape}")
        output = {}
        if self.use_freq:
            x = th.cat((x,f), dim=1) # # (B, ch_in, L)
        # batch-wise conv. implemented by grouped conv.
        x = x.reshape(1, B*self.ch_in, -1)
        weight = weight.reshape(B*self.ch_out, self.ch_in, self.kernel_size)
        # dbg
        # print(f"[HConv1d] x.shape: {x.shape}")
        # print(f"[HConv1d] weight.shape: {weight.shape}")
        wx = F.conv1d(x, weight, padding='same', groups=B).reshape(B, self.ch_out, -1) # (B, ch_out, L)
        #
        wx = wx + bias[:,:,None]
        # print(f"[HConv1d] wx.shape: {wx.shape}") # dbg
        output["x"] = wx
        output["z"] = z
        if self.use_freq:
            output["f"] = f # (B, 1, L)
        
        return output

class HyperConv1dBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, ch_hidden=32, num_hidden=1, droprate=0.0,use_res=True, input_size=3, post_prcs=True, use_freq=True, normalized_dim=[-2,-1], normalized_shape=None):
        super().__init__()
        self.hyperconv1d = HyperConv1d(ch_in, ch_out, kernel_size=kernel_size, ch_hidden=ch_hidden, num_hidden=num_hidden, use_res=use_res, input_size=input_size, droprate=droprate, use_freq=use_freq)
        # if dim_normalize==1:
        #     length = ch_out
        assert len(normalized_dim)==len(normalized_shape)
        if post_prcs:
            self.layers_post = nn.Sequential(
                nn.LayerNorm(normalized_shape),
                nn.ReLU(),
                nn.Dropout(droprate)
            )
        else:
            self.layers_post = nn.Sequential(
                nn.Identity()
            )
        self.use_freq=use_freq
        self.normalized_dim = normalized_dim

    def forward(self, input):
        # x = input["x"] # (B, ch_in, L)
        # z = input["z"] # (B, input_size=3)
        xz = self.hyperconv1d(input) # x,z

        output = {}
        if self.normalized_dim == [-2]:
            output["x"] = self.layers_post(xz["x"].permute(0,2,1)).permute(0,2,1) # (B, ch_out, L) -> (B, L, ch_out) -> (B, ch_out, L)
        else: # [-2,-1],[-1]
            output["x"] = self.layers_post(xz["x"]) # (B, ch_out, L)
        output["z"] = input["z"]
        if self.use_freq:
            output["f"] = input["f"]
        
        return output

class HyperConvTranspose1d(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, input_size=3, ch_hidden=32, num_hidden=1, use_res=True, droprate=0.0, use_freq=True):
        super().__init__()
        if use_freq:
            ch_in += 1
        self.ch_in  = ch_in
        self.ch_out = ch_out
        self.kernel_size = kernel_size
        self.use_freq = use_freq

        #==== weight ======
        modules = [
            nn.Linear(input_size, ch_hidden),
            nn.LayerNorm(ch_hidden),
            nn.ReLU(),
            nn.Dropout(droprate)]
        if not use_res:
            for l in range(num_hidden):
                modules.extend([
                    nn.Linear(ch_hidden, ch_hidden),
                    nn.LayerNorm(ch_hidden),
                    nn.ReLU(),
                    nn.Dropout(droprate)
                ])
        else:
            for l in range(round(num_hidden/2)):
                modules.extend([ResLinearBlock(ch_hidden,droprate=0.0)])
        modules.extend([
             nn.Linear(ch_hidden, ch_out * ch_in * kernel_size)
        ])
        self.weight_layers = nn.Sequential(*modules)

        #==== bias ======
        modules = [
            nn.Linear(input_size, ch_hidden),
            nn.LayerNorm(ch_hidden),
            nn.ReLU(),
            nn.Dropout(droprate)]
        if not use_res:
            for l in range(num_hidden):
                modules.extend([
                    nn.Linear(ch_hidden, ch_hidden),
                    nn.LayerNorm(ch_hidden),
                    nn.ReLU(),
                    nn.Dropout(droprate)
                ])
        else:
            for l in range(round(num_hidden/2)):
                modules.extend([ResLinearBlock(ch_hidden,droprate=0.0)])
        modules.extend([
             nn.Linear(ch_hidden, ch_out)
        ])
        self.bias_layers = nn.Sequential(*modules)

    def forward(self, input):
        if self.use_freq:
            f = input["f"] # (B, 1, L)
        x = input["x"] # (B, ch_in or ch_in-1, L)
        z = input["z"] # (B, input_size=3)
        B = x.shape[0] # batch size
        weight = self.weight_layers(z) # (B,ch_out * ch_in * kernel_size)
        weight = weight.view(B, self.ch_in, self.ch_out, self.kernel_size) # (B, ch_in, ch_out, kernel_size)
        bias = self.bias_layers(z) # (B, ch_out)

        output = {}
        if self.use_freq:
            x = th.cat((x,f), dim=1) # # (B, ch_in, L)
        # batch-wise conv. implemented by grouped conv.
        # dbg
        # print(f"[HConvT1d] x.shape: {x.shape}")
        # print(f"[HConvT1d] z.shape: {z.shape}")
        # print(f"[HConvT1d] weight.shape: {weight.shape}")
        x = x.reshape(1, B*self.ch_in, -1)
        weight = weight.reshape(B*self.ch_in, self.ch_out, self.kernel_size)
        wx = F.conv_transpose1d(x, weight, stride=1, padding=(self.kernel_size-1)//2, groups=B).reshape(B, self.ch_out, -1) # (B, ch_out, L)
        wx = wx + bias[:,:,None]
        output["x"] = wx
        output["z"] = z
        if self.use_freq:
            output["f"] = f # (B, 1, L)
        
        return output

class HyperConvTranspose1dBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, ch_hidden=32, num_hidden=1, droprate=0.0,use_res=True, input_size=3, post_prcs=True, use_freq=True, normalized_dim=[-2,-1], normalized_shape=None):
        super().__init__()
        self.hyperconvT1d = HyperConvTranspose1d(ch_in, ch_out, kernel_size=kernel_size, ch_hidden=ch_hidden, num_hidden=num_hidden, use_res=use_res, input_size=input_size, droprate=droprate, use_freq=use_freq)

        assert len(normalized_dim)==len(normalized_shape)

        if post_prcs:
            self.layers_post = nn.Sequential(
                nn.LayerNorm(normalized_shape),
                nn.ReLU(),
                nn.Dropout(droprate)
            )
        else:
            self.layers_post = nn.Sequential(
                nn.Identity()
            )
        self.use_freq = use_freq
        self.normalized_dim = normalized_dim

    def forward(self, input):
        # x = input["x"] # (B, ch_in, L)
        z = input["z"] # (B, input_size=3)
        xz = self.hyperconvT1d(input) # x,z

        output = {}

        if self.normalized_dim == [-2]:
            output["x"] = self.layers_post(xz["x"].permute(0,2,1)).permute(0,2,1) # (B, ch_out, L) -> (B, L, ch_out) -> (B, ch_out, L)
        else: # [-2,-1],[-1]
            output["x"] = self.layers_post(xz["x"]) # (B, ch_out, L)

        output["z"] = z
        if self.use_freq:
            output["f"] = input["f"]
        
        return output

def conv1d_scratch(input, weight, bias=None, pad='same', device='cuda'):
    '''
    args:
        input: (bs, in_ch, in_length)
        weight: (bs, in_ch, out_ch, out_length, kernel_size)
            * out_length == in_length+2*pad-kernel_size+1
        bias: None or (bs, out_ch, out_length)
        pad: int or str (if 'same' in_length==out_length)
    return:
        output:  (bs, out_ch, out_length)
    '''
    # device = input.device
    bs, in_ch, in_length = input.shape
    _, _, out_ch, _, kernel_size = weight.shape
    if pad == 'same':
        assert kernel_size % 2 == 1
        pad = (kernel_size - 1)//2
    out_length = in_length + 2*pad - kernel_size + 1
    # print(f'[conv1d_scratch] input:{input.shape}, weight:{weight.shape}')
    assert weight.shape[3] == out_length

    # print([bs, in_ch, out_ch, out_length, in_length+2*pad])
    #------------
    # kernel_matrix = th.zeros(bs, in_ch, out_ch, out_length, in_length+2*pad).to(device) # (bs, ch_in, ch_out, L+2p-ks+1, L+2p)
    # kernel_matrix[:,:,:,:,:kernel_size] = weight
    # idx_gather = th.arange(in_length+2*pad).to(device)
    # idx_offset = th.arange(out_length).to(device)
    # idx_gather = (idx_gather[None,:] - idx_offset[:,None]) % (in_length+2*pad) # (L+2p-ks+1, L+2p)
    # idx_gather = idx_gather.reshape(1,1,1,out_length,in_length+2*pad).tile(bs,in_ch, out_ch, 1, 1)  # (bs, ch_in, ch_out, L+2p-ks+1, L+2p)
    # kernel_matrix = th.gather(kernel_matrix, -1, idx_gather)

    # input_pad = F.pad(input, (pad,pad), "constant", 0) # (ch_in, L+2p)
    # input_pad  = th.reshape(input_pad, (bs, in_ch,  1, in_length+2*pad, 1)).tile(1,1,out_ch,1,1) # (bs, ch_in, ch_out, L+2p, 1)

    # output = th.matmul(kernel_matrix, input_pad) # (bs, ch_in, ch_out, L+2p, 1)
    # output = th.sum(output, dim=1).reshape(bs, out_ch, out_length)  # (bs, ch_out, L+2p)
    #------------
    weight = weight.reshape(bs, in_ch, out_ch, out_length, 1, kernel_size) # bs, in_ch, out_ch, out_length, 1, ks
    input_pad = F.pad(input, (pad,pad), "constant", 0) # (bs, in_ch, L+2p)
    input_uf = input_pad.unfold(-1, kernel_size, 1) #  # (bs, in_ch, out_length, ks)
    input_uf = input_uf.reshape(bs, in_ch, 1, out_length, kernel_size, 1).tile(1,1,out_ch,1,1,1) # bs, in_ch, out_ch, out_length, ks, 1
    output = th.matmul(weight, input_uf) # (bs, ch_in, ch_out, out_length, 1, 1)
    output = th.sum(output, dim=1).reshape(bs, out_ch, out_length)  # (bs, ch_out, out_length)
    #------------

    if not bias == None:
        output = output + bias

    return output

def convT1d_scratch(input, weight, bias=None, pad='same', device='cuda'):
    '''
    args:
        input: (bs, in_ch, in_length)
        weight: (bs, in_ch, out_ch, in_length, kernel_size)
        bias: None or (bs, out_ch, out_length)
            * out_length == in_length - 2*pad + kernel_size - 1
        pad: int or str (if 'same' in_length==out_length)
    return:
        output:  (bs, out_ch, out_length)
    '''
    # device = input.device
    bs, in_ch, in_length = input.shape
    _, _, out_ch, _, kernel_size = weight.shape
    if pad == 'same':
        assert kernel_size % 2 == 1
        pad = (kernel_size - 1)//2
    out_length = in_length - 2*pad + kernel_size - 1
    # print(f'[convT1d_scratch] input:{input.shape}, weight:{weight.shape}')
    if not bias == None:
        assert bias.shape[2] == out_length

    # print([bs, in_ch, out_ch, out_length, in_length+2*pad])

    #===================
    # kernel_matrix = th.zeros(bs, in_ch, out_ch, in_length, out_length+2*pad).to(device) # (bs, ch_in, ch_out, L+2p-ks+1, L+2p)
    # kernel_matrix[:,:,:,:,:kernel_size] = weight
    # idx_gather = th.arange(out_length+2*pad).to('cpu')
    # idx_offset = th.arange(in_length).to('cpu')
    # idx_gather = (idx_gather[None,:] - idx_offset[:,None]) % (out_length+2*pad) # (L+2p-ks+1, L+2p)
    # idx_gather = idx_gather.reshape(1,1,1,in_length,out_length+2*pad).tile(bs,in_ch, out_ch, 1, 1)  # (bs, ch_in, ch_out, L+2p-ks+1, L+2p)
    # kernel_matrix = th.gather(kernel_matrix, -1, idx_gather)
    # kernel_matrix = kernel_matrix.permute(0,1,2,4,3)  # (bs, ch_in, ch_out, L+2p, L+2p-ks+1)

    # input = th.reshape(input, (bs, in_ch, 1, in_length, 1)).tile(1,1,out_ch,1,1)  # (bs, ch_in, ch_out, L+2p-ks+1, 1)

    # output = output = th.matmul(kernel_matrix, input) # (bs, ch_in, ch_out, L+2p, 1)

    # output = th.matmul(kernel_matrix, input) # (bs, ch_in, ch_out, L+2p, 1)
    # output = th.sum(output, dim=1).reshape(bs, out_ch, out_length+2*pad)  # (bs, ch_out, L+2p)

    #===================
    #------
    # input: (bs, in_ch, in_length)
    input_uf = input.unfold(-1, kernel_size, 1) #  # (bs, in_ch, in_length-ks+1, ks)
    input_uf = input_uf.reshape(bs, in_ch, 1, in_length - kernel_size + 1, kernel_size, 1).tile(1,1,out_ch,1,1,1) # bs, in_ch, out_ch, in_length-ks+1, ks, 1
    input_uf = th.cat((th.tile(input_uf[:,:,:,:1,:,:], (1,1,1,kernel_size-1,1,1)), input_uf, th.tile(input_uf[:,:,:,-1:,:,:], (1,1,1,kernel_size-1,1,1))), dim=3)  # bs, in_ch, out_ch, in_length+ks-1, ks, 1
    #------
    # weight: (bs, in_ch, out_ch, in_length, kernel_size)
    weight = weight.permute(0,1,2,4,3) # (bs, in_ch, out_ch, kernel_size, in_length)
    weight = F.pad(weight, (0, kernel_size-1)) # (bs, in_ch, out_ch, ks, in_length+ks-1)
    idx_gather = (th.arange(in_length+kernel_size-1)[None,:] - th.arange(kernel_size)[:,None]) % (in_length+kernel_size-1) # ks, in_length+ks-1
    idx_gather = idx_gather.reshape(1,1,1,kernel_size,in_length+kernel_size-1).tile(bs, in_ch, out_ch, 1, 1).to(device) #(bs, in_ch, out_ch, ks, in_length+ks-1)
    weight = th.gather(weight, -1, idx_gather)  # (bs, in_ch, out_ch, ks, in_length+ks-1)
    weight=weight.permute(0,1,2,4,3).flip(-1) # (bs, in_ch, out_ch, in_length+ks-1, ks)
    idx_gather = (th.arange(kernel_size)[None,:] - th.cat((th.arange(1,kernel_size),th.zeros(in_length-kernel_size+1),th.arange(1,kernel_size)), dim=0)[:,None]) % kernel_size # in_length+ks-1, ks
    idx_gather = idx_gather.reshape(1,1,1,in_length+kernel_size-1,kernel_size).tile(bs, in_ch, out_ch,1,1).to(device)  #(bs, in_ch, out_ch, in_length+ks-1,ks)
    weight = th.gather(weight, -1, idx_gather.to(int)) # (bs, in_ch, out_ch, in_length+ks-1, ks)
    weight = weight.reshape(bs, in_ch, out_ch,in_length+kernel_size-1, 1,kernel_size)  # (bs, in_ch, out_ch, in_length+ks-1, 1, ks)
    #-------
    output = th.matmul(weight, input_uf) # (bs, ch_in, ch_out, in_length+ks-1=out_length+2p, 1, 1)
    output = th.sum(output, dim=1).reshape(bs, out_ch, out_length+2*pad)
    #========================

    if not pad == 0:
        output = output[:,:,pad:-pad]

    if not bias == None:
        output = output + bias

    return output

class HyperConv_FD(nn.Module):
    def __init__(self, num_aux, ch_in, ch_out, kernel_size=3, ch_hidden=32, num_hidden=1, use_res=True, droprate=0.0, use_f_aux=True, use_bias=True, pad='same', transpose=False):
        super().__init__()
        self.num_aux = num_aux
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel_size = kernel_size
        self.use_f_aux = use_f_aux
        self.use_bias = use_bias
        self.transpose = transpose
        if pad == 'same':
            assert kernel_size % 2 == 1
            self.pad = (kernel_size - 1)//2
        else:
            self.pad = pad

        #=== weight ====
        modules = [
            nn.Linear(num_aux, ch_hidden),
            nn.LayerNorm(ch_hidden),
            nn.ReLU(),
            nn.Dropout(droprate)]
        if not use_res:
            for l in range(num_hidden):
                modules.extend([
                    nn.Linear(ch_hidden, ch_hidden),
                    nn.LayerNorm(ch_hidden),
                    nn.ReLU(),
                    nn.Dropout(droprate)
                ])
        else:
            for l in range(round(num_hidden/2)):
                modules.extend([ResLinearBlock(ch_hidden,droprate=0.0)])
        modules.extend([
             nn.Linear(ch_hidden, ch_out * ch_in * kernel_size)
        ])
        self.weight_layers = nn.Sequential(*modules)

        #=== bias ====
        if use_bias:
            modules = [
                nn.Linear(num_aux, ch_hidden),
                nn.LayerNorm(ch_hidden),
                nn.ReLU(),
                nn.Dropout(droprate)]
            if not use_res:
                for l in range(num_hidden):
                    modules.extend([
                        nn.Linear(ch_hidden, ch_hidden),
                        nn.LayerNorm(ch_hidden),
                        nn.ReLU(),
                        nn.Dropout(droprate)
                    ])
            else:
                for l in range(round(num_hidden/2)):
                    modules.extend([ResLinearBlock(ch_hidden,droprate=0.0)])
            modules.extend([
                nn.Linear(ch_hidden, ch_out)
            ])
            self.bias_layers = nn.Sequential(*modules)
    
    def forward(self, input):
        '''
        input: dict
            x: (2S, B, ch_in, L_in)
            z: (B, L_out, num_aux)
        returns: dict
            x: (2S, B, ch_out, L_o)
            z: (B, L_out, num_aux)
        '''
        # print(f'transpose:{self.transpose}')
        x = input["x"] 
        z = input["z"]
        S = x.shape[0]//2
        L_i = x.shape[-1]
        B, L_o, _ = z.shape

        if self.use_f_aux:
            weight = self.weight_layers(z) # (B, L_o, ch_out * ch_in * kernel_size)
            weight = weight.reshape(B, L_o, self.ch_in, self.ch_out, self.kernel_size)
            weight = weight.permute(0, 2, 3, 1, 4) # (B, ch_in, ch_out, L_o, kernel_size)
            weight = weight.unsqueeze(0).tile(2*S,1,1,1,1,1) # (2S, B, ch_in, ch_out, L_o, kernel_size)
            weight = weight.reshape(2*S*B, self.ch_in, self.ch_out, L_o, self.kernel_size) # (2SB, ch_in, ch_out, L_o, kernel_size)

            if self.use_bias:
                bias = self.bias_layers(z) # (B, L_o, ch_out)
                bias = bias.permute(0,2,1) # (B, ch_out, L_o)
                bias = bias.unsqueeze(0).tile(2*S,1,1,1)
                bias = bias.reshape(2*S*B, self.ch_out, L_o) # (2SB, ch_out, L_o)
        
        else:
            weight = self.weight_layers(z[:,0,:]) # (B, ch_out * ch_in * kernel_size)
            weight = weight.reshape(1, B, self.ch_in, self.ch_out, 1, self.kernel_size).tile(2*S, 1,1, 1, L_o, 1) # (2S, B, ch_in, ch_out, L_o, kernel_size)
            weight = weight.reshape(2*S*B, self.ch_in, self.ch_out, L_o, self.kernel_size) # (2SB, ch_in, ch_out, L_o, kernel_size)

            if self.use_bias:
                bias = self.bias_layers(z[:,0,:]) # (B, ch_out)
                bias = bias.reshape(1, B, self.ch_out, 1).tile(2*S, 1, 1, L_o) # (2S, B, ch_out, L_o)
                bias = bias.reshape(2*S*B, self.ch_out, L_o)
        
        if not self.use_bias:
            bias = None
        
        x = x.reshape(2*S*B, self.ch_in, L_i) # (2SB, ch_in, L_i)
        if not self.transpose:
            output = conv1d_scratch(input=x, weight=weight, bias=bias, pad=self.pad) # (2SB, ch_out, out_length)
        else:
            output = convT1d_scratch(input=x, weight=weight, bias=bias, pad=self.pad) # (2SB, ch_out, out_length)
        output = output.reshape(2*S, B, self.ch_out, L_o)
        returns = {
            "x": output, # (2S, B, ch_out, L_o)
            "z": z,      # (B, L_out, num_aux)
        }
        
        return returns

class HyperConv_FD_Block(nn.Module):
    def __init__(self, num_aux, ch_in, ch_out, kernel_size=3, ch_hidden=32, num_hidden=1, use_res=True, droprate=0.0, use_f_aux=True, use_bias=True, pad='same',transpose=False, post_prcs=True):
        super().__init__()
        self.hyperconv = HyperConv_FD(num_aux=num_aux, ch_in=ch_in, ch_out=ch_out, kernel_size=kernel_size, ch_hidden=ch_hidden, num_hidden=num_hidden, use_res=use_res, droprate=droprate, use_f_aux=use_f_aux, use_bias=use_bias, pad=pad, transpose=transpose)
        if post_prcs:
            self.layers_post = nn.Sequential(
                nn.LayerNorm(ch_out),
                nn.ReLU(),
                nn.Dropout(droprate)
            )
        else:
            self.layers_post = nn.Sequential(
                nn.Identity()
            )
    
    def forward(self, input):
        xz = self.hyperconv(input)

        returns = {}
        output = xz["x"] # (2S, B, ch_out, L_o)
        output = self.layers_post(output.permute(0,1,3,2)) # (2S, B, L_o, ch_out)
        output = output.permute(0,1,3,2) # (2S, B, ch_out, L_o)
        returns["x"] = output
        returns["z"] = input["z"]

        return returns

class HRTFApproxNetwork(Net):
    def __init__(self,
                 config,
                 model_name='hrtf_approx_network',
                 use_res=True,
                 use_cuda=True):
        super().__init__(model_name, use_cuda)
        self.config = config
        self.max_f = config["max_frequency"]
        self.maxto = config["max_truncation_order"]
        self.fft_length = config["fft_length"]
        self.filter_length = round(self.fft_length/2)# + 1)
        self.f_arr = (np.linspace(0, self.max_f, self.filter_length+1))[1:] # frequency bin, (filterlengh,)
        self.N_arr = th.from_numpy(calcMaxOrder(f=self.f_arr,maxto=self.maxto).astype(np.float32)).clone()
        self.use_freq_as_input = config["use_freq_as_input"]
        if self.use_freq_as_input:
            self.f_in = th.from_numpy(self.f_arr.astype(np.float32)).clone()
            
            self.out_channel = round(2*2*((self.maxto + 1)**2))

            self.channel = config["channel"]
            layers = config["layers"]
            droprate = config["droprate"]

            modules = []
            modules.extend([nn.Conv1d(1,self.channel,kernel_size=1),
                                nn.ReLU(), 
                                nn.Dropout(droprate)])
            self.use_res = use_res
            if self.use_res:
                for ll in range(round(layers/2)):
                    modules.extend([ResConvSingleKernelBlock(channel=self.channel, droprate = droprate, filter_length=self.filter_length)])
            else:
                for l in range(layers):
                    modules.extend([
                                    nn.Conv1d(self.channel,self.channel,kernel_size=1),
                                    # nn.Conv1d(self.channel,self.channel,kernel_size=8,padding='same'),
                                    nn.ReLU(),
                                    nn.Dropout(droprate)])
            modules.append(nn.Conv1d(self.channel,self.out_channel,kernel_size=1))

            self.layers = nn.Sequential(*modules)
        else:
            self.N_arr_np = calcMaxOrder(f=self.f_arr,maxto=self.maxto)
            self.output_size =  round(2*2*np.sum((self.N_arr_np + 1)**2))
            self.layers = nn.Sequential(
                                        nn.Linear(1,self.output_size,bias=False),
                                        )

        # for layer in self.layers:
        #     if hasattr(layer, 'weight'):
        #         nn.init.normal_(layer.weight, 0.0, 1e-3)
        #     if hasattr(layer, 'bias'):
        #         nn.init.normal_(layer.bias, 0.0, 1e-3)
    def forward(self, pos, use_cuda_forward, use_lininv=False, hrtf=None, use_coeff=False,coeff=None):
        '''
        :param pos: the input as a (Bx) 3 tensor ([r,phi,theta])
        :return: out: HRTF produce by the network (Bx)2xfilter_length
        '''
        r, phi, theta = pos[:,0], pos[:,1], pos[:,2]
        # print(f"pos:{pos.shape}") # pos:torch.Size([440, 3])
        SpF = SpecialFunc(r=r, phi=phi, theta=theta, maxto=self.maxto, fft_length=self.fft_length, max_f=self.max_f)
        n_vec, m_vec = SpF.n_vec, SpF.m_vec
        Hank = SpF.SphHankel()
        Hank = th.complex(th.nan_to_num(th.real(Hank)), th.nan_to_num(th.imag(Hank)))
        Harm = SpF.SphHarm()
        if self.use_freq_as_input:
            f_in = th.tile(self.f_in.view(1,1,-1), (pos.shape[0],1,1))
        returns = {}
        if use_cuda_forward:
            Hank = Hank.cuda()
            Harm = Harm.cuda()
            if self.use_freq_as_input:
                f_in = f_in.cuda()
        if use_coeff:
            Coeff = coeff
        elif use_lininv:
            if self.config["underdet"] or self.config["balanced"]:
                t_des = self.config["t_des"]
                vec_cart_gt = sph2cart(pos[:,1],pos[:,2],pos[:,0])
                idx_t_des = aprox_t_des(pts=vec_cart_gt/1.47, t=t_des, plot=False)
                pos_t = pos[idx_t_des,:]
                hrtf_t = hrtf[idx_t_des,:,:]
                self.maxto = self.config["max_truncation_order"] if self.config["underdet"] else t_des
                self.N_arr = th.from_numpy(calcMaxOrder(f=self.f_arr,maxto=self.maxto).astype(np.float32)).clone()
                SpF_t = SpecialFunc(r=pos_t[:,0], phi=pos_t[:,1], theta=pos_t[:,2], maxto=self.maxto, fft_length=self.fft_length, max_f=self.max_f)
                n_vec  = SpF_t.n_vec
                Hank_t = SpF_t.SphHankel()
                Hank_t = th.complex(th.nan_to_num(th.real(Hank_t)), th.nan_to_num(th.imag(Hank_t)))
                Harm_t = SpF_t.SphHarm()
            else:
                pos_t = pos
                hrtf_t = hrtf
                Hank_t = Hank
                Harm_t = Harm
            B = pos.shape[0]
            B_t = pos_t.shape[0]
            Coeff = th.zeros((self.maxto+1)**2, 2, SpF.filter_length, dtype = th.complex64)
            for f_bin in range(SpF.filter_length):
                N = (self.N_arr[f_bin].to('cpu').detach().numpy().copy()).astype(np.int32)
                n_vec,_ = sph_harm_nmvec(N)
                num_base = (N+1)**2
                mat_base = th.zeros(B_t, num_base, dtype = th.complex64) # Phi
                for i,n in enumerate(n_vec):
                    mat_base[:,i] = Hank_t[:,n,f_bin] * Harm_t[:,i]
                h = hrtf_t[:,:,f_bin] # B, ch=2 (Left,Right)
                mat_base_H = th.conj(th.transpose(mat_base,0,1)) # Phi^H
                mat_base_MPI = th.matmul(mat_base_H, mat_base) # Phi^+ = Phi^H * Phi
                # regularization matrix in Duraiswaini+04
                # D=(1+n(n+1))I
                D = th.diag(1+th.from_numpy((n_vec*(n_vec+1)).astype(np.float32)).clone())
                mat_reg = self.config["reg_w"] * D
                c = th.linalg.solve(mat_base_MPI + mat_reg ,th.matmul(mat_base_H, h)) # solve (Phi^+) c = Phi^H h
                # print(c.shape) # 16,2
                pad = (self.maxto+1)**2 - c.shape[0]
                c = F.pad(c,(0,0,0,pad),"constant",0)
                # print(c.shape) # 1296,2
                # print(c[15:20,0]) # padding OK

                Coeff[:,:,f_bin] = c 
            # (maxto+1)**2,2,filter_length -> 2,(maxto+1)**2,filter_length
            Coeff = Coeff.contiguous().permute(1,0,2) 
            # returns['coeff'] = Coeff
            # 2,(maxto+1)**2,filter_length -> B, 2,(maxto+1)**2,filter_length
            Coeff = th.tile(Coeff,(B,1,1,1))
            # print(Coeff.shape)
        elif self.use_freq_as_input:
            # print(f_in.shape) # torch.Size([512, 1, 64])
            # print(f_in[:2,0,:5]) # tensor([[ 250.,  500.,  750., 1000., 1250.],
            # [ 250.,  500.,  750., 1000., 1250.]], device='cuda:0')
            # if th.any(th.isnan(Hank)):
            #     print(f"NaN in Hank") # > NaN
            #     sys.exit()   
            # if th.any(th.isnan(f_in)):
            #     print(f"NaN in f_in") # > NaN
            #     sys.exit()   
            # Coeff = self.layers(f_in/(self.max_f/2)-1) # [-1,1] # []B, 2*2*(maxto+1)^2, filterlength
            Coeff = self.layers(f_in/(self.max_f)) # [0,1] # []B, 2*2*(maxto+1)^2, filterlength
            # if th.any(th.isnan(Coeff)):
            #     print(f"NaN in Coeff_1") # > NaN
            #     sys.exit()   
            Coeff = th.complex(Coeff[:,:2*(self.maxto+1)**2,:], Coeff[:,2*(self.maxto+1)**2:,:])     
            # if th.any(th.isnan(Coeff)):
            #     print(f"NaN in Coeff2") # > NaN
            #     sys.exit()   
            Coeff = Coeff.contiguous().view(Coeff.shape[0], 2, -1, self.filter_length)
            # if th.any(th.isnan(Coeff)):
            #     print(f"NaN in Coeff3") # > NaN
            #     sys.exit()
            # print(Coeff.shape) # B,2,(maxto+1)**2,filter_length # torch.Size([512, 2, 1296, 64])
        else:
            B = pos.shape[0]
            input_one = th.tile(th.tensor([1.0]),(B,1))
            # print(input_one.dtype) # torch.float32
            if use_cuda_forward:
                input_one = input_one.cuda()
            Coeff = self.layers(input_one)
            Coeff = transformCoeff(Coeff, n_vec, self.N_arr, self.maxto, self.filter_length)
        returns['coeff'] = Coeff
        out = th.zeros(Coeff.shape[0], 2, SpF.filter_length, dtype = th.complex64)
        if use_cuda_forward:
            out = out.cuda()
            # print(Hank[0,:,0])
        for i,n in enumerate(n_vec):
            mask = self.N_arr >= n
            if use_cuda_forward:
                mask = mask.cuda()
            Coeff_mn = Coeff[:,:,i,:] #* mask[None,None,:] # masking
            Hank_n = Hank[:,n,:] * mask[None,:] # masking
            # print(Hank_n[0,:1])
            # if i == 1000:
            #     print(n)
            #     print(f"coeff(models):{Coeff_mn[0,0,:50]}")
            #     print(f"Hank_n:{Hank_n[0,:50]}")
            #     sys.exit()

            # print(n)
            # print(self.N_arr)
            # print(self.N_arr >= n)
            # print(Hank_n[0,:])
            # print(th.max(th.abs(Hank_n)))
            Harm_mn = Harm[:,i]
            delta = th.mul(Coeff_mn, Hank_n[:,None]) * Harm_mn[:,None,None]
            out += delta

            if th.any(th.isnan(Coeff_mn)):
                print(f"NaN in Coeff_mn") # > NaN
                sys.exit()
            if th.any(th.isnan(Hank_n)):
                print(f"NaN in Hank_n") # > NaN
                print(Hank[:,n,:])
                print(Hank_n)
                print(mask)
                print(i)
                print(n)
                sys.exit()
            if th.any(th.isnan(Harm_mn)):
                print(f"NaN in Hanrm_mn") # > NaN
                sys.exit()
            # print(Coeff_mn.shape)
            # print(Hank_n.shape)  # B, filterlength
            # print(Hank_n[0,0])
            # print(th.mul(Coeff_mn, Hank_n[:,None]).shape) # B,2,filter_length
            if th.any(th.isnan(th.mul(Coeff_mn, Hank_n[:,None]))):
                print("NaN")
                sys.exit()            
            # print(th.max(th.abs(out)))

        if th.any(th.isnan(out)):
                print(f"NaN in output") # > NaN
                sys.exit()
        returns['output'] = out
        return returns

def initwb(layers,mean_w=0.0,std_w=1e-2,mean_b=0.0,std_b=1e-1):
    for layer in layers:
        if hasattr(layer, 'weight'):
            nn.init.normal_(layer.weight, mean_w, std_w)
        if hasattr(layer, 'bias'):
            nn.init.normal_(layer.bias, mean_b, std_b)

class View(nn.Module):
    def __init__(self, shape):
        super().__init__(self)
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, normalization='BN', size=None, hlayers=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        modules = []
        modules.extend([nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)])
        if normalization == 'BN':
            modules.extend([nn.BatchNorm2d(mid_channels)])
        elif normalization == 'LN':
            modules.extend([nn.LayerNorm(size)])
        for l in range(hlayers):
            modules.extend([
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            ])
            if normalization == 'BN':
                modules.extend([nn.BatchNorm2d(mid_channels)])
            elif normalization == 'LN':
                modules.extend([nn.LayerNorm(size)])
        modules.extend([
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        ])
        if normalization == 'BN':
            modules.extend([nn.BatchNorm2d(mid_channels)])
        elif normalization == 'LN':
            modules.extend([nn.LayerNorm(size)])
        modules.extend([nn.ReLU(inplace=True)])
        self.double_conv = nn.Sequential(*modules)
        
        #=============== 
        # https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py 
        #===============
        # self.double_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, normalization='LN', size=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, normalization=normalization, size=size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, normalization='LN', size=None, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels//2, normalization=normalization, size=size)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, normalization=normalization, size=size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print(x1.shape)
        # print(x2.shape)
        # input is BCHW
        diffH = x1.size()[2] - x2.size()[2]

        x2 = F.pad(x2, [0, 0,
                        0, diffH])
        # print(x2.shape) # same as x1
        # print(x2[0,0,:5,0]) # not zeros
        # print(x2[0,0,-5:,0]) # [0,0,0,0,0]
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = th.cat([x2, x1], dim=1)
        # print(x.shape)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, config, bilinear=False):
        super(UNet, self).__init__()
        self.config = config

        #== Encoder ===
        if self.config["in_magphase"]:
            ch_init = 2*3
        elif self.config["in_mag"]:
            ch_init = 2
        else:
            ch_init = 2*2

        num_layers_en = 4

        ch_en = config["channel_En_z"]
        size = [config["num_pts"], config["fft_length"] // 2]

        self.inc = DoubleConv(ch_init, ch_en, normalization='LN', size=size)

        size = [v // 2 for v in size] # size //= 2
        ch_en *= 2
        self.down1 = Down(ch_en//2, ch_en, normalization='LN', size=size)

        size = [v // 2 for v in size] # size //= 2
        ch_en *= 2
        self.down2 = Down(ch_en//2, ch_en, normalization='LN', size=size)

        size = [v // 2 for v in size] # size //= 2
        ch_en *= 2
        self.down3 = Down(ch_en//2, ch_en, normalization='LN', size=size)

        size = [v // 2 for v in size] # size //= 2
        ch_en *= 2
        self.down4 = Down(ch_en//2, ch_en, normalization='LN', size=size)

        #================
        self.ch_bn = ch_en
        self.H_in = round(np.floor(config["num_pts"]/2**num_layers_en))
        self.W_in = round(np.floor(config["fft_length"] // 2 /2**num_layers_en))
        in_size = self.ch_bn * self.H_in * self.W_in
        self.H_out = round(np.floor((config["max_truncation_order"]+1)**2/2**num_layers_en))
        out_size = self.ch_bn * self.H_out *self.W_in
        modules = []
        modules.extend([
                        nn.Linear(in_size,out_size),
                        nn.LayerNorm(out_size),
                        nn.ReLU()])
        self.MLP = nn.Sequential(*modules)
        #================
        ch_de = ch_en
        size = [self.H_out, self.W_in]
        size = [v * 2 for v in size] # size *= 2
        factor = 2 if bilinear else 1
        self.up4 = Up(ch_de, ch_de // 2 // factor, normalization='LN', size=size, bilinear=bilinear)

        ch_de //= 2
        size = [v * 2 for v in size] # size *= 2
        self.up3 = Up(ch_de, ch_de // 2 // factor, normalization='LN', size=size, bilinear=bilinear)

        ch_de //= 2
        size = [v * 2 for v in size] # size *= 2
        self.up2 = Up(ch_de, ch_de // 2 // factor, normalization='LN', size=size, bilinear=bilinear)

        ch_de //= 2
        size = [v * 2 for v in size] # size *= 2
        self.up1 = Up(ch_de, ch_de // 2, normalization='LN', size=size, bilinear=bilinear)

        self.outc = nn.Conv2d(ch_de//2, 6, kernel_size=1)

    def forward(self, x):
        # returns = {}
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x  = self.down4(x4)
        # print(x.shape) # orch.Size([20, 64, 8, 16])

        B = x.shape[0]
        ch = x.shape[1]
        x = x.view(B,-1)

        x = self.MLP(x)
        # print(x.shape) # torch.Size([20, 51200])
        x = x.view(B,ch,self.H_out,self.W_in)
        # print(x.shape) # torch.Size([20, 64, 50, 16])

        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        out = self.outc(x)
        return out

class ComplexLinear(nn.Module):
    def __init__(self, in_feartures, out_feartures, bias=True):
        super().__init__()
        self.in_feartures  = in_feartures
        self.out_feartures = out_feartures
        self.bias = bias

        self.weight_re = nn.Parameter(th.FloatTensor(self.out_feartures, self.in_feartures))
        self.weight_im = nn.Parameter(th.FloatTensor(self.out_feartures, self.in_feartures))
        if self.bias:
            self.bias_re = nn.Parameter(th.FloatTensor(self.out_feartures))
            self.bias_im = nn.Parameter(th.FloatTensor(self.out_feartures))

    def forward(self, input):
        # [args:    dict input]  input["re"], input["im"] (*, in_feartures)
        # [returns: dict output] output["re"], output["im"] (*, out_feartures)
        x_re = input["re"]
        x_im = input["im"] 
        output = {}
        if self.bias:
            output["re"] = F.linear(x_re, self.weight_re, self.bias_re) -  F.linear(x_im, self.weight_im)
            output["im"] = F.linear(x_re, self.weight_im, self.bias_im) +  F.linear(x_im, self.weight_re)
        else:
            output["re"] = F.linear(x_re, self.weight_re) -  F.linear(x_im, self.weight_im)
            output["im"] = F.linear(x_re, self.weight_im) +  F.linear(x_im, self.weight_re)
        return output

class ComplexReLU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        # [args:    dict input]  input["re"], input["im"] (*, in_feartures)
        # [returns: dict output] output["re"], output["im"] (*, out_feartures)
        # for x in C, output = x if arg(x) in [0,pi/2] else 0
        # N. Guberman, “On complex valued convolutional neural networks,”arXiv preprint arXiv:1602.09046, 2016
        x_re = input["re"]
        x_im = input["im"] 

        phi = th.atan2(x_im, x_re)
        mask1 = (phi >= 0) #and phi <= np.pi/2)
        mask2 = (phi <= np.pi/2)
        mask = mask1 * mask2
        mask = mask.to(x_re.device)
        output = {}
        output["re"] = x_re * mask
        output["im"] = x_im * mask
        return output

class ComplexHyperLinear(nn.Module):
    def __init__(self, ch_in, ch_out, input_size=3, ch_hidden=32, num_hidden=1, use_res=True):
        super().__init__()
        self.ch_in  = ch_in
        self.ch_out = ch_out

        #==== weight_re ======
        modules = [
            nn.Linear(input_size, ch_hidden),
            nn.LayerNorm(ch_hidden),
            nn.ReLU()]
        if not use_res:
            for l in range(num_hidden):
                modules.extend([
                    nn.Linear(ch_hidden, ch_hidden),
                    nn.LayerNorm(ch_hidden),
                    nn.ReLU()
                ])
        else:
            for l in range(round(num_hidden/2)):
                modules.extend([ResLinearBlock(ch_hidden,droprate=0.0)])
        modules.extend([
             nn.Linear(ch_hidden, ch_out * ch_in)
        ])
        self.weight_re_layers = nn.Sequential(*modules)

        #==== weight_im ======
        modules = [
            nn.Linear(input_size, ch_hidden),
            nn.LayerNorm(ch_hidden),
            nn.ReLU()]
        if not use_res:
            for l in range(num_hidden):
                modules.extend([
                    nn.Linear(ch_hidden, ch_hidden),
                    nn.LayerNorm(ch_hidden),
                    nn.ReLU()
                ])
        else:
            for l in range(round(num_hidden/2)):
                modules.extend([ResLinearBlock(ch_hidden,droprate=0.0)])
        modules.extend([
             nn.Linear(ch_hidden, ch_out * ch_in)
        ])
        self.weight_im_layers = nn.Sequential(*modules)

        #==== bias_re ======
        modules = [
            nn.Linear(input_size, ch_hidden),
            nn.LayerNorm(ch_hidden),
            nn.ReLU()]
        if not use_res:
            for l in range(num_hidden):
                modules.extend([
                    nn.Linear(ch_hidden, ch_hidden),
                    nn.LayerNorm(ch_hidden),
                    nn.ReLU()
                ])
        else:
            for l in range(round(num_hidden/2)):
                modules.extend([ResLinearBlock(ch_hidden,droprate=0.0)])
        modules.extend([
             nn.Linear(ch_hidden, ch_out)
        ])
        self.bias_re_layers = nn.Sequential(*modules)

        #==== bias_re ======
        modules = [
            nn.Linear(input_size, ch_hidden),
            nn.LayerNorm(ch_hidden),
            nn.ReLU()]
        if not use_res:
            for l in range(num_hidden):
                modules.extend([
                    nn.Linear(ch_hidden, ch_hidden),
                    nn.LayerNorm(ch_hidden),
                    nn.ReLU()
                ])
        else:
            for l in range(round(num_hidden/2)):
                modules.extend([ResLinearBlock(ch_hidden,droprate=0.0)])
        modules.extend([
             nn.Linear(ch_hidden, ch_out)
        ])
        self.bias_im_layers = nn.Sequential(*modules)

    def forward(self, input):
        x_re = input["re"] # (B,P, ch_in)
        x_im = input["im"] # (B,P, ch_in)
        z = input["z"] # (B,P, input_size=3)
        B = x_re.shape[0] # batch size
        P = x_re.shape[1] # num_pts
        # print(x_re.shape)
        # print(z.shape)
        weight_re = self.weight_re_layers(z) # (B,P,ch_out * ch_in)
        weight_re = weight_re.view(B, P, self.ch_out, self.ch_in) # (B, P, ch_out, ch_in)
        weight_im = self.weight_im_layers(z) # (B,P,ch_out * ch_in)
        weight_im = weight_im.view(B, P, self.ch_out, self.ch_in) # (B, P, ch_out, ch_in)
        bias_re = self.bias_re_layers(z) # (B,P, ch_out)
        bias_im = self.bias_im_layers(z) # (B,P, ch_out)

        output = {}
        output["re"] = th.matmul(weight_re, x_re.view(B,P,-1,1)).view(B,P,-1) - th.matmul(weight_im, x_im.view(B,P,-1,1)).view(B,P,-1) + bias_re
        output["im"] = th.matmul(weight_re, x_im.view(B,P,-1,1)).view(B,P,-1) + th.matmul(weight_im, x_re.view(B,P,-1,1)).view(B,P,-1) + bias_im
        output["z"] = z
        
        return output

class ComplexHyperLinearBlock(nn.Module):
    def __init__(self, ch_in, ch_out, ch_hidden=32, num_hidden=1, use_res=True, input_size=3):
        super().__init__()
        self.complexhyperlinear = ComplexHyperLinear(ch_in, ch_out, ch_hidden=ch_hidden, num_hidden=num_hidden, use_res=use_res, input_size=input_size)
        # self.LN = nn.LayerNorm(ch_out)
        self.complexrelu = ComplexReLU()

    def forward(self, input):
        med = self.complexhyperlinear(input) # re,im,z
        output = self.complexrelu(med) # re, im
        output["z"] = input["z"]
        
        return output

class GCU(nn.Module):
    '''
        Noel, M. M., Trivedi, A., & Dutta, P. (2021). Growing cosine unit: A novel oscillatory activation function that can speedup training and reduce parameters in convolutional neural networks. arXiv preprint arXiv:2108.12943.
    '''
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return input * th.cos(input)

class FourierFeatureMapping(nn.Module):
    def __init__(self, num_features, dim_data, trainable=True):
        super(FourierFeatureMapping, self).__init__()
        self.num_features = num_features
        self.dim_data = dim_data
        multinormdist = MultivariateNormal(th.zeros(self.dim_data), th.eye(self.dim_data))
        self.v = multinormdist.sample(sample_shape=th.Size([self.num_features]))  # self.num_features, dim_data
        if trainable:
            self.v = nn.Parameter(self.v)
            # ic(self.v.device)
        else:
            self.v = self.v.cuda()
    
    def forward(self,x):
        # x: [...,dim_data]
        x_shape = list(x.shape)
        x_shape[-1] = self.num_features
        # print(x_shape)
        x = x.reshape(-1, self.dim_data).permute(1,0)  # d, SB
        # print(x.shape)
        # print(self.v.shape)
        vx = th.matmul(self.v.to(x.device),x) # R, SB
        # print(vx.shape)
        vx = vx.permute(1,0).reshape(x_shape) # R,SB->SB,R->S,B,R
        fourierfeatures = th.cat((th.sin(2*np.pi*vx), th.cos(2*np.pi*vx)), dim=-1) # S,B,2R
        
        return fourierfeatures

class HRTFApproxNetwork_AE(Net):
    def __init__(self,
                config,
                model_name='hrtf_approx_network',
                use_cuda=True,
                c_std=None,
                c_mean=None):
        super().__init__(model_name, use_cuda)
        self.config = config
        self.max_f = config["max_frequency"]
        self.maxto = config["max_truncation_order"]
        self.fft_length = config["fft_length"]
        self.filter_length = round(self.fft_length/2)
        self.f_arr = (np.linspace(0, self.max_f, self.filter_length+1))[1:] # frequency bin, (filterlengh,)
        self.N_arr_np = calcMaxOrder(f=self.f_arr,maxto=self.maxto)
        
        # self.coeff_size = round(2*3*np.sum((self.N_arr_np + 1)**2))
        self.coeff_size = round(2*3*np.sum((self.N_arr_np + 1)**2)) if config["out_magphase"] == True else round(2*2*np.sum((self.N_arr_np + 1)**2))
        self.c_std = c_std
        self.c_mean = c_mean
        if config["num_pts"] < 440:
            self.to_in = round(config["num_pts"]**0.5 - 1)
        else:
            self.to_in = 19

        if self.config["use_nf"]:
            #======== post activation ver.========
            modules = []
            modules.extend([
                scale(1.0),
                WSLinear(2*3*self.filter_length+3, config["channel_En_z"]),
                ScaledMish(), 
                nn.Dropout(config["droprate"])])
            for l in range(round(config["hlayers_En_z"]/2)):
                modules.extend([ResWSLinearBlock(config["channel_En_z"], config["droprate"], config["alpha"])])
            modules.extend([WSLinear(config["channel_En_z"],config["dim_z"]),
                            ScaledMish()])
            self.Encoder_z = nn.Sequential(*modules)

            modules = []
            modules.extend([scale(1.0),
                            # WSLinear(config["dim_z"], 
                            WSLinear(config["dim_z"]+16, config["channel_De_z"]),
                            ScaledMish(), 
                            nn.Dropout(config["droprate"])])
            for l in range(round(config["hlayers_De_z"]/2)):
                modules.extend([ResWSLinearBlock(config["channel_De_z"], config["droprate"], config["alpha"])])
            # modules.extend([WSLinear(config["channel_De_z"], self.coeff_size)])
            modules.extend([nn.Linear(config["channel_De_z"], self.coeff_size)])
            self.Decoder_z = nn.Sequential(*modules)
            #=======================================

            # #======== pre activation ver.=========
            # modules = []
            # modules.extend([
            #     scale(1.25),
            #     WSLinear(2*3*self.filter_length+3, config["channel_En_z"]),
            #     nn.Dropout(config["droprate"])])
            # for l in range(round(config["hlayers_En_z"]/2)):
            #     beta = 1.0 if l == 0 else 1 + config["alpha"]**2
            #     modules.extend([ResWSLinearBlock2(config["channel_En_z"], config["droprate"], config["alpha"], beta)])
            #     if l == round(config["hlayers_En_z"]/2)-1:
            #         modules.extend([ScaledMish(), scale(1/beta)])
            # modules.extend([WSLinear(config["channel_En_z"],config["dim_z"]),
            #                 ScaledMish()])
            # self.Encoder_z = nn.Sequential(*modules)

            # modules = []
            # modules.extend([scale(3),
            #                 WSLinear(config["dim_z"], config["channel_De_z"]),
            #                 nn.Dropout(config["droprate"])])
            # for l in range(round(config["hlayers_De_z"]/2)):
            #     beta = 1.0 if l == 0 else 1 + config["alpha"]**2
            #     modules.extend([ResWSLinearBlock2(config["channel_En_z"], config["droprate"], config["alpha"], beta)])
            #     if l == round(config["hlayers_De_z"]/2)-1:
            #         modules.extend([ScaledMish(), scale(1/beta)])
            # modules.extend([WSLinear(config["channel_De_z"], self.coeff_size)])
            # self.Decoder_z = nn.Sequential(*modules)
            # #=======================================
        elif self.config["use_hypernet"]:
            modules = []
            if self.config["in_magphase"]:
                insize = 2*3*self.filter_length
            elif self.config["in_mag"]:
                if self.config['in_mag_pc']:
                    insize = 2*1*self.config["num_pc"]
                elif self.config["in_latent"]:
                    insize = 2*1*self.config["num_latents"]
                else:
                    insize = 2*1*self.filter_length
            else:
                insize = 2*2*self.filter_length
            if self.config["use_lr_aug"]:
                insize = round(insize/2)
            modules.extend([
                HyperLinearBlock(ch_in=insize, ch_out=config["channel_En_z"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"]),
                ])
            if self.config["En_use_res"]:
                for l in range(round(config["hlayers_En_z"]/2)):
                    modules.extend([ResHyperLinearBlock(channel=config["channel_En_z"], ch_hyper_hidden=config["channel_hyper"], num_hyper_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"])])
            else:
                for l in range(config["hlayers_En_z"]):
                    modules.extend([
                    HyperLinearBlock(ch_in=config["channel_En_z"], ch_out=config["channel_En_z"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"]),
                ])
            dimz_enout = round(config["dim_z"]/self.config["num_classifier"])
            modules.extend([
                HyperLinearBlock(ch_in=config["channel_En_z"], ch_out=dimz_enout, ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"])
                ])
            if self.config["Encoder_add_fc"]:
                modules.extend([
                    HyperLinear(ch_in=dimz_enout, ch_out=dimz_enout, ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], use_res=config["hyper_use_res"]),
                    ])
            self.Encoder_z = nn.Sequential(*modules)
            
            if self.config["use_metric"]:
                if self.config["num_classifier"]==1:
                    modules = [ArcMarginProduct(in_features=config["dim_z_af"], out_features=config["num_cluster"],s=config["metric_scale"],m=config['metric_margin'])]
                    self.Classifier_z = nn.Sequential(*modules)
                else:
                    for i in range(self.config["num_classifier"]):
                        modules = [ArcMarginProduct(in_features=config["dim_z_af"], out_features=config['num_sub'],s=config["metric_scale"],m=config['metric_margin'])]
                        exec(f'self.Classifier_z_{i} = nn.Sequential(*modules)')
            
            if self.config["use_attention"]:
                if self.config["pos2weight"]:
                    modules = [Pos2Weight(config=self.config)]
                else:
                    modules = [Attention()]
                
                self.attention = nn.Sequential(*modules)

            if self.config["Decoder"] == "coeff":
                modules = []
                modules.extend([nn.Linear(config["dim_z"]*config["num_classifier"], config["channel_De_z"])])
                modules.extend([nn.LayerNorm(config["channel_De_z"])])
                modules.extend([nn.ReLU(), 
                                nn.Dropout(config["droprate"])])
                if self.config["De_use_res"]:
                    for l in range(round(config["hlayers_De_z"]/2)):
                        modules.extend([ResLinearBlock(config["channel_De_z"], config["droprate"], 'ln')])
                else:
                    for l in range(config["hlayers_De_z"]):
                        modules.extend([nn.Linear(config["channel_De_z"], config["channel_De_z"]),
                        nn.LayerNorm(config["channel_De_z"]),
                        nn.ReLU(),
                        nn.Dropout(config["droprate"])])
                linear = nn.Linear(config["channel_De_z"], self.coeff_size)
                if self.config["coeff_skipconnection"]:
                    nn.init.constant_(linear.weight, 0.0)
                modules.extend([linear])
                self.Decoder_z = nn.Sequential(*modules)

                if self.config["out_magphase"] == False:
                    modules = [nn.Tanh()]
                    self.out_z = nn.Sequential(*modules)
            elif self.config["Decoder"] == "hrtf_hyper":
                modules = []
                if self.config["out_magphase"]:
                    insize = 2*3*self.filter_length
                elif self.config["out_mag"]:
                    if self.config["out_mag_pc"]: # Principal Component Analysis
                        insize = 2*1*self.config["num_pc"]
                        self.pd_train = th.load("pd_train_64.pt").cuda()
                    elif self.config["out_latent"]:
                        insize = 2*1*self.config["num_latents"]
                    else:
                        insize = 2*1*self.filter_length
                else:
                    insize = 2*2*self.filter_length
                if self.config["use_lr_aug"]:
                    insize = round(insize/2)
                modules.extend([
                    HyperLinearBlock(ch_in=config["dim_z"], ch_out=config["channel_De_z"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"]),
                    ])
                if self.config["De_use_res"]:
                    for l in range(round(config["hlayers_De_z"]/2)):
                        modules.extend([ResHyperLinearBlock(channel=config["channel_De_z"], ch_hyper_hidden=config["channel_hyper"], num_hyper_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"])])
                else:
                    for l in range(config["hlayers_De_z"]):
                        modules.extend([
                        HyperLinearBlock(ch_in=config["channel_De_z"], ch_out=config["channel_De_z"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"]),
                    ])
                modules.extend([
                    HyperLinear(ch_in=config["channel_De_z"], ch_out=insize, ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], use_res=config["hyper_use_res"])
                    ])
                self.Decoder_z = nn.Sequential(*modules)
            elif self.config["Decoder"] == "hrtf_mlp":
                modules = []
                if self.config["out_magphase"]:
                    insize = 2*3*self.filter_length
                elif self.config["out_mag"]:
                    insize = 2*1*self.filter_length
                else:
                    insize = 2*2*self.filter_length
                modules.extend([
                    nn.Linear(in_features=config["dim_z"]+3, out_features=config["channel_De_z"]),
                    nn.LayerNorm(config["channel_De_z"]),
                    nn.ReLU(),
                    nn.Dropout(config["droprate"])
                ])
                if self.config["De_use_res"]:
                    for l in range(round(config["hlayers_De_z"]/2)):
                        modules.extend([ResLinearBlock(config["channel_De_z"], config["droprate"], config["normalize"], config["num_pts"])])
                else:
                    for l in range(config["hlayers_De_z"]):
                        modules.extend([
                            nn.Linear(in_features=config["channel_De_z"], out_features=config["channel_De_z"]),
                            nn.LayerNorm(config["channel_De_z"]),
                            nn.ReLU(),
                            nn.Dropout(config["droprate"])
                        ])
                modules.extend([
                    nn.Linear(in_features=config["channel_De_z"], out_features=insize)
                    ])
                self.Decoder_z = nn.Sequential(*modules)
            elif self.config["Decoder"] == "coeff_hyper":
                modules = []
                if self.config["out_magphase"]:
                    outsize = 2*3*self.filter_length
                elif self.config["in_mag"]:
                    outsize = 2*1*self.filter_length
                else:
                    outsize = 2*2*self.filter_length
                modules.extend([
                    HyperLinearBlock(ch_in=config["dim_z"], ch_out=config["channel_De_z"], ch_hidden=config["channel_hyper_De"], num_hidden=config["hlayers_hyper_De"], droprate=config["droprate"], use_res=config["hyper_use_res"], input_size=2),
                    ])
                if self.config["De_use_res"]:
                    for l in range(round(config["hlayers_De_z"]/2)):
                        modules.extend([ResHyperLinearBlock(channel=config["channel_De_z"], ch_hyper_hidden=config["channel_hyper_De"], num_hyper_hidden=config["hlayers_hyper_De"], droprate=config["droprate"], use_res=config["hyper_use_res"], input_size=2)])
                else:
                    for l in range(config["hlayers_De_z"]):
                        modules.extend([
                        HyperLinearBlock(ch_in=config["channel_De_z"], ch_out=config["channel_De_z"], ch_hidden=config["channel_hyper_De"], num_hidden=config["hlayers_hyper_De"], droprate=config["droprate"], use_res=config["hyper_use_res"], input_size=2),
                    ])
                modules.extend([
                    HyperLinear(ch_in=config["channel_De_z"], ch_out=outsize, ch_hidden=config["channel_hyper_De"], num_hidden=config["hlayers_hyper_De"], use_res=config["hyper_use_res"], input_size=2)
                    ])
                self.Decoder_z = nn.Sequential(*modules)

                modules = []
                modules.extend([nn.LayerNorm(self.filter_length),nn.Tanh()])
                # modules.extend([nn.Sigmoid()])
                self.out_z = nn.Sequential(*modules)
            else:
                print("[Error] config['Decoder'] must be one of 'coeff', 'coeff_hyper', 'hrtf_mlp', 'hrtf_hyper'.")
                sys.exit()
        elif self.config["use_complexhypernet"]:
            modules = []
            insize = 2*self.filter_length
            modules.extend([
                ComplexHyperLinearBlock(ch_in=insize, ch_out=config["channel_En_z"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], use_res=config["hyper_use_res"]),
                ])
            for l in range(config["hlayers_En_z"]):
                modules.extend([
                ComplexHyperLinearBlock(ch_in=config["channel_En_z"], ch_out=config["channel_En_z"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], use_res=config["hyper_use_res"]),
            ])
            modules.extend([
                ComplexHyperLinearBlock(ch_in=config["channel_En_z"], ch_out=config["dim_z"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], use_res=config["hyper_use_res"])
                ])
            modules.extend([
                ComplexHyperLinear(ch_in=config["dim_z"], ch_out=config["dim_z"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], use_res=config["hyper_use_res"]),
                ])
            self.Encoder_z = nn.Sequential(*modules)
            
            modules = []
            insize = 2*self.filter_length
            modules.extend([
                ComplexHyperLinearBlock(ch_in=config["dim_z"], ch_out=config["channel_De_z"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], use_res=config["hyper_use_res"]),
                ])
            for l in range(config["hlayers_De_z"]):
                modules.extend([
                ComplexHyperLinearBlock(ch_in=config["channel_De_z"], ch_out=config["channel_De_z"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], use_res=config["hyper_use_res"]),
            ])
            modules.extend([
                ComplexHyperLinear(ch_in=config["channel_De_z"], ch_out=insize, ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], use_res=config["hyper_use_res"])
                ])
            self.Decoder_z = nn.Sequential(*modules)
            
        elif self.config["in_coeff"]:
            if self.config["use_cae"]:
                #== Encoder 1 ===
                modules = []
                if self.config["in_magphase"]:
                    ch_init = 2*3
                elif self.config["in_mag"]:
                    ch_init = 2
                else:
                    ch_init = 2*2
                num_layers_en = 4
                ch_en = config["channel_En_z"]
                h_in = self.config["num_pts"] if self.config["num_pts"] < 440 else 400
                size = [round(h_in), round(self.filter_length)]
                # size = round(self.filter_length)
                modules.extend([
                        nn.LayerNorm(size),
                        nn.Conv2d(ch_init, ch_en, config["kernel_size"], stride=1, padding=round((config["kernel_size"][0]-1)/2)),
                        nn.LayerNorm(size),
                        nn.ReLU()])
                for l in range(num_layers_en):
                    # ch_in = ch_init if l==0 else ch_en
                    ch_z = 1 if self.config["z_flatten"] else ch_en
                    ch_out = ch_z if l==num_layers_en-1 else ch_en
                    size = [round(np.floor(round(h_in)/2**l)), round(np.floor(self.filter_length/2**l))]
                    # size = round(np.floor(self.filter_length/2**l))

                    for ll in range(config["num_layers_res"]+1):
                        modules.extend([
                            ResConv2dBlock(size,ch_en,config["kernel_size"])])
                    modules.extend([
                        nn.Conv2d(ch_en, ch_out, (2,2), stride=2, padding=0),
                        ])
                # print(modules)
                # for m in modules:
                #     if hasattr(m,'weight'):
                #         th.nn.init.normal_(m.weight)
                #         print(m)
                self.Encoder_z_1 = nn.Sequential(*modules)

                if self.config["z_flatten"]:
                    #== Encoder 2 ===
                    modules = []
                    in_size = round(np.floor(h_in/2**(l+1)) * np.floor(self.filter_length/2**(l+1)))
                    modules.extend([nn.Linear(in_size,config["dim_z"]),
                                    nn.LayerNorm(config["dim_z"]),
                                    nn.ReLU()])
                    self.Encoder_z_2 = nn.Sequential(*modules)

                    #== Classifier ===
                    if self.config["use_metric"]:
                        modules = [ArcMarginProduct(in_features=config["dim_z_af"], out_features=77,m=0.5)]
                        self.Classifier_z = nn.Sequential(*modules)

                    #== Decoder 1 ===
                    modules = []
                    out_size = round(np.floor((self.maxto+1)**2/2**(l+1)) * np.floor(self.filter_length/2**(l+1)))
                    modules.extend([nn.Linear(config["dim_z"], out_size),
                                    nn.LayerNorm(out_size),
                                    nn.ReLU()])
                    self.Decoder_z_1 = nn.Sequential(*modules)

                #== Decoder 2 ===
                modules = []
                num_layers_de = num_layers_en
                ch_de = config["channel_En_z"]
                size = [round(np.floor((self.maxto+1)**2/2**(num_layers_de))), round(np.floor(self.filter_length/2**(num_layers_de)))]
                # size = round(np.floor(self.filter_length/2**(num_layers_de)))
                modules.extend([
                        nn.ConvTranspose2d(ch_z, ch_de, config["kernel_size"], stride=1, padding=round((config["kernel_size"][0]-1)/2)),
                        nn.LayerNorm(size),
                        nn.ReLU()])
                for l in range(num_layers_de):
                    size = [round(np.floor((self.maxto+1)**2/2**(num_layers_de-l-1))), round(np.floor(self.filter_length/2**(num_layers_de-l-1)))]
                    # size =round(np.floor(self.filter_length/2**(num_layers_de-l-1)))

                    modules.extend([
                        nn.ConvTranspose2d(ch_de, ch_de, (2,2), stride=2, padding=0),
                        ])
                    for ll in range(config["num_layers_res"]+1):
                        modules.extend([
                            ResConvTranspose2dBlock(size,ch_de,config["kernel_size"])])
                if self.config["out_magphase"]:
                    ch_out = 6
                else:
                    ch_out = 4
                modules.extend([
                        nn.ConvTranspose2d(ch_de, ch_out, config["kernel_size"], stride=1, padding=round((config["kernel_size"][0]-1)/2))])
                self.Decoder_z_2 = nn.Sequential(*modules)
            
            elif self.config["use_unet"]:
                self.unet = UNet(self.config)
            
            else:
                #== Encoder ===
                modules = []
                if self.config["in_magphase"]:
                    ch_init = 2*3
                elif self.config["in_mag"]:
                    ch_init = 2
                else:
                    ch_init = 2*2
                insize = round(ch_init * config["num_pts"] * self.filter_length)
                modules.extend([
                    nn.Linear(insize, config["channel_En_z"]),
                    nn.LayerNorm(config["channel_En_z"]),
                    nn.ReLU(),
                    nn.Dropout(config["droprate"])
                ])
                for l in range(round(config["hlayers_En_z"]/2)):
                    modules.extend([ResLinearBlock(config["channel_En_z"], config["droprate"], config["normalize"], config["num_pts"])])
                modules.extend([nn.Linear(config["channel_En_z"],config["dim_z"]),
                            nn.LayerNorm(config["dim_z"]),
                            nn.ReLU()])
                self.Encoder_z = nn.Sequential(*modules)

                #== Classifier ===
                if self.config["use_metric"]:
                    modules = [ArcMarginProduct(in_features=config["dim_z_af"], out_features=77,m=0.5)]
                    self.Classifier_z = nn.Sequential(*modules)

                #== Decoder ===
                modules = []
                modules.extend([
                    nn.Linear(config["dim_z"], config["channel_De_z"]),
                    nn.LayerNorm(config["channel_De_z"]),
                    nn.ReLU(), 
                    nn.Dropout(config["droprate"])])
                for l in range(round(config["hlayers_De_z"]/2)):
                   modules.extend([ResLinearBlock(config["channel_En_z"], config["droprate"], 'ln')])
                modules.extend([nn.Linear(config["channel_De_z"], self.coeff_size)])
                self.Decoder_z = nn.Sequential(*modules)
        else:
            modules = []
            if self.config["in_magphase"]:
                insize = 2*3*self.filter_length+3
            elif self.config["in_mag"]:
                if self.config["only_left"]:
                    insize = 1*1*self.filter_length+3
                else:
                    insize = 2*1*self.filter_length+3
            else:
                insize = 2*2*self.filter_length+3
            modules.extend([
                # nn.Linear(config["channel_En"], config["channel_En_z"]),
                nn.Linear(insize, config["channel_En_z"]),
                ])
            #=============
            if config["normalize"] == "ln":
                 modules.extend([nn.LayerNorm(config["channel_En_z"])])
            elif config["normalize"] == "bn":
                modules.extend([nn.BatchNorm1d(config["num_pts"])])
            #=============

            modules.extend([
                nn.ReLU(),
                nn.Dropout(config["droprate"])])
            for l in range(round(config["hlayers_En_z"]/2)):
                modules.extend([ResLinearBlock(config["channel_En_z"], config["droprate"], config["normalize"], config["num_pts"])])
            # for l in range(config["hlayers_En_z"]):
            #     modules.extend([nn.Linear(config["channel_En_z"], config["channel_En_z"]),
            #                     nn.LayerNorm(config["channel_En_z"]),
            #                     nn.ReLU(), 
            #                     nn.Dropout(config["droprate"])])
            modules.extend([nn.Linear(config["channel_En_z"],config["dim_z"]),
                            nn.LayerNorm(config["dim_z"]),
                            nn.ReLU()])
            self.Encoder_z = nn.Sequential(*modules)
            # initwb(self.Encoder_z)
            if self.config["use_metric"]:
                modules = [ArcMarginProduct(in_features=config["dim_z_af"], out_features=77,m=0.5)]
                self.Classifier_z = nn.Sequential(*modules)

            modules = []
            modules.extend([nn.Linear(config["dim_z"], config["channel_De_z"])])
            # if config["use_ln"]:
            modules.extend([nn.LayerNorm(config["channel_De_z"])])
            modules.extend([nn.ReLU(), 
                            nn.Dropout(config["droprate"])])
            # for l in range(config["hlayers_De_z"]):
            #     modules.extend([nn.Linear(config["channel_De_z"], config["channel_De_z"]),
            #                     nn.ReLU(), 
            #                     nn.Dropout(config["droprate"])])
            for l in range(round(config["hlayers_De_z"]/2)):
                # modules.extend([ResLinearBlock(config["channel_De_z"], config["droprate"], config["normalize"], config["num_pts"])])
                modules.extend([ResLinearBlock(config["channel_En_z"], config["droprate"], 'ln')])

            #===========
            # mid_ch = 2048
            # modules.extend(
            #     [nn.Linear(config["channel_De_z"], mid_ch),
            #                 nn.LayerNorm(mid_ch),
            #                 nn.ReLU(), 
            #                 nn.Dropout(config["droprate"]),
            #                 nn.Linear(mid_ch, self.coeff_size)
            #                 ])
            # - - - - -
            modules.extend([nn.Linear(config["channel_De_z"], self.coeff_size)])
            #============
            self.Decoder_z = nn.Sequential(*modules)
            # initwb(self.Decoder_z)
            
            
    def forward(self, input, use_cuda_forward=True, use_srcpos=True, srcpos=None, sub=None,mode='train', coeff=None):
        '''
        :param hrtf: the input as a (B, ch, L) or (B, ch, L, S) complex tensor
        :param srcpos: B x 3 float tensor
        :param sub: int
        :return: out: HRTF produce by the network, (B, ch, L) complex tensor
        '''
        # print(input.shape)  # torch.Size([440, 512, 4])
        # print(srcpos.shape) # torch.Size([440, 3, 4])
        # print(input.shape)
        returns = {}
        L = self.filter_length
        B = input.shape[0]
        S = input.shape[-1]
        mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude', top_db = 80) 

        # print(f"r:{r.shape}")
        # print(f"theta:{theta.shape}")
        # print(f"phi:{phi.shape}")
        if use_srcpos:
            srcpos = srcpos.cuda()
            r,phi,theta = srcpos[:,0,:],srcpos[:,1,:],srcpos[:,2,:] # phi \in [0,2pi)
            # r = srcpos[:,0]
            # phi = srcpos[:,1]
            # theta = srcpos[:,2]
            vec_cart_gt = sph2cart(phi,theta,r)
            # print(vec_cart_gt.shape) # torch.Size([440, 3, 4])

        # print(hrtf.dtype)
        # input = th.cat((th.real(hrtf[:,0,:]), th.imag(hrtf[:,0,:]), th.real(hrtf[:,1,:]), th.imag(hrtf[:,1,:])), dim=1)
        # print(f"input:{input.shape}") # input:torch.Size([440, 512])

        # # HyperNet
        # Hweight = self.HyperEn_w(vec_cart_gt)
        # Hweight = Hweight.view(-1,2*2*self.filter_length, self.config["channel_En"])
        # Hbias = self.HyperEn_b(vec_cart_gt)
        # input = th.bmm(input.view(-1,1,2*2*self.filter_length),Hweight).squeeze() + Hbias
        if self.config["in_coeff"] or self.config['coeff_skipconnection']:
            if self.config["num_pts"] < 440 or coeff == None:
                hrtf_re = th.cat((input[:,   :L].unsqueeze(1), input[:,2*L:3*L].unsqueeze(1)), dim=1)
                hrtf_im = th.cat((input[:,L:2*L].unsqueeze(1), input[:,3*L:4*L].unsqueeze(1)), dim=1)
                hrtf = th.complex(hrtf_re, hrtf_im)

                #=== solve linear inverse problem ====
                vec_cart_gt = sph2cart(srcpos[:,1,:],srcpos[:,2,:],srcpos[:,0,:]) # sub==0
                if self.config["num_pts"] < 440:
                    idx_t_des = aprox_t_des(pts=vec_cart_gt[:,:,0]/1.47, t=self.to_in, plot=False)
                    pos_t = srcpos[idx_t_des,:,:]
                    hrtf_t  = hrtf[idx_t_des,:,:,:]
                else:
                    pos_t = srcpos
                    hrtf_t  = hrtf
                #--- Special Function ----
                SpF_t = SpecialFunc(r=pos_t[:,0,0], phi=pos_t[:,1,0], theta=pos_t[:,2,0], maxto=self.to_in, fft_length=self.fft_length, max_f=self.max_f)
                Hank_t = SpF_t.SphHankel()
                Hank_t = th.complex(th.nan_to_num(th.real(Hank_t)), th.nan_to_num(th.imag(Hank_t)))
                Harm_t = SpF_t.SphHarm()
                #---
                B_t = pos_t.shape[0]

                #--- Linear inv problem. ---
                N = self.to_in # fixed truncation
                n_vec,_ = sph_harm_nmvec(N)
                num_base = (N+1)**2

                #--- matrix ---
                mat_base = th.zeros(self.filter_length, B_t, num_base, dtype = th.complex64).to(hrtf.device) # Phi; matrix contains bases. (L, B, (N+1)**2)
                # Hank_t: (B,N,L), Harm_t; (B, (N+1)**2)
                for i,n in enumerate(n_vec):
                    mat_base[:,:,i] = Hank_t.permute(2,0,1)[:,:,n] * Harm_t[None,:,i]
                #--- Phi^H
                mat_base_H = th.conj(th.transpose(mat_base,1,2)) 
                #--- Phi^+ = Phi^H * Phi
                mat_base_MPI = th.matmul(mat_base_H, mat_base) 
                #--- regularization matrix in Duraiswaini+04
                #--- D=(1+n(n+1))I
                D = th.diag(1+th.from_numpy((n_vec*(n_vec+1)).astype(np.float32)).clone()).to(hrtf.device)
                # reg_w = self.config["loss_weights"]["reg"]
                reg_w = self.config["reg_w"]
                mat_reg = reg_w * D # (100,100)
                mat = mat_base_MPI + mat_reg[None,:,:] # (L, B, (N+1)**2)

                #--- vector ---
                # hrtf_t: (100, 2, 128, 77)
                # print(hrtf_t.shape)
                h = hrtf_t.permute(2,0,1,3) # (128, 100, 2, 77)
                h = h.contiguous().view(h.shape[0], h.shape[1], -1) # (128, 100, 154)

                #--- solve ---
                # c: (128, 100, 154)
                c = th.linalg.solve(mat ,th.matmul(mat_base_H, h))
                c = c.view(c.shape[0], c.shape[1], 2, -1) # (128, 100, 2, 77)
                Coeff_in = c.permute(2,1,0,3) # (2, 100, 128, 77)
                returns['coeff_in'] = Coeff_in
                Coeff_in = Coeff_in.permute(3,0,1,2) # (77, 2, 100, 128)
                #=====================================
                Coeff_in = th.cat((th.real(Coeff_in),th.imag(Coeff_in)), dim=1)
                Coeff_in = Coeff_in.permute(1,2,3,0)
                # print(Coeff_in.shape) # torch.Size([4, 400, 128, 1])
            else:
                Coeff_in = coeff
                # print(coeff.shape) # torch.Size([4, 400, 128, 16])

        if self.config["in_coeff"]:
            #===============
            if self.config["in_mag"]:
                Coeff_in_db = mag2db(th.abs(Coeff_in))/20 # dB/20
                input = (Coeff_in_db - 0.1588) / 0.3769
                # print(th.mean(Coeff_in_db)) # tensor(0.1588, device='cuda:0')
                # print(th.mean(th.std(Coeff_in_db, dim=[2,3]))) # tensor(0.6873, device='cuda:0')
                # print(th.mean(th.std(Coeff_in_db, dim=[0,2]))) # tensor(0.3769, device='cuda:0')
            elif self.config["in_magphase"]:
                Coeff_in_db = mag2db(th.abs(Coeff_in))/20 # dB/20
                Coeff_in_db = (Coeff_in_db - 0.1588) / 0.3769

                Coeff_in_re_norm = th.real(Coeff_in)/th.abs(Coeff_in)
                Coeff_in_im_norm = th.imag(Coeff_in)/th.abs(Coeff_in)
                Coeff_phase = th.cat((Coeff_in_re_norm, Coeff_in_im_norm), dim=1)
                # Coeff_phase = Coeff_phase / 0.7
                # print(th.mean(Coeff_phase)) # tensor(-0.0002, device='cuda:0')
                # print(th.mean(th.std(Coeff_phase, dim=[0,2]))) # tensor(0.7001, device='cuda:0')
                input = th.cat((Coeff_in_db,Coeff_phase), dim=1)
                # print(th.max(input)) # tensor(6.7974, device='cuda:0')
                # print(th.min(input)) # tensor(-26.9536, device='cuda:0') # tensor(-4.3663, device='cuda:0') (if top_db=80)
            else:
                if self.config["num_pts"] < 440 or coeff == None:
                    # print(Coeff_in.shape)
                    Coeff_in_re, Coeff_in_im = th.real(Coeff_in), th.imag(Coeff_in)
                    input = th.cat((Coeff_in_re, Coeff_in_im), dim=1) # (77, 4, 100, 128)
                else:
                    # print(Coeff_in.shape) # torch.Size([4, 400, 128, 16])
                    input = Coeff_in.permute(3,0,1,2)
                # print(th.max(input)) # tensor(505.8159, device='cuda:0')
                # print(th.min(input)) # tensor(-504.0186, device='cuda:0')
                scale = 500
                input = input / scale
            #---------------
            if self.config["use_cae"]:
                # print(input.shape) # torch.Size([1, 6, 400, 128])
                z = self.Encoder_z_1(input)
                # print(th.max(z)) # tensor(29.2497, device='cuda:0', grad_fn=<MaxBackward1>)
                # print(th.min(z)) # tensor(-66.0351, device='cuda:0', grad_fn=<MinBackward1>)
                if th.any(th.isnan(z)):
                    print('Encoder_z_1')
                    sys.exit()
                # print(z.shape) # torch.Size([1, 1, 25, 8])
                if self.config["z_flatten"]:
                    z = z.view(z.shape[0],-1)
                    # print(z.shape) # torch.Size([1, 200])
                    z = self.Encoder_z_2(z)
                    if th.any(th.isnan(z)):
                        print('Encoder_z_2')
                        sys.exit()
                    # print(z.shape) # torch.Size([77, 64])
                    returns['z'] = z.permute(1,0).unsqueeze(0)
                else:
                    # print(input.shape) # torch.Size([16, 4, 400, 128])
                    returns['z'] = th.zeros(1,16,input.shape[0]) # dammy
                    returns['z_cae'] = z
                # print(returns['z'].shape) # torch.Size([1,64,77])
            elif self.config["use_unet"]:
                Coeff = self.unet(input)
                returns['z'] = th.zeros([B,16,S]).to(Coeff.device) # dammy
            else:
                # print(input.shape) # torch.Size([8, 6, 100, 128])
                input = input.view(input.shape[0],-1)
                z = self.Encoder_z(input)
                # print(z.shape) # torch.Size([8, 64])
                returns['z'] = z.permute(1,0).unsqueeze(0)
                # print(returns['z'].shape) # torch.Size([1,64,8])

        else:
            if self.config["in_magphase"] or self.config["in_mag"]:
                # re/im -> mag/phase =======================
                # mag (dB/20)
                mag_l = th.sqrt(th.abs(input[:,0*L:1*L])**2 + 
                                th.abs(input[:,1*L:2*L])**2)
                mag_r = th.sqrt(th.abs(input[:,2*L:3*L])**2 + 
                                th.abs(input[:,3*L:4*L])**2)
                # eps = 1e-10*th.ones(mag_l.shape).to(input.device)
                magdb_l = mag2db(th.abs(mag_l))/20 #th.log10(th.max(mag_l, eps))
                magdb_r = mag2db(th.abs(mag_r))/20 #th.log10(th.max(mag_r, eps))
                magdb_lr = th.cat((magdb_l,magdb_r), dim=1)
                # print(magdb_lr.shape)
                
                # dep = (1,2) if self.config["batch_size"] == 1 else (1,2,1)
                dep = (1,2,1)
                phase_l = input[:,0*L:2*L] / th.tile(mag_l, dep)
                phase_r = input[:,2*L:4*L] / th.tile(mag_r, dep)
                
                magdb_mean = -1.5852 if self.config["green"] else -0.3180
                magdb_std = 0.5688
                if self.config["in_mag"]:
                    if self.config["in_mag_pc"]:
                        Q = self.config["num_pc"]
                        V = self.pd_train
                        # print(magdb_l.shape) # torch.Size([440, 128, 32])
                        # print(V.shape) # torch.Size([128, 64])
                        input = th.cat((th.matmul(magdb_l.permute(2,0,1), V[None,:,:Q]),th.matmul(magdb_r.permute(2,0,1), V[None,:,:Q])),dim=2)
                        # print(input.shape) # torch.Size([S, 440, 2Q])
                        input = input.permute(1,2,0)  # torch.Size([440, 2Q, S])
                        # sys.exit()
                    elif self.config["only_left"]:
                        magdb_l = (magdb_l + 1.5845) / 0.4343
                        input = magdb_l
                    else:
                        magdb_lr = (magdb_lr - magdb_mean) / magdb_std
                        input = magdb_lr
                elif self.config["in_magphase"]:
                    magdb_lr = (magdb_lr - magdb_mean) / magdb_std
                    # [mag_lr] mean:-1.5852205753326416, std:0.568795919418335
                    # print(f"mean:{th.mean(magdb_lr):.2f}, std:{th.std(magdb_lr):.2f}")
                    phase_lr = th.cat((phase_l,phase_r), dim=1)
                    # phase_lr = (phase_lr - 0.021) / 0.707
                    # [phase_lr] mean:0.021020939573645592, std:0.707058846950531
                    input = th.cat((magdb_lr,phase_lr), dim=1)
                    # print(f"(mag,phase)...  mean:{th.mean(input):.2f}, std:{th.std(input):.2f}")
                    
                    # print([th.max(input), th.min(input)])
                    # [tensor(-0.3284, device='cuda:0'), tensor(-4.8487, device='cuda:0')]
                    # print([th.mean(input), th.std(input)])
                    # [tensor(-1.4708, device='cuda:0'), tensor(0.5376, device='cuda:0')]
                    # print(input.shape) # torch.Size([440, 256])
                    #===========================================
            else:
                input = input / 0.0483
            if self.config["en_aux"] == 'cart_coord':
                input = th.cat((input, vec_cart_gt), dim=1)
            elif self.config["en_aux"] == 'sph_coord':
                input = th.cat((input, srcpos), dim=1)
            elif self.config["en_aux"] == 'RSH':
                print("[Error] Not implemented.")
                sys.exit()
            # print(input.shape)
            # zz = self.Encoder(input)
            # print(f"zz[0,0:3]: {zz[0,:3]}")
            # z = self.Encoder_z(zz)
            #====== random sampling ========
            if self.config["random_sample"] and mode == 'train':
                #-----------------
                # # idx_rs = th.arange(0,input.shape[0])
                # # idx_rs=idx_rs[th.randperm(idx_rs.size()[0])]
                # idx_rs = th.argsort(th.rand(input.shape[-1], input.shape[0])) # 4,440
                # idx_rs = idx_rs[:,:self.config["num_pts"]].permute(1,0) # 25,4
                # input_rs = th.zeros(idx_rs.shape[0],input.shape[1],idx_rs.shape[1]).to(input.device)
                # for b in range(input.shape[-1]):
                #     input_rs[:,:,b] = input[idx_rs[:,b],:,b]
                # # input_rs = input[idx_rs,:]
                # # print(input.shape)  # torch.Size([440, 512, 4])
                # # idx_t_des = idx_rs
                #---- (2022/07/01~↓)-------------
                perm = th.randperm(input.shape[0])
                idx_t_des = perm[:self.config["num_pts"]]
                input_rs = input[idx_t_des]
                #-----------------
            elif self.config["num_pts"] == 1:
                idx = 211 # idx=202(azimth=0deg),211(90),220(180),229(270)
                input_rs = input[idx:idx+1]
            elif self.config["pln_smp"]:
                idx_t_des = plane_sample(pts=vec_cart_gt[:,:,0], axes=self.config["pln_smp_axes"], thr=0.01)
                input_rs = input[idx_t_des] # (B', 2L+3, S)
            elif self.config["pln_smp_paral"]:
                idx_t_des = parallel_planes_sample(pts=vec_cart_gt[:,:,0], 
                values=th.tensor([-.5,0.0,.5]).to(srcpos.device)*srcpos[0,0,0], axis=self.config['pln_smp_paral_axis'], thr=0.01)
                input_rs = input[idx_t_des] # (B', 2L+3, S)
            elif self.config["num_pts"] < min(input.shape[0],362):
                self.to_in = round(self.config["num_pts"]**0.5-1)
                idx_t_des = aprox_t_des(pts=vec_cart_gt[:,:,0]/1.47, t=self.to_in, plot=False)
                input_rs = input[idx_t_des]
            else:
                idx_t_des = range(0,B)
                input_rs = input
            # print(input_rs.shape) # torch.Size([25, 771, 4])
            #===============================
            returns["idx_mes_pos"] = idx_t_des

            input_rs = input_rs.permute(2,0,1) # torch.Size([4, 25, 771])
            # print(input_rs.shape)
            # print(input_rs.shape) # torch.Size([4, 25, 771])
            if self.config["use_hypernet"]:
                if self.config["use_lr_aug"]:
                    input_rs_x = th.cat((input_rs[:,:,0*L:1*L],input_rs[:,:,1*L:2*L]),dim=0) # (2S,B,L)
                    pos_l = input_rs[:,:,-3:]
                    if self.config["en_aux"] == 'cart_coord':
                        pos_r = pos_l * th.tensor([1,-1,1],device=pos_l.device)[None, None, :] # y座標 反転
                    elif self.config["en_aux"] == 'sph_coord':
                        pos_r = pos_l * th.tensor([1,-1,1],device=pos_l.device)[None, None, :] + th.tensor([0,2*np.pi,0],device=pos_l.device)
                    input_rs_z = th.cat((pos_l, pos_r), dim=0)  # (2S,B,3)
                    # print(input_rs_x.shape)
                    # print(input_rs_z.shape)
                else:
                    input_rs_x = input_rs[:,:,:-3]
                    input_rs_z = input_rs[:,:,-3:]
                input_rs_dict = {
                    "x": input_rs_x,
                    "z": input_rs_z,
                }
                z = self.Encoder_z(input_rs_dict)["x"]
                # print(f'[input_rs_x] mean:{th.mean(input_rs_x)}, std:{th.std(input_rs_x)}, max:{th.max(input_rs_x)}, min:{th.min(input_rs_x)}')
                # print(f'[input_rs_z] mean:{th.mean(input_rs_z)}, std:{th.std(input_rs_z)}, max:{th.max(input_rs_z)}, min:{th.min(input_rs_z)}')
                # print(f'[z] mean:{th.mean(z)}, std:{th.std(z)}, max:{th.max(z)}, min:{th.min(z)}')
                if th.any(th.isnan(z)):
                    print('nan is detected in z.')
                if th.any(th.isinf(z)):
                    print('inf is detected in z.')
                z = th.nan_to_num(z,nan=0.0,posinf=0.0,neginf=0.0)
            elif self.config["use_complexhypernet"]:
                input_rs_x = input_rs[:,:,:-3]
                input_rs_z = input_rs[:,:,-3:]
                input_rs_dict = {
                    "re": input_rs_x[:,:,:2*L],
                    "im": input_rs_x[:,:,2*L:],
                    "z": input_rs_z,
                }
                z = self.Encoder_z(input_rs_dict)
                z = th.cat((z["re"],z["im"]),dim=-1)
                # print(z.shape) # (S,B,2*dimz)
                returns['z'] = z.permute(1,2,0)  # torch.Size([B, dimz, S])
                z = th.mean(z, dim=1)
                # print(z.shape) # (S,2*dimz)
                z = th.tile(z.unsqueeze(1),dims=(1,B,1))
                # print(z.shape) # (S,B,2*dimz)
                input_dec_dict = {
                    "re": z[:,:,:self.config["dim_z"]], # (S,B,dimz)
                    "im": z[:,:,self.config["dim_z"]:], # (S,B,dimz)
                    "z": vec_cart_gt.permute(2,0,1)  # (S,B,3)
                }
                #=== Decode ====
                out_f = self.Decoder_z(input_dec_dict) # (S,B,6L) or (S,B,4L)
                #=== Transform ====
                out = th.zeros(S,B,2,L, dtype=th.complex64, device=out_f["re"].device)
                # out_f: (S,B,4L) float -> out:(S,B,2,L) complex
                out[:,:,0,:] = th.complex(out_f["re"][:,:,0*L:1*L], out_f["im"][:,:,0*L:1*L])
                out[:,:,1,:] = th.complex(out_f["re"][:,:,1*L:2*L], out_f["im"][:,:,1*L:2*L])

                returns['output'] = out.permute(1,2,3,0)
                #torch.Size([440, 2, 128, S])
                # print('----------')
                # sys.exit()
                if 'weight_mean' not in returns.keys():
                    bb = len(idx_t_des) if 'idx_t_des' in locals() else B
                    returns['weight_mean'] = th.ones(bb)/bb
                ### dammy
                returns['vec'] = vec_cart_gt
                returns['theta'] = theta # zenith
                returns['phi'] = phi # azimuth
                ###
                if 'coeff' not in returns.keys():
                    returns['coeff'] = th.zeros(2,(self.maxto+1)**2,L,S, dtype=th.complex64)

                return returns
            else:
                z = self.Encoder_z(input_rs)
            # print(z.shape) # torch.Size([32, 169, 128])
            # print(th.norm(z)**2) # 7304597.5
            if self.config["z_norm"]:
                if self.config["dim_z_af"] < self.config["dim_z"]:
                    # print(z.shape)
                    z = th.cat((F.normalize(z[:,:,:self.config["dim_z_af"]],dim=-1),F.normalize(z[:,:,self.config["dim_z_af"]:],dim=-1)), dim=-1)
                else:
                    z = F.normalize(z,dim=-1)
            # print(th.norm(z)**2) # 5408
            # print(z.shape) # torch.Size([S, B, dimz])
            z = z.permute(1,0,2) # torch.Size([B, S, dimz])
            if self.config["num_classifier"]==1:
                returns['z'] = z.permute(0,2,1) # torch.Size([B, dimz, S])
            # print(z.shape)
            if self.config["num_classifier"]==8:
                bool_xp = input_rs_z[0,:,0] > 0
                bool_yp = input_rs_z[0,:,1] > 0
                bool_zp = input_rs_z[0,:,2] > 0
                # print(bool_xp.shape) # torch.Size([361])
                itr = 0
                for i in range(self.config["num_classifier"]):
                    bool_x = th.logical_xor(bool_xp,th.tensor([(itr>>0)&1],device=z.device)) 
                    bool_y = th.logical_xor(bool_yp,th.tensor([(itr>>1)&1],device=z.device)) 
                    bool_z = th.logical_xor(bool_zp,th.tensor([(itr>>2)&1],device=z.device)) 
                    # print(bool_x.shape) # torch.Size([361])
                    bool_xyz = th.logical_and(th.logical_and(bool_x,bool_y),bool_z)
                    # print(bool_xyz.shape) # torch.Size([361])
                    exec(f'z_{i}=z[bool_xyz,:,:]')
                    # print(eval(f"z_{i}.shape")) 
                    # torch.Size([35, 32, 64])
                    # torch.Size([40, 32, 64])
                    # torch.Size([45, 32, 64])
                    # torch.Size([44, 32, 64])
                    # torch.Size([43, 32, 64])
                    # torch.Size([50, 32, 64])
                    # torch.Size([56, 32, 64])
                    # torch.Size([48, 32, 64])
                    # sum: 361 --> OK!
                    exec(f"returns[f'z_{i}'] = z_{i}.permute(0,2,1)")
                    itr += 1
            
            if not self.config["metric_before"]:
                if self.config["use_attention"]:
                    att_input = {
                        "value": z,
                        "target": input_rs_z[0,:,:]
                    }
                    zz = self.attention(att_input)
                    z = zz["w_mean"]
                    returns['weight_mean'] = zz["weight"]
                else:
                    z = th.mean(z, dim=0) # torch.Size([4, 16])
                if self.config["z_norm"]:
                    if self.config["dim_z_af"] < self.config["dim_z"]:
                        z = th.cat((F.normalize(z[:,:,:self.config["dim_z_af"]],dim=-1),F.normalize(z[:,:,self.config["dim_z_af"]:],dim=-1)), dim=-1)
                        # z = th.cat((F.normalize(z[:,:self.config["dim_z_af"]],dim=-1),z[:,self.config["dim_z_af"]:]), dim=-1)
                    else:
                        z = F.normalize(z,dim=-1)
                    # z = F.normalize(z[:,:,:self.config["dim_z_af"]],dim=-1)
        #====== classifier (ArcFace) ===
        if self.config["use_metric"]:
            if not sub==None:
                if self.config["num_classifier"]==1:
                    if self.config["metric_before"]:
                        label = th.tile(sub,(z.shape[0],1)).to(z.device) # torch.Size([100, 77])
                        label = label.contiguous().view(label.shape[0]*label.shape[1]) # torch.Size([7700])
                        z_m = z.contiguous().view(z.shape[0]*z.shape[1],-1)[:,:self.config["dim_z_af"]] # torch.Size([7700, 64])
                    else:
                        label = sub.to(z.device)
                        z_m = z[:,:self.config["dim_z_af"]]
                    # print(label) # tensor([0, 1, 2, 3], device='cuda:0')
                    in_metric = {
                        'input': z_m,
                        'label': label
                    }
                    out_metric = self.Classifier_z(in_metric)
                    returns['metric'] = out_metric
                    returns['label'] = label
                    # print(z.shape) # torch.Size([169, 32, 128])
           
            elif self.config["num_classifier"]>1:
                for i in range(self.config["num_classifier"]):
                    z_i = eval(f'z_{i}')
                    # print(z_i.shape)
                    label = th.tile(sub,(z_i.shape[0],1)).to(z_i.device) 
                    label = label.contiguous().view(label.shape[0]*label.shape[1]) 
                    z_m = z_i.contiguous().view(z_i.shape[0]*z_i.shape[1],-1) 
                    in_metric = {
                    'input': z_m,
                    'label': label
                    }
                    out_metric = eval(f"self.Classifier_z_{i}(in_metric)")
                    returns[f'metric_{i}'] = out_metric
                    returns[f'label_{i}'] = label

        if self.config["num_classifier"]>1:
            # print(self.config["num_classifier"])
            for i in range(self.config["num_classifier"]):
                z_i = eval(f'z_{i}')
                z_i_prot = th.mean(z_i, dim=0)
                if self.config["z_norm"]:
                    z_i_prot = F.normalize(z_i_prot,dim=-1)
                if i == 0:
                    z = z_i_prot
                    z_ret = z_i
                else:
                    z = th.cat((z,z_i_prot),dim=-1)
                    z_ret = th.cat((z_ret,z_i),dim=0)
            returns['z'] = z_ret.permute(0,2,1) # torch.Size([B, dimz, S])
            # print(z_ret.shape)
            # print(z.shape)
            # print(z.shape) # (S,8L) # torch.Size([32, 512])
        
        if self.config["metric_before"]:
            z = th.mean(z, dim=0)
            # print(th.norm(z)**2) # 
            if self.config["z_norm"]:
                if self.config["dim_z_af"] < self.config["dim_z"]:
                    z = th.cat((F.normalize(z[:,:self.config["dim_z_af"]],dim=-1),F.normalize(z[:,self.config["dim_z_af"]:],dim=-1)), dim=-1)
                    # z = th.cat((F.normalize(z[:,:self.config["dim_z_af"]],dim=-1),z[:,self.config["dim_z_af"]:]), dim=-1)
                else:
                    z = F.normalize(z,dim=-1)

        if self.config["Decoder"] == "hrtf_hyper":
            #=== Decoder: Hyper ====
            # z: (S,dimz) -> (S,B,dimz)
            # num_tile = self.config["num_pts"] if mode == 'train' or 'valid' else B
            
            z = th.tile(z.unsqueeze(1),dims=(1,B,1))
            input_dec_dict = {}
            if self.config["use_lr_aug"]:
                input_dec_dict["x"] = z # (2S,B,dimz)
                if self.config["de_aux"] == 'cart_coord':
                    pos_l_gt = vec_cart_gt.permute(2,0,1)  # (S,B,3)
                    pos_r_gt = pos_l_gt * th.tensor([1,-1,1], device=pos_l_gt.device)[None, None, :] # y座標 反転
                elif self.config["de_aux"] == 'sph_coord':
                    pos_l_gt = srcpos.permute(2,0,1)  # (S,B,3)
                    pos_r_gt = pos_l_gt * th.tensor([1,-1,1], device=pos_l_gt.device)[None, None, :] + th.tensor([0,2*np.pi,0], device=pos_l_gt.device)
                
                input_dec_dict["z"] = th.cat((pos_l_gt, pos_r_gt),dim=0) # (2S,B,L)
            else:
                input_dec_dict["x"] = z # (S,B,dimz)
                input_dec_dict["z"] = vec_cart_gt.permute(2,0,1)  # (S,B,3)
            # input_dec_dict["z"] = input_rs_z if mode ==  'train' or 'valid' else vec_cart_gt  # (S,B,3)
            #=== Decode ====
            out_f = self.Decoder_z(input_dec_dict)["x"] # (S,B,6L) or (S,B,4L)
            #=== Transform ====
            out = th.zeros(S,B,2,L, dtype=th.complex64, device=out_f.device)
            if self.config["out_magphase"]:
                # out_f: (S,B,6L) float -> out:(S,B,2,L) complex
                out[:,:,0,:] = 10**(out_f[:,:,0*L:1*L] * magdb_std + magdb_mean) * th.exp(1j * th.atan2(out_f[:,:,1*L:2*L], out_f[:,:,2*L:3*L]))
                out[:,:,1,:] = 10**(out_f[:,:,3*L:4*L] * magdb_std + magdb_mean) * th.exp(1j * th.atan2(out_f[:,:,4*L:5*L], out_f[:,:,5*L:6*L]))
            elif self.config["out_mag"]:
                if self.config["out_mag_pc"]: # Principal Component Analysis
                    # out_f: (S,B,2Q) float -> out:(S,B,2,L) complex
                    Q = self.config["num_pc"]
                    V = self.pd_train
                    out = th.zeros(S,B,2,L, dtype=th.complex64, device=out_f.device)
                    out[:,:,0,:] = 10**(th.matmul(out_f[:,:,0*Q:1*Q],V.T[None,None,:Q,:]))
                    out[:,:,1,:] = 10**(th.matmul(out_f[:,:,1*Q:2*Q],V.T[None,None,:Q,:]))
                else:
                    # out_f: (S,B,2L) float -> out:(S,B,2,L) complex
                    out = th.zeros(S,B,2,L, dtype=th.complex64, device=out_f.device)
                    if self.config["use_lr_aug"]:
                        out[:,:,0,:] = 10**(out_f[0*S:1*S,:,:] * magdb_std + magdb_mean)
                        out[:,:,1,:] = 10**(out_f[1*S:2*S,:,:] * magdb_std + magdb_mean)
                    else:
                        out[:,:,0,:] = 10**(out_f[:,:,0*L:1*L] * magdb_std + magdb_mean)
                        out[:,:,1,:] = 10**(out_f[:,:,1*L:2*L] * magdb_std + magdb_mean)
            else:
                # out_f: (S,B,4L) float -> out:(S,B,2,L) complex
                out[:,:,0,:] = th.complex(out_f[:,:,0*L:1*L], out_f[:,:,1*L:2*L])
                out[:,:,1,:] = th.complex(out_f[:,:,2*L:3*L], out_f[:,:,3*L:4*L])
            # print(th.max(th.abs(out))) # tensor(45.1883, grad_fn=<MaxBackward1>)
            # print(th.min(th.abs(out))) # tensor(0.4808, grad_fn=<MinBackward1>)
            
        elif  self.config["Decoder"] == "hrtf_mlp":
            #=== Decoder: MLP(not Hyper) ===
            z = th.tile(z.unsqueeze(1),dims=(1,B,1)) # (S,B,dimz)
            # print(z.shape) # torch.Size([32, 440, 64])
            #=== concat ([z,pos]) ===
            z = th.cat((z,vec_cart_gt.permute(2,0,1)), dim=-1) # (S,B,simz+3)
            # print(z.shape) # torch.Size([32, 440, 67])
            #=== Decode ===
            out_f = self.Decoder_z(z) # (S,B,6L) or (S,B,4L)
            #=== Transform ====
            if self.config["in_magphase"]:
                # out_f: (S,B,6L) float -> out:(S,B,2,L) complex
                out = th.zeros(S,B,2,L, dtype=th.complex64, device=out_f.device)
                out[:,:,0,:] = 10**(out_f[:,:,0*L:1*L] * magdb_std + magdb_mean) * th.exp(1j * th.atan2(out_f[:,:,1*L:2*L], out_f[:,:,2*L:3*L]))
                out[:,:,1,:] = 10**(out_f[:,:,3*L:4*L] * magdb_std + magdb_mean) * th.exp(1j * th.atan2(out_f[:,:,4*L:5*L], out_f[:,:,5*L:6*L]))
            else:
                # out_f: (S,B,4L) float -> out:(S,B,2,L) complex
                out = th.zeros(S,B,2,L, dtype=th.complex64, device=out_f.device)
                out[:,:,0,:] = th.complex(out_f[:,:,0*L:1*L], out_f[:,:,1*L:2*L])
                out[:,:,1,:] = th.complex(out_f[:,:,2*L:3*L], out_f[:,:,3*L:4*L])
        elif self.config["Decoder"] == "coeff" or "coeff_hyper":
            #=== Decoder's return: spherical wavefunction coeff. ===
            #=== Special Function ====
            SpF = SpecialFunc(r=r[:,0], phi=phi[:,0], theta=theta[:,0], maxto=self.maxto, fft_length=self.fft_length, max_f=self.max_f)
            n_vec, m_vec = SpF.n_vec, SpF.m_vec
            Hank = SpF.SphHankel()
            Hank = th.complex(th.nan_to_num(th.real(Hank)), th.nan_to_num(th.imag(Hank)))
            Harm = SpF.SphHarm()
            if use_cuda_forward:
                Hank = Hank.cuda()
                Harm = Harm.cuda()
            #==========

            if self.config["in_coeff"]:
                if self.config["use_cae"]:
                    if self.config["z_flatten"]:
                        # print(f'z:{z.shape}')
                        Coeff = self.Decoder_z_1(z)
                        if th.any(th.isnan(Coeff)):
                            print('Decoder_z_1')
                            sys.exit()
                        # print(Coeff.shape) # torch.Size([77, 200])
                        H_z = round(np.floor((self.maxto+1)**2/2**4))
                        W_z = round(np.floor(self.filter_length/2**4))
                        Coeff = Coeff.view(Coeff.shape[0],1,H_z,W_z)
                        # print(Coeff.shape)
                    else:
                        Coeff = z
                    Coeff = self.Decoder_z_2(Coeff)
                    if th.any(th.isnan(Coeff)):
                        print('Decoder_z_2')
                        sys.exit()
                    # print(Coeff.shape) # torch.Size([77, 6, 400, 128])
                    if self.config["out_magphase"]:
                        Coeff = transformCoeff_MagdBPhase_CNN(Coeff,self.maxto, self.filter_length, self.c_std.to(Coeff.device), self.c_mean.to(Coeff.device),self.config["std_dim"])
                    else:
                        Coeff = th.complex(Coeff[:,:2,:,:],Coeff[:,2:,:,:])
                        Coeff = Coeff * scale
                    if th.any(th.isnan(Coeff)):
                        print('transformCoeff_MagdBPhase_CNN')
                        sys.exit()
                    # print(Coeff.shape) # torch.Size([77, 2, 400, 128])
                elif self.config["use_unet"]:
                    Coeff = transformCoeff_MagdBPhase_CNN(Coeff,self.maxto, self.filter_length, self.c_std.to(Coeff.device), self.c_mean.to(Coeff.device),self.config["std_dim"])
                    if th.any(th.isnan(Coeff)):
                        print('transformCoeff_MagdBPhase_CNN')
                        sys.exit()
                else:
                    Coeff = self.Decoder_z(z)
                    # print(Coeff.shape) # torch.Size([8, 288276])
                    Coeff = transformCoeff_MagdBPhase(Coeff.view(S,-1), n_vec, th.from_numpy(self.N_arr_np.astype(np.float32)).clone(), self.maxto, self.filter_length, self.c_std.to(Coeff.device), self.c_mean.to(Coeff.device),self.config["std_dim"])
                    # print(Coeff.shape) # torch.Size([8, 2, 400, 128])
            elif self.config["Decoder"] == "coeff":
                Coeff = self.Decoder_z(z)
                # print(Coeff.shape) # torch.Size([4, 261132])
                if self.config["out_magphase"]:
                    if self.config["coeff_skipconnection"]:
                        self.c_std = th.ones_like(self.c_std)
                        self.c_mean = th.zeros_like(self.c_mean)
                    Coeff = transformCoeff_MagdBPhase(Coeff.view(S,-1), n_vec, th.from_numpy(self.N_arr_np.astype(np.float32)).clone(), self.maxto, self.filter_length, self.c_std.to(Coeff.device), self.c_mean.to(Coeff.device),self.config["std_dim"])
                else:
                    Coeff = self.out_z(Coeff)
                    scale = 750 # if self.config["coeff_skipconnection"] else 750
                    Coeff = transformCoeff(Coeff.view(S,-1), n_vec, th.from_numpy(self.N_arr_np.astype(np.float32)).clone(), self.maxto, self.filter_length, scale)
                if self.config["coeff_skipconnection"]:
                    # print(Coeff.shape) # torch.Size([S, 2, (N+1)**2, L])
                    # print(Coeff_in.shape) # torch.Size([4, (N+1)**2, L, S]) # coeff_in = th.cat((th.real(coeff),th.imag(coeff)),dim=0)

                    # print(th.complex(Coeff_in[:2],Coeff_in[2:]).permute(3,0,1,2).shape) # torch.Size([S, 2, (N+1)**2, L])
                    Coeff_skip = th.complex(Coeff_in[:2],Coeff_in[2:]).permute(3,0,1,2)
                    Coeff_skip = F.pad(Coeff_skip, (0,0,0, Coeff.shape[2]-Coeff_skip.shape[2],0,0,0,0))  # torch.Size([4, (N+1)**2, L, S])

                    Coeff = Coeff + Coeff_skip
                    # Coeff = Coeff_skip

                    # print(Coeff_skip.shape)
                    # print(Coeff_skip[0,0,0,0])
                    # print(Coeff_skip[0,0,-1,0])
            elif self.config["Decoder"] == "coeff_hyper":
                #=== Decoder: Coeff Hyper ====
                # z: (S,dimz) -> (S,(N+1)**2,dimz)
                z = th.tile(z.unsqueeze(1),dims=(1,(self.maxto+1)**2,1))
                # m,n: ((N+1)**2) --> (S,(N+1)**2,2)
                n_t, m_t = th.from_numpy(n_vec.astype(np.float32)), th.from_numpy(m_vec.astype(np.float32))
                nm = th.cat((n_t.unsqueeze(1), m_t.unsqueeze(1)), dim=1)
                nm = th.tile(nm.unsqueeze(0),dims=(S,1,1)).to(z.device)
                #=== Decode ====
                input_dec_dict = {
                    "x": z,  # (S,(N+1)**2,dimz)
                    "z": nm, # (S,(N+1)**2,2)
                }
                Coeff_f = self.Decoder_z(input_dec_dict)["x"] # (S,(N+1)**2,6L)
                #=== Transform ====
                if self.config["in_magphase"]:
                    # Coeff_f: (S,(N+1)**2,6L) float -> Coeff:(S,2,(N+1)**2,L) complex
                    Coeff = Coeff_f.view(S,(self.maxto+1)**2,6,-1) # (S,(N+1)**2,6,L)
                    Coeff = Coeff.permute(0,2,1,3)             # (S,6,(N+1)**2,L)
                    Coeff_mag = self.out_z(Coeff[:,0:2,:,:]) # sigmoid to avoid overflow
                    Coeff = th.cat((Coeff_mag,Coeff[:,2:,:,:]), dim=1)
                    Coeff = transformCoeff_MagdBPhase_CNN(Coeff, self.maxto, self.filter_length, self.c_std.to(Coeff.device), self.c_mean.to(Coeff.device),self.config["std_dim"], scale=self.config["trans_scale"]) # 420239: scale==1.25
                else:
                    "[Error] Not implemented yet."
                    sys.exit()
            
            out = th.zeros(S, B, 2, SpF.filter_length, dtype = th.complex64)
            # print(out.shape) #torch.Size([4, 440, 2, 128])
            if use_cuda_forward:
                out = out.cuda()
                # print(Hank[0,:,0])
            if th.any(th.isnan(Coeff)) or th.any(th.isinf(Coeff)):
                if th.any(th.isnan(Coeff)):
                    print('nan is detected in Coeff.')
                if th.any(th.isinf(Coeff)):
                    print('inf is detected in Coeff.')
                Coeff_re = th.nan_to_num(th.real(Coeff),nan=0.0,posinf=0.0,neginf=0.0)
                Coeff_im = th.nan_to_num(th.imag(Coeff),nan=0.0,posinf=0.0,neginf=0.0)
                Coeff = th.complex(Coeff_re, Coeff_im)
            # print(f'Coeff_b:{Coeff.dtype}')
            # Coeff_out = th.zeros(Coeff.shape,dtype=Coeff.dtype,device=Coeff.device)
            mask = th.zeros(Coeff.shape[-2:],dtype=Coeff.dtype,device=Coeff.device)
            for i,n in enumerate(n_vec):
                mask_i = th.from_numpy(self.N_arr_np.astype(np.float32)).clone() >= n
                mask_i = mask_i.to(Coeff.device).to(Coeff.dtype)
                mask[i,:] = mask_i
            # print(mask.shape)
            # print(Coeff.shape)
                
            for i,n in enumerate(n_vec):
                # mask = th.from_numpy(self.N_arr_np.astype(np.float32)).clone() >= n
                # mask = mask.to(Coeff.device).to(Coeff.dtype)
                # mask = mask.to(th.float64)
                if th.any(th.isnan(mask)):
                    print('mask')
                    sys.exit()
                # print(Coeff.shape)
                # print(Hank.shape)
                Coeff_mn = Coeff[:,:,i,:] * mask[None,None,i,:]
                Hank_n = Hank[:,n,:] * mask[None,i,:] 
                # print(Hank_n.shape) # torch.Size([440, 128])
                Harm_mn = Harm[:,i]
                # print(Harm_mn.shape) # torch.Size([440])
                delta = th.mul(Coeff_mn[:,None,:,:], Hank_n[None,:,None,:]) * Harm_mn[None,:,None,None]
                # print(delta.shape) # torch.Size([4,440, 2, 128])
                if th.any(th.isnan(Coeff)):
                    print('Coeff')
                if th.any(th.isnan(Coeff_mn)):
                    print('Coeff_mn')
                if th.any(th.isnan(Hank_n)):
                    print('Hank_n')
                if th.any(th.isnan(Harm_mn)):
                    print('Harm_mn')
                if th.any(th.isnan(delta)):
                    print('nan is detected in "delta".')
                    delta_re = th.nan_to_num(th.real(delta),nan=0.0)
                    delta_im = th.nan_to_num(th.imag(delta),nan=0.0)
                    delta = th.complex(delta_re, delta_im)
                    print(Coeff.dtype)
                    print(Coeff_mn.dtype)
                out += delta
            if th.any(th.isnan(out)):
                print('out')
                sys.exit()
            returns['coeff'] = (Coeff*mask[None,None,:,:].to(Coeff.device)).permute(1,2,3,0)
            # print(returns['coeff'].device)
            # print(Coeff.permute(1,2,3,0).shape) # torch.Size([2, 361, 128, 4])

        if self.config["windowfunc"]:
            # out:torch.Size([4, 440, 2, 128])
            # print(out.shape)
            zeros = th.zeros([out.shape[0],out.shape[1],out.shape[2],1]).to(out.device).to(out.dtype)
            hrtf = th.cat((zeros,out,th.conj(th.flip(out[:,:,:,:-1],dims=(-1,)))), dim=-1)
            # print(hrtf.shape)
            hrir = th.real(th.fft.ifft(th.conj(hrtf), dim=-1))
            # print(hrir.shape)

            #===== window =====
            if self.config['window'] == 'kaiser_half':
                #== kaiser ==
                window = th.kaiser_window(window_length=self.config['fft_length'], periodic=True, beta=10.0).to(hrir.device).to(hrir.dtype)
                # window = th.hann_window(self.config['fft_length'])
                shift = round(self.config['fft_length']/2) #128
                window = th.roll(window,-shift,0)
                window_final = F.pad(input=window[:round(len(window)/2)],pad=(0,round(len(window)/2)))
            elif self.config['window'] == 'square':
                #== square ==
                window_final = th.cat((th.zeros(8),th.ones(120),th.zeros(self.config['fft_length']-128))).to(hrir.device).to(hrir.dtype)

            #===================
            hrir_window = hrir*window_final[None,None,None,:]
            returns['hrir_extra'] = hrir - hrir_window
            hrtf_window = th.conj(th.fft.fft(hrir_window, dim=-1))
            # print(hrtf_window.shape)
            hrtf_window = hrtf_window[:,:,:,1:round(self.config['fft_length']/2)+1]
            if self.config["out_nowindow"]:
                returns['output'] = out.permute(1,2,3,0)
            else:
                returns['output'] = hrtf_window.permute(1,2,3,0)
            returns['output_nowindow'] = out.permute(1,2,3,0)
        else:
            returns['output'] = out.permute(1,2,3,0)
            #torch.Size([440, 2, 128, S])
        # print('----------')
        # sys.exit()
        if 'weight_mean' not in returns.keys():
            bb = len(idx_t_des) if 'idx_t_des' in locals() else B
            returns['weight_mean'] = th.ones(bb)/bb
        ### dammy
        returns['vec'] = vec_cart_gt
        returns['theta'] = theta # zenith
        returns['phi'] = phi # azimuth
        ###
        if 'coeff' not in returns.keys():
            returns['coeff'] = th.zeros(2,(self.maxto+1)**2,L,S, dtype=th.complex64)

        return returns

class HRTFApproxNetwork_FIAE(Net): # Frequency Independent AutoEncoder
    def __init__(self,
                config,
                model_name='hrtf_approx_network',
                use_cuda=True):
        super().__init__(model_name, use_cuda)
        self.config = config
        self.max_f = config["max_frequency"]
        self.maxto = config["max_truncation_order"]
        self.fft_length = config["fft_length"]
        self.filter_length = round(self.fft_length/2)
        self.f_arr = (np.linspace(0, self.max_f, self.filter_length+1))[1:] # frequency bin, (filterlengh,)
    
        #====== model =======================
        #------ pre_dnn ---------------------
        if self.config["in_latent"]:
            modules = []
            for l in range(config["hlayers_latents"]):
                if l == 0: # first layer
                    ch_in = self.filter_length 
                else:
                    ch_in = config["channel_latents"]
                
                if l == config["hlayers_latents"] - 1: # last layer
                    ch_out = config["num_latents"]
                    post_prcs = False
                else:
                    ch_out = config["channel_latents"]
                    post_prcs = True
                modules.extend([nn.Linear(ch_in, ch_out)])
                if post_prcs:
                    modules.extend([
                        nn.LayerNorm(ch_out),
                        nn.ReLU(),
                        nn.Dropout(config["droprate"])
                        ])
            self.pre_nn = nn.Sequential(*modules)
        elif self.config["in_cnn"]:
            modules = []
            L = self.filter_length
            for b in range(config["blocks_cnn"]):
                for l in range(config["hlayers_cnn_pb"]):
                    ch_in  = 1 if b==0 and l==0 else config["channel_cnn"]
                    ch_out = 1 if b==config["blocks_cnn"]-1 and l==config["hlayers_cnn_pb"]-1 else config["channel_cnn"]
                    post_prcs = False if b==config["blocks_cnn"]-1 and l==config["hlayers_cnn_pb"]-1 else True

                    modules.extend([nn.Conv1d(in_channels=ch_in, out_channels=ch_out, kernel_size=config["ks_cnn"], padding='same', bias=True)])
                    if post_prcs:
                        if config["LN_dim"]==[-2,-1]:
                            modules.extend([nn.LayerNorm([ch_out, L])])
                        elif config["LN_dim"]==[-1]:
                            modules.extend([nn.LayerNorm(L)])
                        else:
                            raise NotImplementedError
                        modules.extend([
                            nn.ReLU(),
                            nn.Dropout(config["droprate"])
                        ])
                if not b== config["blocks_cnn"]-1:
                    pad_same = (config["ks_cnn_plg"]-config["stride_cnn"])//2
                    if config["pooling_cnn"] == "avg":
                        modules.extend([nn.AvgPool1d(kernel_size=config["ks_cnn_plg"], stride=config["stride_cnn"], padding=pad_same)])
                    elif config["pooling_cnn"] == "max":
                        modules.extend([nn.MaxPool1d(kernel_size=config["ks_cnn_plg"], stride=config["stride_cnn"], padding=pad_same)])
                    elif config["pooling_cnn"] == "stcv":
                        modules.extend([nn.Conv1d(in_channels=ch_out, out_channels=ch_out, kernel_size=config["ks_cnn_plg"], padding=pad_same, stride=config["stride_cnn"], bias=True)])
                    elif config["pooling_cnn"] == None:
                        pass
                    else:
                        raise NotImplementedError
                    if not config["pooling_cnn"] == None:
                        L //= config["stride_cnn"]
            self.pre_cnn = nn.Sequential(*modules)
        elif self.config["in_cnn_ch"]:
            modules = []
            L = self.filter_length
            ch_in = config["channel_inout"]
            if self.config["use_freq_for_cnn_ch"]:
                raise NotImplementedError
            if config["channel_inout"] > 1:
                modules.extend([nn.Conv1d(in_channels=1, out_channels=ch_in, kernel_size=config["ks_cnn"], padding='same', bias=True)])

            for b in range(config["blocks_cnn"]):
                for l in range(config["hlayers_cnn_pb"]):
                    post_prcs = False if b==config["blocks_cnn"]-1 and l==config["hlayers_cnn_pb"]-1 else True
                    ch_out = config["stride_cnn"]*ch_in
                    modules.extend([nn.Conv1d(in_channels=ch_in, out_channels=ch_out, kernel_size=config["ks_cnn"], padding='same', bias=True)])
                    if post_prcs:
                        if config["LN_dim"]==[-2,-1]:
                            modules.extend([nn.LayerNorm([ch_out, L])])
                        elif config["LN_dim"]==[-1]:
                            modules.extend([nn.LayerNorm(L)])
                        else:
                            raise NotImplementedError
                        modules.extend([
                            nn.ReLU(),
                            nn.Dropout(config["droprate"])
                        ])
                # if not b== config["blocks_cnn"]-1:
                pad_same = (config["ks_cnn_plg"]-config["stride_cnn"])//2
                if config["pooling_cnn"] == "avg":
                    modules.extend([nn.AvgPool1d(kernel_size=config["ks_cnn_plg"], stride=config["stride_cnn"], padding=pad_same)])
                elif config["pooling_cnn"] == "max":
                    modules.extend([nn.MaxPool1d(kernel_size=config["ks_cnn_plg"], stride=config["stride_cnn"], padding=pad_same)])
                elif config["pooling_cnn"] == "stcv":
                    modules.extend([nn.Conv1d(in_channels=ch_out, out_channels=ch_out, kernel_size=config["ks_cnn_plg"], padding=pad_same, stride=config["stride_cnn"], bias=True)])
                elif config["pooling_cnn"] == None:
                    pass
                else:
                    raise NotImplementedError
                if not config["pooling_cnn"] == None:
                    L //= config["stride_cnn"]

                ch_in = ch_out
            self.pre_cnn = nn.Sequential(*modules)
        
        #------ Fourier Feature Mapping -----
        if len(self.config["data_kind_ffm"]) > 0:
            # self.ffm = {}
            for data_kind in self.config["data_kind_ffm"]:
                # self.ffm[data_kind] = FourierFeatureMapping(num_features=self.config["num_ff"][data_kind], dim_data=self.config["dim_data_hyper"][data_kind], trainable=self.config["ffm_trainable"])
                exec(f'self.ffm_{data_kind} = FourierFeatureMapping(num_features=self.config["num_ff"][data_kind], dim_data=self.config["dim_data_hyper"][data_kind], trainable=self.config["ffm_trainable"])')

        # if self.config["use_ffm_for_hyper_cartpos"]:
        #     self.ffm_cartpos = FourierFeatureMapping(self.config["num_ff_cartpos"], 3, trainable=self.config["ffm_trainable"])
        # if self.config["use_ffm_for_hyper_freq"]:
        #     self.ffm_freq = FourierFeatureMapping(self.config["num_ff_freq"], 1, trainable=self.config["ffm_trainable"])
        # if self.config["use_ffm_for_hyper_Bp"]:
        #     self.ffm_Bp = FourierFeatureMapping(self.config["num_ff_Bp"], 1, trainable=self.config["ffm_trainable"])
        

        #------ Encoder ---------------------
        # en_0
        # x:(2S,L,B',1)
        # z:(2S,L,B', 3or4)
        if self.config["hlayers_En_0"]>0:
            modules = []
            
            for l in range(config["hlayers_En_0"]):
                if l == 0: # first layer
                    ch_in = 1
                else:
                    ch_in = config["channel_En_0"]

                if l == config["hlayers_En_0"] - 1: # last layer
                    ch_out = 1 if not config["aggregation_mean"] else config["dim_z"]
                    post_prcs = False
                else:
                    ch_out = config["channel_En_0"]
                    post_prcs = True
                if self.config["hyperlinear_en_0"]:
                    input_size_en_0 = 0
                    for data_kind in self.config["data_kind_hyper_en"]:
                        if self.config["data_kind_interp"]==['ITD'] and data_kind=='Freq':
                            continue
                        elif data_kind in self.config["data_kind_ffm"]:
                            input_size_en_0 += self.config["num_ff"][data_kind] * 2
                        else:
                            input_size_en_0 += self.config["dim_data_hyper"][data_kind]
                    if len(self.config["data_kind_interp"]) > 1:
                        input_size_en_0 += 1 # delta
                    # if self.config["use_ffm_for_hyper_cartpos"]:
                    #     input_size_en_0 = self.config["num_ff_cartpos"] * 2
                    # else:
                    #     input_size_en_0 = 3
                    # if self.config["use_freq_for_hyper_en"]:
                    #     if self.config["use_ffm_for_hyper_freq"]:
                    #         input_size_en_0 += self.config["num_ff_freq"] * 2
                    #     else:
                    #         input_size_en_0 += 1
                    # if self.config["use_Bp_for_hyper_en"]:
                    #     if self.config["use_ffm_for_hyper_Bp"]:
                    #         input_size_en_0 += self.config["num_ff_Bp"] * 2
                    #     else:
                    #         input_size_en_0 += 1
                    modules.extend([
                        HyperLinearBlock(ch_in=ch_in, ch_out=ch_out, ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"], input_size=input_size_en_0, post_prcs=post_prcs),
                    ])
                elif self.config["hyperconv_en_0"]:
                    input_size_en_0 = 3

                    if self.config["use_Bp_for_hyper_en"]:
                        input_size_en_0 += 1
                    if self.config["LN_dim"] == [-2,-1]:
                        normalized_shape = [ch_out, self.filter_length]
                    elif self.config["LN_dim"] == [-1]:
                        normalized_shape = [self.filter_length]
                    elif self.config["LN_dim"] == [-2]:
                        normalized_shape = [ch_out]
                    else:
                        raise NotImplementedError

                    modules.extend([
                        HyperConv1dBlock(ch_in=ch_in, ch_out=ch_out, kernel_size=config["hyperconv_en_0_ks"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"], input_size=input_size_en_0, post_prcs=post_prcs, use_freq=config["use_freq_for_hyper_en"], normalized_dim=self.config["LN_dim"], normalized_shape=normalized_shape)
                    ])
                elif self.config["hyperconv_FD_en_0"]:
                    input_size_en_0 = 3
                    if self.config["use_freq_for_hyper_en"]:
                        input_size_en_0 += self.config["hyperconv_en_0_ks"]
                    modules.extend([
                        HyperConv_FD_Block(num_aux=input_size_en_0, ch_in=ch_in, ch_out=ch_out, kernel_size=config["hyperconv_en_0_ks"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"], use_f_aux=config["use_freq_for_hyper_en"], pad=config["hyperconv_en_0_pad"],post_prcs=post_prcs, transpose=False),
                    ])
                else:
                    raise NotImplementedError

            self.en_0 = nn.Sequential(*modules)
        # x:(2S,L,B',1)

        if not self.config["aggregation_mean"]:
            # en_1
            # in_features for 1st hypernet
            modules = []
            if self.config["use_RSH_for_hyper_en"]:
                insize_hyper_en = (self.maxto+1)**2
            else:
                insize_hyper_en = 3
            if self.config["use_freq_for_hyper_en"]:
                insize_hyper_en += 1 # [x,y,z,freq] or [..., freq]
            
            if config["hlayers_En_z"] == 0:
                ch_out = config["dim_z"]
            else:
                ch_out = config["channel_En_z"]
            modules.extend([
                HyperLinear_FIAE(ch_out=ch_out, input_size=insize_hyper_en, ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"],use_bias=config["use_bias_FIAE_en"], pinv=config["pinv_FIAE_en"], reg_w = config["reg_w"], identity=config["hyper_identity_en"], freq_in=config["use_freq_for_hyper_en"], reg_mat_learn=config["reg_mat_learn"], reg_mat_base=config["reg_mat_base"], relax=config["cstr_relax"], rel_l=config["rel_l"])
            ])
            if not self.config["en_1_linear"]:
                modules.extend([
                    nn.LayerNorm(ch_out),
                    nn.ReLU(),
                    nn.Dropout(config["droprate"])
                ])
            self.en_1 = nn.Sequential(*modules)
            # x:(S,L,ch_out), z:(S,L,B,input_size)

        if self.config["pos_all_en_attn"]:
            #== Key ===
            modules = []
            input_size_en_attn_k = 4 if self.config["use_freq_for_hyper_en"] else 3
            for l in range(config["hlayers_En_attn_k"]):
                if l == 0: # first layer
                    ch_in = input_size_en_attn_k
                else:
                    ch_in = config["channel_En_attn_k"]

                if l == config["hlayers_En_attn_k"] - 1: # last layer
                    ch_out = config["dim_z"]
                    post_prcs = False
                else:
                    ch_out = config["channel_En_attn_k"]
                    post_prcs = True

                modules.extend([nn.Linear(ch_in, ch_out)])
                if post_prcs:
                    modules.extend([
                        nn.LayerNorm(ch_out),
                        nn.ReLU(),
                        nn.Dropout(config["droprate"])])
            for layers in modules:
                if hasattr(layers, "weight"):
                    nn.init.normal_(layers.weight, mean=0.0, std=0.01)
                if hasattr(layers, "bias"):
                    nn.init.constant_(layers.bias, 0.0)
            self.en_attn_k = nn.Sequential(*modules)
            # in: (L,B',4) if use_freq_for_hyper_en else (B',3)
            # out: (L,B',d_z) if use_freq_for_hyper_en else (B',d_z)

            if self.config["pos_all_en_attn_gram"]:
                pass
            else:
                #== Query ===
                modules = []
                input_size_en_attn_q = 4 if self.config["use_freq_for_hyper_en"] else 3
                for l in range(config["hlayers_En_attn_q"]):
                    if l == 0: # first layer
                        ch_in = input_size_en_attn_q
                    else:
                        ch_in = config["channel_En_attn_q"]

                    if l == config["hlayers_En_attn_q"] - 1: # last layer
                        ch_out = config["dim_z"]
                        post_prcs = False
                    else:
                        ch_out = config["channel_En_attn_q"]
                        post_prcs = True

                    modules.extend([nn.Linear(ch_in, ch_out)])
                    if post_prcs:
                        modules.extend([
                            nn.LayerNorm(ch_out),
                            nn.ReLU(),
                            nn.Dropout(config["droprate"])])
                for layers in modules:
                    if hasattr(layers, "weight"):
                        nn.init.normal_(layers.weight, mean=0.0, std=0.01)
                    if hasattr(layers, "bias"):
                        nn.init.constant_(layers.bias, 0.0)
                self.en_attn_q = nn.Sequential(*modules)
                # in: (L,B',4) if use_freq_for_hyper_en else (B',3)
                # out: (L,B',d_z) if use_freq_for_hyper_en else (B',d_z)

        # en_2
        if config["hlayers_En_z"] > 0:
            modules = []
            for l in range(config["hlayers_En_z"]):
                if l == config["hlayers_En_z"] - 1: # last layer
                    ch_out = config["dim_z"]
                else:
                    ch_out = config["channel_En_z"]
                if self.config["use_freq_for_hyper"]:
                    # x:(S,L,ch_out), z:(S,L,1)
                    modules.extend([
                        HyperLinearBlock(ch_in=config["channel_En_z"], ch_out=ch_out, ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"], input_size=1),
                    ])
                else:
                    # input:(S,L,ch_out)
                    modules.extend([
                        nn.Linear(in_features=config["channel_En_z"], out_features=ch_out, bias=True),
                        nn.LayerNorm(ch_out),
                        nn.ReLU(),
                        nn.Dropout(config["droprate"])
                    ])
            # x or output:(S,L,dim_z)
            self.en_2 = nn.Sequential(*modules)
        #------ Encoder end ---------------------

        #------ Mid-CNN -------------------------
        if config["use_mid_conv"]:
            modules = []
            # - - - - - -
            ch_first = config["dim_z"]
            if self.config["use_freq_for_mid_conv"]:
                ch_first += 1
            L = config['fft_length']//2
            for l in range(config["hlayers_mid_conv"]):
                if l == 0: # first layer
                    ch_in = ch_first
                else:
                    ch_in = config["channel_mid_conv"]

                ch_out = config["channel_mid_conv"]
                post_prcs = True

                modules.extend([
                    nn.Conv1d(in_channels=ch_in, out_channels=ch_out, kernel_size=config["ks_mid_conv"], stride=1, padding=0, bias=True)
                ])
                L -= config["ks_mid_conv"] -1 
                if post_prcs:
                    modules.extend([
                        nn.LayerNorm(L),
                        nn.ReLU(),
                        nn.Dropout(config["droprate"])
                    ])
            # - - - - - -
            for l in range(config["hlayers_mid_conv"]):
                ch_in = config["channel_mid_conv"]
                
                if l == config["hlayers_mid_conv"]-1: # last layer
                    ch_out = config["dim_z"]
                    post_prcs = False
                else:
                    config["dim_z"] = config["channel_mid_conv"]
                    post_prcs = True
                
                modules.extend([
                    nn.ConvTranspose1d(in_channels=ch_in, out_channels=ch_out, kernel_size=config["ks_mid_conv"], stride=1, padding=0, output_padding=0, bias=True)
                ])
                L += config["ks_mid_conv"] -1
                if post_prcs:
                    modules.extend([
                        nn.LayerNorm(L),
                        nn.ReLU(),
                        nn.Dropout(config["droprate"])
                    ])
            # - - - - - -
            self.mid_cnn = nn.Sequential(*modules)
        
        if config["use_mid_conv_simple"]:
            modules = []
            ch_mid_in = config["ch_mid_conv_simple"]
            if self.config["use_freq_for_mid_conv"]:
                ch_mid_in += 1
            modules.extend([
                nn.Conv1d(in_channels=ch_mid_in, out_channels=config["ch_mid_conv_simple"], kernel_size=config["ks_mid_conv"], stride=1, padding='same', bias=True)
            ])
            self.mid_cnn = nn.Sequential(*modules)
                
        #------ Mid-CNN end ---------------------

        #------ Mid-Linear (freq_compress) -------------------------
        if config["use_mid_linear"]:
            # modules = [nn.Linear(in_features=config['fft_length']//2, out_features=1, bias=True)]
            # self.mid_linear = nn.Sequential(*modules)

            LLL = 0
            for data_kind in self.config['data_kind_interp']:
                if data_kind in ['HRTF_mag','HRTF']:
                    LLL += self.filter_length * 2
                elif data_kind in ['HRIR']:
                    LLL += self.filter_length * 4
                elif data_kind in ['ITD']:
                    LLL += 1
            mid_linear = nn.Linear(in_features=LLL, out_features=LLL, bias=True)
            mid_linear.weight = th.nn.Parameter(th.eye(LLL,LLL) + th.normal(0,1e-3,(LLL, LLL)))
            mid_linear.bias = th.nn.Parameter(th.normal(0,1e-3,(LLL, )))
            modules = [mid_linear]
            self.mid_linear = nn.Sequential(*modules)

        #------ Mid-Linear end ---------------------

        #------ Decoder -------------------------
        if config["hlayers_De_z"] > 0:
            modules = []
            for l in range(config["hlayers_De_z"]):
                if l == 0: # last layer
                    ch_in = config["dim_z"]
                else:
                    ch_in = config["channel_De_z"]
                if self.config["use_freq_for_hyper"]:
                    # x:(S,L,ch_out), z:(S,L,1)
                    modules.extend([
                        HyperLinearBlock(ch_in=ch_in, ch_out=config["channel_De_z"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"], input_size=1),
                    ])
                else:
                    # input:(S,L,ch_out)
                    modules.extend([
                        nn.Linear(in_features=ch_in, out_features=config["channel_De_z"], bias=True),
                        nn.LayerNorm(config["channel_De_z"]),
                        nn.ReLU(),
                        nn.Dropout(config["droprate"])
                    ])
                # x or output:(S,L,ch_out)
            self.de_1 = nn.Sequential(*modules)

        if config["hlayers_De_z"] == 0:
            ch_in = config["dim_z"]
        else:
            ch_in = config["channel_De_z"]
        
        if not self.config["de_2_skip"]:
            modules = []
            if self.config["use_RSH_for_hyper_de"]:
                insize_hyper_de = (self.maxto+1)**2
            else:
                insize_hyper_de = 3
            if self.config["use_freq_for_hyper_de"]:
                insize_hyper_de += 1 # [x,y,z,freq] or [..., freq]
            modules.extend([
                HyperLinear_FIAE_Trans(ch_in=ch_in, input_size=insize_hyper_de, ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"],use_bias=config["use_bias_FIAE_de"], identity=config["hyper_identity_de"], freq_in=config["use_freq_for_hyper_de"])
            ])
            # modules.extend([
            #     nn.LayerNorm(ch_out),
            #     nn.ReLU(),
            #     nn.Dropout(config["droprate"])
            # ])
            self.de_2 = nn.Sequential(*modules)
            # x:(S,L,B)
        else:
            # modules = [nn.Identity()]
            # self.de_2 = nn.Sequential(*modules)
            pass
        
        if self.config["hlayers_De_-1"]>0:
            # x:(S,L,ch_out)->(S,L,B,ch_out)->(S,L,B,1)->(S,L,B)
            modules = []
            for l in range(config["hlayers_De_-1"]):
                if l == 0: # first layer
                    if not self.config["de_2_skip"]:
                        ch_in = 1
                    else:
                        ch_in = config["dim_z"]
                else:
                    ch_in = config["channel_De_-1"]

                if l == config["hlayers_De_-1"] - 1: # last layer
                    ch_out = 1
                    post_prcs = False
                else:
                    ch_out = config["channel_De_-1"]
                    post_prcs = True

                if self.config["hyperlinear_de_-1"]:
                    
                    input_size_de_m1 = 0
                    for data_kind in self.config["data_kind_hyper_de"]:
                        if self.config["data_kind_interp"]==['ITD'] and data_kind=='Freq':
                            continue
                        elif data_kind in self.config["data_kind_ffm"]:
                            input_size_de_m1 += self.config["num_ff"][data_kind] * 2
                        else:
                            input_size_de_m1 += self.config["dim_data_hyper"][data_kind]
                    if len(self.config["data_kind_interp"]) > 1:
                        input_size_de_m1 += 1 # delta

                    
                    # if self.config["use_ffm_for_hyper_cartpos"]:
                    #     input_size_de_m1 = self.config["num_ff_cartpos"] * 2
                    # else:
                    #     input_size_de_m1 = 3
                    # if self.config["use_freq_for_hyper_de"]:
                    #     if self.config["use_ffm_for_hyper_freq"]:
                    #         input_size_de_m1 += self.config["num_ff_freq"] * 2
                    #     else:
                    #         input_size_de_m1 += 1
      
                    modules.extend([
                        HyperLinearBlock(ch_in=ch_in, ch_out=ch_out, ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"], input_size=input_size_de_m1, post_prcs=post_prcs),
                    ])
                elif self.config["hyperconv_de_-1"]:
                    input_size_de_m1 = 3

                    if self.config["use_Bp_for_hyper_en"]:
                        input_size_en_0 += 1
                    if self.config["LN_dim"] == [-2,-1]:
                        normalized_shape = [ch_out, self.filter_length]
                    elif self.config["LN_dim"] == [-1]:
                        normalized_shape = [self.filter_length]
                    elif self.config["LN_dim"] == [-2]:
                        normalized_shape = [ch_out]
                    else:
                        raise NotImplementedError

                    modules.extend([
                        HyperConvTranspose1dBlock(ch_in=ch_in, ch_out=ch_out, kernel_size=config["hyperconv_de_-1_ks"],ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"], input_size=input_size_de_m1, post_prcs=post_prcs, use_freq=config["use_freq_for_hyper_de"], normalized_dim=self.config["LN_dim"], normalized_shape=normalized_shape),
                    ])
                elif self.config["hyperconv_FD_de_-1"]:
                    input_size_de_m1 = 3
                    if self.config["use_freq_for_hyper_de"]:
                        input_size_de_m1 += self.config["hyperconv_de_-1_ks"]
                    modules.extend([
                        HyperConv_FD_Block(num_aux=input_size_en_0, ch_in=ch_in, ch_out=ch_out, kernel_size=config["hyperconv_de_-1_ks"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"], use_f_aux=config["use_freq_for_hyper_de"], pad=config["hyperconv_de_-1_pad"],post_prcs=post_prcs, transpose=True),
                    ])
            self.de_m1 = nn.Sequential(*modules)
            # x:(S,L,ch_out)->(S,L,B,ch_out)->(S,L,B,1)->(S,L,B)
        
        #------- post_nn --------------
        if self.config["out_latent"]:
            modules = []
            for l in range(config["hlayers_latents"]):
                if l == 0: # first layer
                    ch_in = config["num_latents"]
                else:
                    ch_in = config["channel_latents"]
                
                if l == config["hlayers_latents"] - 1: # last layer
                    ch_out = self.filter_length 
                    post_prcs = False
                else:
                    ch_out = config["channel_latents"]
                    post_prcs = True
                modules.extend([nn.Linear(ch_in, ch_out)])
                if post_prcs:
                    modules.extend([
                        nn.LayerNorm(ch_out),
                        nn.ReLU(),
                        nn.Dropout(config["droprate"])
                        ])
            self.post_nn = nn.Sequential(*modules)
        elif self.config["out_cnn"]:
            modules = []
            L = self.filter_length // (config["stride_cnn"]**(config["blocks_cnn"]-1))
            for b in range(config["blocks_cnn"]):
                for l in range(config["hlayers_cnn_pb"]):
                    ch_in  = 1 if b==0 and l==0 else config["channel_cnn"]
                    ch_out = 1 if b==config["blocks_cnn"]-1 and l==config["hlayers_cnn_pb"]-1 else config["channel_cnn"]
                    post_prcs = False if b==config["blocks_cnn"]-1 and l==config["hlayers_cnn_pb"]-1 else True

                    modules.extend([nn.ConvTranspose1d(in_channels=ch_in, out_channels=ch_out, kernel_size=config["ks_cnn"], padding=(config["ks_cnn"]-1)//2, bias=True)])
                    if post_prcs:
                        if config["LN_dim"]==[-2,-1]:
                            modules.extend([nn.LayerNorm([ch_out, L])])
                        elif config["LN_dim"]==[-1]:
                            modules.extend([nn.LayerNorm(L)])
                        else:
                            raise NotImplementedError
                        modules.extend([
                            nn.ReLU(),
                            nn.Dropout(config["droprate"])
                        ])
                if not b==config["blocks_cnn"]-1:
                    if not config["pooling_cnn"]==None:
                        modules.extend([nn.Upsample(scale_factor=config["stride_cnn"])])
                        L *= config["stride_cnn"]
            self.post_cnn = nn.Sequential(*modules)
        elif self.config["out_cnn_ch"]:
            modules = []
            
            ch_in = config["channel_inout"]*config["stride_cnn"]**config["blocks_cnn"]
            L = self.filter_length // ch_in
            for b in range(config["blocks_cnn"]):
                if not config["pooling_cnn"]==None:
                    modules.extend([nn.Upsample(scale_factor=config["stride_cnn"])])
                    L *= config["stride_cnn"]
                for l in range(config["hlayers_cnn_pb"]):
                    post_prcs = False if b==config["blocks_cnn"]-1 and l==config["hlayers_cnn_pb"]-1 else True
                    ch_out = ch_in // config["stride_cnn"]
                    modules.extend([nn.ConvTranspose1d(in_channels=ch_in, out_channels=ch_out, kernel_size=config["ks_cnn"], padding=(config["ks_cnn"]-1)//2, bias=True)])
                    if post_prcs:
                        if config["LN_dim"]==[-2,-1]:
                            modules.extend([nn.LayerNorm([ch_out, L])])
                        elif config["LN_dim"]==[-1]:
                            modules.extend([nn.LayerNorm(L)])
                        else:
                            raise NotImplementedError
                        modules.extend([
                            nn.ReLU(),
                            nn.Dropout(config["droprate"])
                        ])
                # if not b==config["blocks_cnn"]-1:
                #     if not config["pooling_cnn"]==None:
                #         modules.extend([nn.Upsample(scale_factor=config["stride_cnn"])])
                #         L *= config["stride_cnn"]
                ch_in = ch_out
            if config["channel_inout"] > 1:
                modules.extend([nn.Conv1d(in_channels=ch_out, out_channels=1, kernel_size=config["ks_cnn"], padding=(config["ks_cnn"]-1)//2, bias=True)])
            self.post_cnn = nn.Sequential(*modules)

    # HRTF_mag only
    def encode(self, data):
        returns = {}
        db_name = data["db_name"][0] # I have no idea why db_name is a list
        # ic(db_name)

        LL = self.filter_length # Frequency bins for one ear

        S,B,_ = data["SrcPos"].shape
        # ic(data["SrcPos"])
        
        #===v sampling v====
        # for full dataset just use idx_mes_pos
        idx_mes_pos = range(0, B) 
        
        returns["idx_mes_pos"] = idx_mes_pos
        B_mes = len(idx_mes_pos)
        data_mes = {}

        #=== Standardize ====
        for data_kind in self.config['data_kind_interp']:
            if data_kind in ['HRTF_mag','HRTF','HRIR','ITD']:
                data[data_kind] = (data[data_kind] - self.config["Val_Standardize"][db_name][data_kind]["mean"]) / (self.config["Val_Standardize"][db_name][data_kind]["std"])
        data['SrcPos_Cart'] = data['SrcPos_Cart']/(self.config["SrcPos_Cart_norm"])
        data['Freq'] = (th.arange(self.filter_length+1)[1:]/self.filter_length).to(data['HRTF_mag'].device).unsqueeze(-1)

        for k in data:
            if k in ['SrcPos','SrcPos_Cart','HRTF','HRTF_mag','HRIR','ITD']:
                assert data[k].shape[1] == B
                data_mes[k] = data[k][:,idx_mes_pos]
        #===^ sampling ^====

        B_mes_norm = self.config["Bp_norm"] if "Bp_norm" in self.config.keys() else B
        data['B_mes'] = th.tensor([B_mes/B_mes_norm]).to(data['HRTF_mag'].device).unsqueeze(-1)
        
        if self.config["use_lr_aug"]:
            lr_flip_tensor = th.tensor([1,-1,1], device=data['SrcPos_Cart'].device)
            data['SrcPos_Cart_lr_flip'] = data['SrcPos_Cart'] * lr_flip_tensor[None,None,:]
            data_mes['SrcPos_Cart_lr_flip'] = data_mes['SrcPos_Cart'] * lr_flip_tensor[None,None,:]
            # (S,B,3)
        else:
            raise NotImplementedError
        
        for data_kind in self.config["data_kind_ffm"]:
            for data_str in ['data','data_mes']:
                if data_kind in eval(data_str):
                    exec(f'eval(data_str)[data_kind] = self.ffm_{data_kind}(eval(data_str)[data_kind])')
                    if data_kind == 'SrcPos_Cart':
                        exec(f"eval(data_str)['SrcPos_Cart_lr_flip'] = self.ffm_{data_kind}(eval(data_str)['SrcPos_Cart_lr_flip'])")

        device = data['HRTF_mag'].device
        # ic(data_mes['HRTF_mag'].shape)
        hyper_en_x = th.zeros(S,B_mes,0,1, device=device, dtype=th.float32)
        hyper_en_x = th.cat((hyper_en_x, data_mes['HRTF_mag'][:,:,0,:,None], data_mes['HRTF_mag'][:,:,1,:,None]), dim=2) # (S,B_mes,2L,1)
        
        hyper_en_z = th.zeros(S,B_mes,2*LL,0, device=device, dtype=th.float32)

        # left and right ear concatenated here
        for data_kind in self.config["data_kind_hyper_en"]:
            if data_kind in ['SrcPos_Cart']:
                hyper_en_z = th.cat((hyper_en_z, th.cat((data_mes[data_kind].unsqueeze(2).tile(1,1,LL,1), data_mes[f'{data_kind}_lr_flip'].unsqueeze(2).tile(1,1,LL,1)), dim=2)), dim=3) # (S,B_mes,2*LL,3 or num_ff)
            elif data_kind in ['Freq']:
                hyper_en_z = th.cat((hyper_en_z, data[data_kind].tile(2,1)[None,None,:,:].tile(S,B_mes,1,1)), dim=3)  # (S,B_mes,2*LL,1 or num_ff)
            elif data_kind in ['B_mes']:
                hyper_en_z = th.cat((hyper_en_z, data[data_kind][None,None,:,:].tile(S,B_mes,2*LL,1)), dim=3) # (S,B_mes,2*LL,1 or num_ff)
        
        # ic(hyper_en_x.shape)
        # ic(hyper_en_z.shape)

        # ic(hyper_en_x.device)
        # ic(hyper_en_z.device)
        
        latents = self.en_0({
            "x": hyper_en_x, # (S,B_mes, 1 or 2LL or 2LL+1, 1)
            "z": hyper_en_z, # (S,B_mes, 1 or 2LL or 2LL+1, *)
        })["x"]
        # (S,B_mes, 1 or 2LL or 2LL+1, d)
        
        # ic(latents.shape)
        if self.config["use_mid_linear"]:
            latents = self.mid_linear(latents.permute(0,1,3,2)).permute(0,1,3,2)
            # ic(latents.shape)

        returns["z_bm"] = latents
        latents = th.mean(latents, dim=self.config["mid_mean_dim"], keepdim=True)  # (S, 1, 1 or 2LL or 2LL+1, d)
        returns["z"] = latents ## Latent variables
        # ic(latents.shape, latents)
        ## HRTF_mag: (S (batch dimention), 1 (source positions), 2LL (frequency bins LR), d)

        return returns
    
    # HRTF_mag only
    def decode(self, data, latents):
        returns = {}
        db_name = data["db_name"][0]

        LL = self.filter_length # Frequency bins for one ear
        S,B,_ = data["SrcPos"].shape
        LLL = latents.shape[2]

        device = data['HRTF_mag'].device

        if self.config["mid_mean_dim"] == (1,):
            latents = latents.tile(1,B,1,1)
        elif self.config["mid_mean_dim"] == (1,2):
            latents = latents.tile(1,B,LLL,1)

        # latents = latents.unsqueeze(1).tile(1,B,1,1) # (S, B, 1 or 2LL or 2LL+1, d)
        
        hyper_de_z = th.zeros(S,B,2*LL,0, device=device, dtype=th.float32)
        for data_kind in self.config["data_kind_hyper_en"]:
            if data_kind in ['SrcPos_Cart']:
                hyper_de_z = th.cat((hyper_de_z, th.cat((data[data_kind].unsqueeze(2).tile(1,1,LL,1), data[f'{data_kind}_lr_flip'].unsqueeze(2).tile(1,1,LL,1)), dim=2)), dim=3) # (S,B,2*LL,3 or num_ff)
            elif data_kind in ['Freq']:
                hyper_de_z = th.cat((hyper_de_z, data[data_kind].tile(2,1)[None,None,:,:].tile(S,B,1,1)), dim=3)  # (S,B,2*LL,1 or num_ff)

        out_f = self.de_m1({
            "x": latents,    # (S, B, 1 or 2LL or 2LL+1, d)
            "z": hyper_de_z, # (S, B, 1 or 2LL or 2LL+1, *)
        })["x"]

        # ic(out_f.shape)

        data_kind = self.config["data_kind_interp"][0]
        returns[data_kind] = th.cat((out_f[:,:,None,:LL,0], out_f[:,:,None,LL:,0]), dim=2) # (S,B,2,LL)
        
        for data_kind in self.config["data_kind_interp"]:
            returns[data_kind] = returns[data_kind] * self.config["Val_Standardize"][db_name][data_kind]["std"] + self.config["Val_Standardize"][db_name][data_kind]["mean"]
            
        return returns

    def forward(self, data, mode='train'):
        '''
        :param: data: dict contains
            'SrcPos':       (S,B,3) torch.float32 [r,phi,theta]
            'SrcPos_Cart':  (S,B,3) torch.float32 [x,y,z]
            'HRTF':         (S,B,2,L,2) torch.float32 not implemented yet
            'HRTF_mag':     (S,B,2,L) torch.float32
            'HRIR':         (S,B,2,2L) torch.float32
            'ITD':          (S,B) torch.float32
            'db_name':      list of str
            'sub_idx':      (S) torch.int64

        :return: out: 
            - HRTF produce as a (B, 4L, S) complex tensor
                4L ... [l_re, l_im, r_re, r_im]
            - HRTF's magnitude as a (B,2,L,S) float tensor
            - HRIR as a (B,2,L',S) float tensor
            - ITDs as a (B, S) float tensor
        '''
        if True:
            returns = {}
            db_name = data["db_name"][0]
            # for k in data:
            #     ic(k)
            #     if hasattr(data[k], 'shape'):
            #         ic(data[k].shape)
            #         ic(data[k].dtype)
            
            if 'HRTF' in self.config['data_kind_interp']:
                raise NotImplementedError

            for data_kind in self.config['data_kind_interp']:
                if data_kind in ['HRTF_mag','HRTF']:
                    LL = self.filter_length # Frequency bins for one ear
                elif data_kind in ['HRIR']:
                    LL = self.filter_length * 2

            S,B,_ = data["SrcPos"].shape
            # ic(data["SrcPos"])
            # ic(data["SrcPos"].shape)
            
            #===v sampling v====
            # if self.config["random_sample"] and mode == 'train':
            #     perm = th.randperm(B)
            #     idx_mes_pos = perm[:self.config["num_pts"]]
            # elif self.config["lap_smp"]:
            #     idx_mes_pos = self.config["lap_smp_idx"][db_name][self.config["num_pts"]]['idx']
            # elif self.config["pln_smp"]:
            #     idx_mes_pos = plane_sample(pts=data['SrcPos_Cart'][0], axes=self.config["pln_smp_axes"], thr=0.01)
            # elif self.config["pln_smp_paral"]:
            #     idx_mes_pos = parallel_planes_sample(pts=data['SrcPos_Cart'][0], values=th.tensor([-.5,0.0,.5]).to(data['SrcPos'].device)*(data['SrcPos'][0,0,0]), axis=self.config['pln_smp_paral_axis'], thr=0.01)
            # elif self.config["num_pts"] >= min(B,362):

            # for full dataset just use idx_mes_pos
            idx_mes_pos = range(0, B) 

            # elif self.config["random_sample"] and mode == 'train':
            #     perm = th.randperm(B)
            #     idx_mes_pos = perm[:self.config["num_pts"]]
            # elif self.config["num_pts"] >= min(B,362):
            #     idx_mes_pos = range(0,B) 
            # elif self.config["num_pts"] < 9:
            #     idx_mes_pos = aprox_reg_poly(pts=data['SrcPos_Cart'][0]/(data['SrcPos'][0,0,0]), num_pts=self.config["num_pts"], db_name=db_name)
            # else:
            #     self.t = round(self.config["num_pts"]**0.5-1)
            #     idx_mes_pos = aprox_t_des(pts=data['SrcPos_Cart'][0]/(data['SrcPos'][0,0,0]), t=self.t, plot=False, db_name=db_name)
            
            # returns["idx_mes_pos"] = idx_mes_pos
            B_mes = len(idx_mes_pos)
            data_mes = {}

            #=== Standardize ====
            for data_kind in self.config['data_kind_interp']:
                if data_kind in ['HRTF_mag','HRTF','HRIR','ITD']:
                    device = data[data_kind].device
                    # ic(data[data_kind].shape)
                    data[data_kind] = (data[data_kind] - self.config["Val_Standardize"][data_kind][db_name]["mean"].to(device)) / (self.config["Val_Standardize"][data_kind][db_name]["std"].to(device))
            data['SrcPos_Cart'] = data['SrcPos_Cart']/(self.config["SrcPos_Cart_norm"])
            data['Freq'] = (th.arange(self.filter_length+1)[1:]/self.filter_length).to(data['HRTF_mag'].device).unsqueeze(-1)
            # data['Freq'] = (th.arange(self.filter_length+1)[1:]/(self.filter_length/2)-1).to(data['HRTF_mag'].device).unsqueeze(-1)

            for k in data:
                if k in ['SrcPos','SrcPos_Cart','HRTF','HRTF_mag','HRIR','ITD']:
                    assert data[k].shape[1] == B
                    data_mes[k] = data[k][:,idx_mes_pos]
            #===^ sampling ^====
            

            B_mes_norm = self.config["Bp_norm"] if "Bp_norm" in self.config.keys() else B
            data['B_mes'] = th.tensor([B_mes/B_mes_norm]).to(data['HRTF_mag'].device).unsqueeze(-1)
            # data['B_mes'] = th.tensor([B_mes/B_mes_norm*2-1]).to(data['HRTF_mag'].device).unsqueeze(-1)
            
            if self.config["use_lr_aug"]:
                lr_flip_tensor = th.tensor([1,-1,1], device=data['SrcPos_Cart'].device)
                data['SrcPos_Cart_lr_flip'] = data['SrcPos_Cart'] * lr_flip_tensor[None,None,:]
                data_mes['SrcPos_Cart_lr_flip'] = data_mes['SrcPos_Cart'] * lr_flip_tensor[None,None,:]
                # (S,B,3)
            else:
                raise NotImplementedError
            
            for data_kind in self.config["data_kind_ffm"]:
                # ic.enable()
                # ic(data_kind)
                # ic(data[data_kind].device)

                for data_str in ['data','data_mes']:
                    if data_kind in eval(data_str):
                        # eval(data_str)[data_kind] = self.ffm[data_kind](eval(data_str)[data_kind])
                        exec(f'eval(data_str)[data_kind] = self.ffm_{data_kind}(eval(data_str)[data_kind])')
                        if data_kind == 'SrcPos_Cart':
                            # eval(data_str)['SrcPos_Cart_lr_flip'] = self.ffm[data_kind](eval(data_str)['SrcPos_Cart_lr_flip'])
                            exec(f"eval(data_str)['SrcPos_Cart_lr_flip'] = self.ffm_{data_kind}(eval(data_str)['SrcPos_Cart_lr_flip'])")
                # if data_kind in data:
                #     data[data_kind] = self.ffm[data_kind](data[data_kind])
                #     if data_kind == 'SrcPos_Cart':
                #         data['SrcPos_Cart_lr_flip'] = self.ffm[data_kind](data['SrcPos_Cart_lr_flip'])
                # if data_kind in data_mes:
                #     data_mes[data_kind] = self.ffm[data_kind](data_mes[data_kind])
                #     if data_kind == 'SrcPos_Cart':
                #         data_mes['SrcPos_Cart_lr_flip'] = self.ffm[data_kind](data_mes['SrcPos_Cart_lr_flip'])
            
            device = data['HRTF_mag'].device
            hyper_en_x = th.zeros(S,B_mes,0,1, device=device, dtype=th.float32)
            for data_kind in self.config["data_kind_interp"]:
                if data_kind in ['ITD']:
                    hyper_en_x = th.cat((hyper_en_x, data_mes[data_kind][:,:,None,None]), dim=2) # (S,B_mes,1,1)
                elif data_kind in ['HRTF_mag','HRIR']:
                    hyper_en_x = th.cat((hyper_en_x, data_mes[data_kind][:,:,0,:,None], data_mes[data_kind][:,:,1,:,None]), dim=2) # (S,B_mes,2L,1)
            
            if self.config["data_kind_interp"] == ['ITD']:
                hyper_en_z = th.zeros(S,B_mes,1,0, device=device, dtype=th.float32)
                for data_kind in self.config["data_kind_hyper_en"]:
                    if data_kind in ['SrcPos_Cart']:
                        hyper_en_z = th.cat((hyper_en_z, data_mes[data_kind][:,:,None,:]), dim=3) # (S,B_mes,1,3 or num_ff)
                    elif data_kind in ['Freq']:
                        pass
                    elif data_kind in ['B_mes']:
                        hyper_en_z = th.cat((hyper_en_z, data[data_kind][None,None,:,:].tile(S,B_mes,1,1)), dim=3) # (S,B_mes,1,1 or num_ff)
            elif len(self.config["data_kind_interp"]) == 1:
                hyper_en_z = th.zeros(S,B_mes,2*LL,0, device=device, dtype=th.float32)
                # left and right ear concatenated here
                for data_kind in self.config["data_kind_hyper_en"]:
                    if data_kind in ['SrcPos_Cart']:
                        hyper_en_z = th.cat((hyper_en_z, th.cat((data_mes[data_kind].unsqueeze(2).tile(1,1,LL,1), data_mes[f'{data_kind}_lr_flip'].unsqueeze(2).tile(1,1,LL,1)), dim=2)), dim=3) # (S,B_mes,2*LL,3 or num_ff)
                    elif data_kind in ['Freq']:
                        hyper_en_z = th.cat((hyper_en_z, data[data_kind].tile(2,1)[None,None,:,:].tile(S,B_mes,1,1)), dim=3)  # (S,B_mes,2*LL,1 or num_ff)
                    elif data_kind in ['B_mes']:
                        hyper_en_z = th.cat((hyper_en_z, data[data_kind][None,None,:,:].tile(S,B_mes,2*LL,1)), dim=3) # (S,B_mes,2*LL,1 or num_ff)
            else:
                hyper_en_z = th.zeros(S,B_mes,2*LL+1,0, device=device, dtype=th.float32)
                for data_kind in self.config["data_kind_hyper_en"]:
                    if data_kind in ['SrcPos_Cart']:
                        hyper_en_z = th.cat((hyper_en_z, th.cat((data_mes[data_kind].unsqueeze(2).tile(1,1,LL,1), data_mes[f'{data_kind}_lr_flip'].unsqueeze(2).tile(1,1,LL,1), data_mes[data_kind].unsqueeze(2)), dim=2)), dim=3) # (S,B_mes,2*LL+1,3 or num_ff)
                    elif data_kind in ['Freq']:
                        freq_tensor = data[data_kind].tile(2,1)
                        freq_dammy  = th.zeros_like(freq_tensor)[0:1,:]
                        hyper_en_z = th.cat((hyper_en_z, th.cat((freq_tensor,freq_dammy), dim=0)[None,None,:,:].tile(S,B_mes,1,1)), dim=3)  # (S,B_mes,2*LL+1,1 or num_ff)
                    elif data_kind in ['B_mes']:
                        hyper_en_z = th.cat((hyper_en_z, data[data_kind][None,None,:,:].tile(S,B_mes,2*LL+1,1)), dim=3) # (S,B_mes,2*LL+1,1 or num_ff)
                delta = th.cat((th.zeros(2*LL, device=device, dtype=th.float32), th.ones(1, device=device, dtype=th.float32)), dim=0)
                hyper_en_z = th.cat((hyper_en_z, delta[None,None,:,None].tile(S,B_mes,1,1)), dim=3)
            
            # ic(hyper_en_x.shape)
            # ic(hyper_en_z.shape)
            
            latents = self.en_0({
                "x": hyper_en_x, # (S,B_mes, 1 or 2LL or 2LL+1, 1)
                "z": hyper_en_z, # (S,B_mes, 1 or 2LL or 2LL+1, *)
            })["x"]
            # (S,B_mes, 1 or 2LL or 2LL+1, d)
            LLL = latents.shape[2]
            # ic(latents.shape)
            if self.config["use_mid_linear"]:
                latents = self.mid_linear(latents.permute(0,1,3,2)).permute(0,1,3,2)
                # ic(latents.shape)

            returns["z_bm"] = latents
            latents = th.mean(latents, dim=self.config["mid_mean_dim"], keepdim=True)  # (S, 1, 1 or 2LL or 2LL+1, d)
            returns["z"] = latents ## Latent variables
            # ic(latents.shape, latents)
            ## HRTF_mag: (S (batch dimention), 1 (source positions), 2LL (frequency bins LR), d)
            ## Test and print D

            ## Decoding process
            if self.config["mid_mean_dim"] == (1,):
                latents = latents.tile(1,B,1,1)
            elif self.config["mid_mean_dim"] == (1,2):
                latents = latents.tile(1,B,LLL,1)

            # latents = latents.unsqueeze(1).tile(1,B,1,1) # (S, B, 1 or 2LL or 2LL+1, d)

            if self.config["data_kind_interp"] == ['ITD']:
                hyper_de_z = th.zeros(S,B,1,0, device=device, dtype=th.float32)
                for data_kind in self.config["data_kind_hyper_en"]:
                    if data_kind in ['SrcPos_Cart']:
                        hyper_de_z = th.cat((hyper_de_z, data[data_kind][:,:,None,:]), dim=3) # (S,B,1,3 or num_ff)
                    elif data_kind in ['Freq']:
                        pass
            elif len(self.config["data_kind_interp"]) == 1:
                hyper_de_z = th.zeros(S,B,2*LL,0, device=device, dtype=th.float32)
                for data_kind in self.config["data_kind_hyper_en"]:
                    if data_kind in ['SrcPos_Cart']:
                        hyper_de_z = th.cat((hyper_de_z, th.cat((data[data_kind].unsqueeze(2).tile(1,1,LL,1), data[f'{data_kind}_lr_flip'].unsqueeze(2).tile(1,1,LL,1)), dim=2)), dim=3) # (S,B,2*LL,3 or num_ff)
                    elif data_kind in ['Freq']:
                        hyper_de_z = th.cat((hyper_de_z, data[data_kind].tile(2,1)[None,None,:,:].tile(S,B,1,1)), dim=3)  # (S,B,2*LL,1 or num_ff)
            else:
                hyper_de_z = th.zeros(S,B,2*LL+1,0, device=device, dtype=th.float32)
                for data_kind in self.config["data_kind_hyper_en"]:
                    if data_kind in ['SrcPos_Cart']:
                        hyper_de_z = th.cat((hyper_de_z, th.cat((data[data_kind].unsqueeze(2).tile(1,1,LL,1), data[f'{data_kind}_lr_flip'].unsqueeze(2).tile(1,1,LL,1), data[data_kind].unsqueeze(2)), dim=2)), dim=3) # (S,B,2*LL+1,3 or num_ff)
                    elif data_kind in ['Freq']:
                        freq_tensor = data[data_kind].tile(2,1)
                        freq_dammy  = th.zeros_like(freq_tensor)[0:1,:]
                        hyper_de_z = th.cat((hyper_de_z, th.cat((freq_tensor,freq_dammy), dim=0)[None,None,:,:].tile(S,B,1,1)), dim=3)  # (S,B,2*LL+1,1 or num_ff)
   
                delta = th.cat((th.zeros(2*LL, device=device, dtype=th.float32), th.ones(1, device=device, dtype=th.float32)), dim=0)
                hyper_de_z = th.cat((hyper_de_z, delta[None,None,:,None].tile(S,B,1,1)), dim=3)
            
            # ic(latents.shape)
            # ic(hyper_de_z.shape)

            out_f = self.de_m1({
                "x": latents,    # (S, B, 1 or 2LL or 2LL+1, d)
                "z": hyper_de_z, # (S, B, 1 or 2LL or 2LL+1, *)
            })["x"]

            # ic(out_f.shape)

            if self.config["data_kind_interp"] == ['ITD']:
                returns['ITD'] = out_f[:,:,0,0]  # (S,B)
            elif len(self.config["data_kind_interp"]) == 1:
                data_kind = self.config["data_kind_interp"][0]
                returns[data_kind] = th.cat((out_f[:,:,None,:LL,0], out_f[:,:,None,LL:,0]), dim=2) # (S,B,2,LL)
            else:
                for data_kind in self.config["data_kind_interp"]:
                    if data_kind == 'ITD':
                        returns['ITD'] = out_f[:,:,0,0]
                        out_f = out_f[:,:,0:,:]
                    else:
                        returns[data_kind] = th.cat((out_f[:,:,None,:LL,0], out_f[:,:,None,LL:2*LL,0]), dim=2) # (S,B,2,LL)
                        out_f = out_f[:,:,2*LL:,:]
            
            for data_kind in self.config["data_kind_interp"]:
                device = returns[data_kind].device
                returns[data_kind] = returns[data_kind] * self.config["Val_Standardize"][data_kind][db_name]["std"].to(device) + self.config["Val_Standardize"][data_kind][db_name]["mean"].to(device)
                # for dic in ['returns','data']:
                #     eval(dic)[data_kind] = eval(dic)[data_kind] * self.config["Val_Standardize"][data_kind][db_name]["std"] + self.config["Val_Standardize"][data_kind][db_name]["mean"]

        else:
        

            # 'SrcPos_Cart':  (S,B,3) torch.float32 [x,y,z]
            # 'HRTF':         (S,B,2,L,2) torch.float32 not implemented yet
            # 'HRTF_mag':     (S,B,2,L) torch.float32
            # 'HRIR':         (S,B,2,2L) torch.float32
            input_rs = input_rs.permute(2,0,1) # (S,B',2L+3) or (S,B',1+3)
            # B_p = input_rs.shape[1] # B'
            if self.config["DNN_for_interp_ITD"]:
                input_rs_x = input_rs[:,:,0:1] # (S,B',1)
                input_rs_z = input_rs[:,:,-3:]  # (S,B',3)
            elif self.config["use_lr_aug"]:
                input_rs_x = th.cat((input_rs[:,:,0*L:1*L],input_rs[:,:,1*L:2*L]),dim=0) # (2S,B',L)
                pos_l = input_rs[:,:,-3:]
                pos_r = pos_l * th.tensor([1,-1,1],device=pos_l.device)[None, None, :] # y座標 反転
                input_rs_z = th.cat((pos_l, pos_r), dim=0)  # (2S,B',3)
            else:
                raise NotImplementedError
            
            if self.config["use_ffm_for_hyper_cartpos"]:
                input_rs_z = self.ffm_cartpos(input_rs_z/self.config["ffm_norm_cartpos"])

            if self.config["in_mag_pc"]:
                V = th.load("v_mag_train.pt").to(input_rs_x.device) # (L,L)
                q = self.config['num_pc']
                input_rs_x = input_rs_x.reshape(-1,L) # (2S,B',L) -> (2S*B',L)
                input_rs_x = th.matmul(input_rs_x, V[:,:q]) # (2S*B',L)@(L,q) = (2S*B',q)
                input_rs_x = input_rs_x.reshape(2*S,B_p,q) # (2S, B',q)
                L = q
                # print(input_rs_x.shape)
            elif self.config["in_latent"]:
                input_rs_x = self.pre_nn(input_rs_x)
                q = self.config["num_latents"]
                L = q
            elif self.config["in_cnn"]:
                input_rs_x = input_rs_x.reshape(2*S*B_p,1,L)
                input_rs_x = self.pre_cnn(input_rs_x) # (2SB',1, Q)
                input_rs_x = input_rs_x.reshape(2*S,B_p,-1)  # (2S, B', Q)
                L = input_rs_x.shape[-1]
            elif self.config["in_cnn_ch"]:
                # print(input_rs_x.shape) # debug_0805
                input_rs_x = input_rs_x.reshape(2*S*B_p,1,L)
                # print(input_rs_x.shape) # debug_0805
                input_rs_x = self.pre_cnn(input_rs_x) # (2SB',ch, L//ch)
                cnn_max_ch = input_rs_x.shape[1]
                # print(input_rs_x.shape) # debug_0805
                input_rs_x = input_rs_x.reshape(2*S,B_p,-1)  # (2S, B', L)
                L = input_rs_x.shape[-1]
                # print("------") # debug_0805
            


            input_rs_x = input_rs_x.permute(0,2,1)  # (2S,L,B') or (S,1,B')
            if self.config["DNN_for_interp_ITD"]:
                # input_rs_z = input_rs_z.unsqueeze(1) #  (S,1,B',3)
                pass # input_rs_z (S,B',3)
            else:
                if not self.config["hyperconv_en_0"]:
                    input_rs_z = th.tile(input_rs_z.unsqueeze(1), (1,L,1,1)) #  (2S,L,B',3)

                if self.config["use_RSH_for_hyper_en"]:
                    r_en = th.tile(r[idx_t_des,0], (2,))
                    theta_en = th.tile(theta[idx_t_des,0], (2,))
                    phi_en = th.cat((phi[idx_t_des,0], 2*np.pi-phi[idx_t_des,0]))
                    SpF_en = SpecialFunc(r=r_en, phi=phi_en, theta=theta_en, maxto=self.config["max_truncation_order"], fft_length=self.fft_length, max_f=self.max_f)
                    RSH_en = SpF_en.RealSphHarm().to(input_rs_x.device)
                    # print(SH_en.shape) # (2*B',dimz=(N+1)^2) # torch.Size([880, 64])
                    RSH_en = th.tile(RSH_en.reshape(1,1,2*B_p,-1), (S,L,1,1))
                    # print(RSH_en.shape) # (S,L,2B',(N+1)^2) # torch.Size([16, 128, 880, 64])
                    RSH_en = th.cat((RSH_en[:,:,:B_p,:], RSH_en[:,:,B_p:,:]), dim=0)
                    # print(RSH_en.shape) # (2S,L,B',(N+1)^2)
                    # sys.exit()
                    input_rs_z = RSH_en.to(dtype=input_rs_z.dtype)
                    
                if not self.config["hyperconv_en_0"] and self.config["use_freq_for_hyper_en"]:
                    # freq_th = th.from_numpy(self.f_arr.astype(np.float32)).clone().to(input_rs_z.device) # (L)
                    # freq_th = freq_th / (freq_th[-1]/2) -1  # (0,1]
                    freq_th = (th.arange(L+1)[1:]/(L/2)-1).to(input_rs_z.device)
                    if not self.config["hyperconv_en_0"]:
                        freq_th = th.tile(freq_th.view(1,L,1,1), (2*S,1,B_p,1)) # (2S,L,B',1)
                        if self.config["use_ffm_for_hyper_freq"]:
                            freq_th = self.ffm_freq(freq_th)
                        input_rs_z = th.cat((input_rs_z,freq_th), dim=-1) #  (2S,L,B',4)
                    else:
                        if self.config["hyperconv_en_0_pad"] == 'same':
                            freq_th = freq_th.reshape(1,L).tile(L,1) # (L,L)
                            idx_gather = th.arange(L).to(freq_th.device)
                            idx_offset = th.arange(L).to(freq_th.device)
                            idx_gather = (idx_gather[None,:] + idx_offset[:,None]) % L
                            ks = self.config["hyperconv_en_0_ks"]
                            freq_th = th.gather(freq_th, -1, idx_gather)[:,:ks] # (L, ks)
                            freq_th = freq_th.reshape(1,L,1,ks).tile(2*S,1,B_p,1) # (2S, L, B', ks)
                            input_rs_z = th.cat((input_rs_z,freq_th), dim=-1) # (2S, L, B', 3+ks)
                        else:
                            raise NotImplementedError
            
            if self.config["use_Bp_for_hyper_en"]:
                Bp_norm = self.config["Bp_norm"] if "Bp_norm" in self.config.keys() else B
                if self.config["DNN_for_interp_ITD"]:
                    B_p_th = th.tensor([B_p/Bp_norm*2-1]).reshape(1,1,1).tile(S,B_p,1).to(input_rs_z.device) # (S,B',1)
                elif self.config["hyperconv_en_0"]:
                    B_p_th = th.tensor([B_p/Bp_norm*2-1]).reshape(1,1,1).tile(2*S,B_p,1).to(input_rs_z.device)
                else:
                    B_p_th = th.tensor([B_p/Bp_norm*2-1]).reshape(1,1,1,1).tile(2*S,L,B_p,1).to(input_rs_z.device)
                if self.config["use_ffm_for_hyper_Bp"]:
                    B_p_th = self.ffm_Bp(B_p_th)
                # print(input_rs_z.shape)
                # print(B_p_th.shape)
                # print(B_p_th[0,0,0,0])
                input_rs_z = th.cat((input_rs_z, B_p_th), dim=-1)  #  (2S,L,B',3/4) or (2S,B',3/4) or (S,B',3/4)

            #==============================
            if self.config["hlayers_En_0"]>0:
                # input_rs_x,  # (2S,L,B')
                # input_rs_z,  # (2S,L,B', 3or4)
                if self.config["DNN_for_interp_ITD"]:
                    input_rs_x_0 = input_rs_x.reshape(S, B_p, 1)
                    input_rs_z_0 = input_rs_z.reshape(S, B_p,-1)
                    input_hyper_en_0 = {
                        "x": input_rs_x_0,  # (S,B',1)
                        "z": input_rs_z_0,  # (S,B', 3or4)
                    }
                    input_rs_x = self.en_0(input_hyper_en_0)["x"] # x:(S,1,B',1 or d)
                    input_rs_x = input_rs_x.reshape(S, 1, B_p, -1)
                elif self.config["hyperlinear_en_0"]:
                    input_rs_x_0 = input_rs_x.reshape(2*S, L*B_p, 1)
                    input_rs_z_0 = input_rs_z.reshape(2*S, L*B_p,-1)
                    input_hyper_en_0 = {
                        "x": input_rs_x_0,  # (2S,LB',1)
                        "z": input_rs_z_0,  # (2S,LB', 3or4)
                    }
                    # print(input_hyper_en_0["x"].dtype)
                    # print(input_hyper_en_0["z"].dtype)
                    input_rs_x = self.en_0(input_hyper_en_0)["x"] # x:(2S,L,B',1 or d)
                    input_rs_x = input_rs_x.reshape(2*S, L, B_p, -1)
                elif self.config["hyperconv_en_0"]:
                    input_rs_x_0 = input_rs_x.permute(0,2,1).reshape(2*S*B_p, 1, L) # (2S,L,B') -> (2S,B',L) -> (2SB',1,L)
                    input_rs_z_0 = input_rs_z.reshape(2*S*B_p,-1) # (2S,B',3/4) -> (2SB',3/4)
                    input_hyper_en_0 = {
                        "x": input_rs_x_0,  # (2SB',1or2,L)
                        "z": input_rs_z_0,  # (2SB',3or4)
                    }
                    if self.config["use_freq_for_hyper_en"]:
                        freq_th = (th.arange(L+1)[1:]/(L/2)-1).to(input_rs_x_0.device)
                        freq_th = th.tile(freq_th.view(1,1,L), (2*S*B_p,1,1)) # (2SB',1,L)
                        input_hyper_en_0["f"] = freq_th

                    input_rs_x = self.en_0(input_hyper_en_0)["x"] # (2SB',d, L)
                    input_rs_x = input_rs_x.reshape(2*S, B_p, -1, L)
                    input_rs_x = input_rs_x.permute(0,3,1,2) # (2S, L, B',d)

                    # print(f"input_rs_x.shape: {input_rs_x.shape}") # dbg

                elif self.config["hyperconv_FD_en_0"]:
                    input_rs_x_0 = input_rs_x.permute(0,2,1) # (2S,B',L)
                    input_rs_x_0 = input_rs_x_0.reshape(2*S, B_p, 1, L) # (2S,B',1, L)
                    input_rs_z_0 = input_rs_z.permute(0,2,1,3) # (2S, B', L, 3+ks)
                    input_rs_z_0 = input_rs_z_0[0,:,:,:] # (B', L, 3+ks)
                    input_hyper_en_0 = {
                        "x": input_rs_x_0,  # (2S,B',1, L)
                        "z": input_rs_z_0,  # (B', L, 3+ks)
                    }
                    input_rs_x = self.en_0(input_hyper_en_0)["x"] # (2S, B', d, L)
                    input_rs_x = input_rs_x.permute(0,3,1,2)  # (2S, L, B', d)
                if not self.config["aggregation_mean"]:
                    input_rs_x = input_rs_x.reshape(2*S, L, B_p)
            else:
                pass
            

            if self.config["aggregation_mean"]:
                # print(input_rs_x.shape) # torch.Size([8, 128, 440, 441])
                if self.config["z_norm"]:
                    input_rs_x = F.normalize(input_rs_x, dim=-1)
                returns["z_bm"] = input_rs_x # (2S, L, B', d)
                latents = th.mean(input_rs_x, dim=2) # (2S, L, d)
                if self.config["z_norm"]:
                    latents = F.normalize(latents, dim=-1)
                # print(latents.shape) # torch.Size([8, 128, 441])
            else:
                # print(f'max|input_rs_x|:{th.max(th.abs(input_rs_x))}') # tensor(9.4058, device='cuda:0') # nan_debug
                # print(th.max(th.abs(input_rs_z))) # tensor(1.4700, device='cuda:0')
                input_hyper_en_1 = {
                    "x": input_rs_x,  # (2S,L,B')
                    "z": input_rs_z,  # (2S,L,B', 3or4)
                }
                # print(input_rs_x.shape) # torch.Size([64, 128, 440])
                # print(input_rs_z.shape) # torch.Size([64, 128, 440, 3])
                latents = self.en_1(input_hyper_en_1)
                returns["weight_en_1"] = latents["weight"] # (S,L,B',ch_en)
                if self.config["weight_en_sph_harm"]:
                    # flip のこと考えてなかった．要修正
                    SpF_en = SpecialFunc(r=r[idx_t_des,0], phi=phi[idx_t_des,0], theta=theta[idx_t_des,0], maxto=self.config["max_truncation_order"], fft_length=self.fft_length, max_f=self.max_f)
                    RSH_en = SpF_en.RealSphHarm().to(input_rs_x.device)
                    # print(SH_en.shape) # (B',dimz=(N+1)^2) # torch.Size([440, 64])
                    if self.config["pinv_FIAE_en"]:
                        returns["RSH_en"] = RSH_en
                    else:
                        returns["RSH_en"] = RSH_en / (4*np.pi)

                latents = latents["x"]
                # print(latents.shape) # (2S,L,ch_en or dim_z) # torch.Size([64, 128, 128])
                # print(f'max|latents|:{th.max(th.abs(latents))}') # tensor(281.0440, device='cuda:0', grad_fn=<MaxBackward1>) # nan_debug
            

            if self.config["hlayers_En_z"] > 0:
                if self.config["use_freq_for_hyper_en"]:
                    input_hyper_en_2 = {
                        "x": latents,  # (2S,L,ch_en)
                        "z": freq_th[:,:,0,:],  # (2S,L, 1)
                    }
                    latents = self.en_2(input_hyper_en_2)["x"]
                else:
                    latents = self.en_2(latents)

            #======== attention to reflect all mes.pos. info. ====================
            # print(th.max(th.abs(latents)))
            if self.config["pos_all_en_attn"]:
                #=== key ====
                if self.config["use_freq_for_hyper_en"]:
                    input_rs_z_attn = input_rs_z[0,:,:,:] #  (2S,L,B',4) -> (L,B',4)
                else:
                    input_rs_z_attn = input_rs_z[0,0,:,:] #  (2S,L,B',4) -> (B',4)
                mat_en_k = self.en_attn_k(input_rs_z_attn) # (L,B',d_z) or (B',d_z)
                if self.config["pos_all_en_attn_gram"]:
                    mat_en_q = self.en_attn_k(input_rs_z_attn) # (L,B',d_z) or (B',d_z)
                else:
                    mat_en_q = self.en_attn_q(input_rs_z_attn) # (L,B',d_z) or (B',d_z)

                if not self.config["use_freq_for_hyper_en"]:
                    mat_en_k = th.tile(mat_en_k.unsqueeze(0), (L,1,1)) # (L,B',d_z)
                    mat_en_q = th.tile(mat_en_q.unsqueeze(0), (L,1,1)) # (L,B',d_z)
                
                mat_en_kq = th.bmm(mat_en_k.permute(0,2,1), mat_en_q) # (L,d_z,d_z)
                mat_en_kq = th.tile(mat_en_kq.unsqueeze(0), (2*S,1,1,1))  # (2*S,L,d_z,d_z)
                mat_en_kq = th.reshape(mat_en_kq, (2*S*L,self.config["dim_z"],self.config["dim_z"])) # (2*S*L,d_z,d_z)
                mat_en_kq += self.config["pos_all_en_attn_eye_coeff"]*th.eye(self.config["dim_z"])[None,:,:].to(mat_en_kq.device)
                latents = th.reshape(latents, (2*S*L,self.config["dim_z"],1)) # # (2S,L,d_z) -> (2SL, d_z, 1)
                latents = th.bmm(mat_en_kq, latents) # (2SL, d_z, 1)
                latents = th.reshape(latents, (2*S,L,self.config["dim_z"]))
            else:
                pass

            #===== Mid-CNN -------------------------------
            if self.config["use_mid_conv"]:
                if self.config["res_mid_conv"]:
                    latents_res = latents.clone()
                latents = latents.permute(0,2,1) # 2S, d_z, L
                if ["use_freq_for_mid_conv"]:
                    # freq_th = th.from_numpy(self.f_arr.astype(np.float32)).clone().to(latents.device) # (L)
                    # freq_th = freq_th / (freq_th[-1]/2) -1  # (0,1]
                    freq_th = (th.arange(L+1)[1:]/(L/2)-1).to(latents.device)
                    freq_th = th.tile(freq_th.view(1,1,L), (2*S,1,1)) # (2S,1,L)
                    latents = th.cat((latents,freq_th),dim=1) # (2S,d_z+1,L)
                latents = self.mid_cnn(latents)
                latents = latents.permute(0,2,1) # 2S, L, d_z
                if self.config["res_mid_conv"]:
                    latents = latents + latents_res

            if self.config["use_mid_conv_simple"]:
                latents = latents.permute(0,2,1) # 2S, d_z, L
                if self.config["ch_mid_conv_simple"] == 1:
                    latents = latents.reshape(-1,1, latents.shape[-1]) # 2S, 1, L
                if self.config["use_freq_for_mid_conv"]:
                    freq_th = (th.arange(L+1)[1:]/(L/2)-1).to(latents.device)
                    freq_th = th.tile(freq_th.view(1,1,L), (latents.shape[0],1,1)) # (2S,1,L) or (2S*d,1,L)
                    latents = th.cat((latents,freq_th),dim=1) # (2S,1+d,L) or  # (2S*d,1+1,L)
                latents = self.mid_cnn(latents)
                latents = latents.reshape(2*S,-1,L)  # 2S, d, L
                latents = latents.permute(0,2,1)  # 2S, L, d_z

            #=============================================
            if self.config["use_mid_linear"]:
                latents = latents.permute(0,2,1) # 2S, d_z, L
                latents = self.mid_linear(latents) # 2S, d_z, 1
                latents = latents.tile(1,1,self.config['fft_length']//2).permute(0,2,1)  # 2S, L, d_z

            # print(th.max(th.abs(latents)))
            # sys.exit()
            # print(latents.shape) # (2S,L,dim_z) # torch.Size([64, 128, 64]) # torch.Size([4, 128, 441])
            # sys.exit()
            returns["z"] = latents

            if self.config["hlayers_De_z"] > 0:
                if self.config["use_freq_for_hyper_de"]:
                    input_hyper_de_1 = {
                        "x": latents,  # (2S,L,dim_z)
                        "z": freq_th[:,:,0,:],  # (2S,L, 1)
                    }
                    output = self.de_1(input_hyper_de_1)["x"]
                else:
                    output = self.de_1(latents)
                # output (2S,L,ch_de)
            else:
                output = latents # (2S, L, d) or (S,1,d)
            if self.config["DNN_for_interp_ITD"]:
                input_rs_z_out = vec_cart_gt.permute(2,0,1) # (S,B,3)
            else:
                pos_l_out = vec_cart_gt.permute(2,0,1) # (S,B,3)
                pos_r_out = pos_l_out * th.tensor([1,-1,1],device=pos_l_out.device)[None,None, :] # y座標 反転
                input_rs_z_out = th.cat((pos_l_out, pos_r_out), dim=0)  # (2S,B,3)
                if not self.config["hyperconv_de_-1"]:
                    input_rs_z_out = th.tile(input_rs_z_out.unsqueeze(1), (1,L,1,1)) #  (2S,L,B,3)
            if self.config["use_ffm_for_hyper_cartpos"]:
                input_rs_z_out = self.ffm_cartpos(input_rs_z_out/self.config["ffm_norm_cartpos"])

            if self.config["use_RSH_for_hyper_de"]:
                r_de = th.tile(r[:,0], (2,))
                theta_de = th.tile(theta[:,0], (2,))
                phi_de = th.cat((phi[:,0], 2*np.pi-phi[:,0]))
                SpF_de = SpecialFunc(r=r_de, phi=phi_de, theta=theta_de, maxto=self.config["max_truncation_order"], fft_length=self.fft_length, max_f=self.max_f)
                RSH_de = SpF_de.RealSphHarm().to(output.device)
                # print(RSH_de.shape) # (2*B,dimz=(N+1)^2) # torch.Size([880, 64])
                RSH_de = th.tile(RSH_de.reshape(1,1,2*B,-1), (S,L,1,1))
                # print(RSH_de.shape) # (S,L,2B,(N+1)^2) # torch.Size([16, 128, 880, 64])
                RSH_de = th.cat((RSH_de[:,:,:B,:], RSH_de[:,:,B:,:]), dim=0)
                # print(RSH_de.shape) # (2S,L,B,(N+1)^2)
                input_rs_z_out = RSH_de.to(dtype=input_rs_z_out.dtype)
                # sys.exit()

            if self.config["use_freq_for_hyper_de"]:
                freq_th_out = (th.arange(L+1)[1:]/(L/2)-1).to(input_rs_z_out.device)
                if not self.config["hyperconv_de_-1"]:
                    freq_th_out = th.tile(freq_th_out.view(1,L,1,1), (2*S,1,B,1)) # (2S,L,B,1)
                    if self.config["use_ffm_for_hyper_freq"]:
                        freq_th_out = self.ffm_freq(freq_th_out)
                    input_rs_z_out = th.cat((input_rs_z_out,freq_th_out), dim=-1) #  (2S,L,B,4)

                else:
                    if self.config["hyperconv_de_-1_pad"] == 'same':
                        freq_th_out = freq_th_out.reshape(1,L).tile(L,1) # (L,L)
                        idx_gather = th.arange(L).to(freq_th_out.device)
                        idx_offset = th.arange(L).to(freq_th_out.device)
                        idx_gather = (idx_gather[None,:] + idx_offset[:,None]) % L
                        ks = self.config["hyperconv_de_-1_ks"]
                        freq_th_out = th.gather(freq_th_out, -1, idx_gather)[:,:ks] # (L, ks)
                        freq_th_out = freq_th_out.reshape(1,L,1,ks).tile(2*S,1,B,1) # (2S, L, B, ks)
                        input_rs_z_out = th.cat((input_rs_z_out,freq_th_out), dim=-1) # (2S, L, B, 3+ks)
                    else:
                        raise NotImplementedError

            # print(input_rs_z_out.shape)
            if not self.config["de_2_skip"]:
                input_hyper_de_2 = {
                    "x": output,  # (2S,L,ch_de or dim_z)
                    "z": input_rs_z_out,  # (2S,L,B,3 or 4)
                }
                out_f = self.de_2(input_hyper_de_2) # (2S,L,B)
                returns["weight_de_2"] = out_f["weight"] # (S,L,B,ch_de)
                # print(returns["weight_en_1"].shape)
                # print(returns["weight_de_2"].shape)
                # sys.exit()
                out_f = out_f["x"]
                # print(out_f.shape)
            else:
                out_f = output #self.de_2(output)

            if self.config["weight_de_sph_harm"]:
                # flip のこと考えてなかった．要修正
                SpF_de = SpecialFunc(r=r[:,0], phi=phi[:,0], theta=theta[:,0], maxto=self.config["max_truncation_order"], fft_length=self.fft_length, max_f=self.max_f)
                RSH_de = SpF_de.RealSphHarm().to(returns["weight_de_2"].device)
                # print(SH_en.shape) # (B,dimz=(N+1)^2) # torch.Size([440, 64])
                returns["RSH_de"] = RSH_de
            
            if self.config["hlayers_De_-1"]>0:
                if self.config["de_2_skip"]:
                    input_rs_x_m1 = th.tile(out_f.unsqueeze(2),(1,1,B,1)) # (2S,L,B,ch_de or dim_z) or (S,1,B,ch_de or dim_z)
                else:
                    input_rs_x_m1 = out_f.unsqueeze(3) # (2S,L,B,1)
                # print(input_rs_x_m1.shape)
                if self.config["DNN_for_interp_ITD"]:
                    input_rs_x_m1 = input_rs_x_m1.reshape(S,B,-1)
                    input_rs_z_m1 = input_rs_z_out  # (S,1,B,3 or 4)
                    input_rs_z_m1 = input_rs_z_m1.reshape(S,B,-1)
                    input_hyper_de_m1 = {
                        "x": input_rs_x_m1,  # (S,B, dim_z)
                        "z": input_rs_z_m1,  # (S,B, 3or4)
                    }
                    # print(input_rs_x_m1.shape)
                    # print(input_rs_z_m1.shape)
                    out_f = self.de_m1(input_hyper_de_m1)["x"] # x:(S,B,1)
                    out_f = out_f.reshape(S, 1, B) # x:(S,1,B)
                elif self.config["hyperlinear_de_-1"]:
                    input_rs_x_m1 = input_rs_x_m1.reshape(2*S,L*B,-1)
                    input_rs_z_m1 = input_rs_z_out  # (2S,L,B,3 or 4)
                    input_rs_z_m1 = input_rs_z_m1.reshape(2*S,L*B,-1)
                    input_hyper_de_m1 = {
                        "x": input_rs_x_m1,  # (2S,L*B, dim_z)
                        "z": input_rs_z_m1,  # (2S,L*B, 3or4)
                    }
                    # print(input_rs_x_m1.shape)
                    # print(input_rs_z_m1.shape)
                    out_f = self.de_m1(input_hyper_de_m1)["x"] # x:(2S,L*B,1)
                    out_f = out_f.reshape(2*S, L, B) # x:(2S,L,B)
                    # print(out_f.shape)
                    # sys.exit()
                elif self.config["hyperconv_de_-1"]:
                    # print(f"input_rs_x_m1: {input_rs_x_m1.shape}") # dbg
                    input_rs_x_m1 = input_rs_x_m1.permute(0,2,3,1).reshape(2*S*B, -1, L) # (2S,L,B,d) -> (2S,B,d,L) -> (2SB,d,L)
                    input_rs_z_m1 = input_rs_z_out  # (2S,B,3)
                    input_rs_z_m1 = input_rs_z_m1.reshape(2*S*B,-1) # (2S,B,3) -> (2SB,3)
                    input_hyper_de_m1 = {
                        "x": input_rs_x_m1,  # (2SB,1or2,L)
                        "z": input_rs_z_m1,  # (2SB',3)
                    }
                    if self.config["use_freq_for_hyper_de"]:
                        freq_th = (th.arange(L+1)[1:]/(L/2)-1).to(input_rs_x_0.device)
                        freq_th = th.tile(freq_th.view(1,1,L), (2*S*B,1,1)) # (2SB,1,L)
                        input_hyper_de_m1["f"] = freq_th  # (2SB,1,L)
                    
                    out_f = self.de_m1(input_hyper_de_m1)["x"] # x:(2SB,1,L)
                    out_f = out_f.reshape(2*S, B, L).permute(0,2,1) # x:(2S,L,B)
                    # print(out_f.shape) # dbg
                elif self.config["hyperconv_FD_de_-1"]:
                    # input_rs_x_m1: (2S,L,B,dim_z) -> (2S,B,dim_z,L)
                    input_rs_x_m1 = input_rs_x_m1.permute(0,2,3,1)
                    input_rs_z_m1 = input_rs_z_out  # (2S,L,B,3+ks)
                    # input_rs_z_m1: (2S,L,B,3+ks) -> (2S,B,L,3+ks)
                    input_rs_z_m1 = input_rs_z_m1.permute(0,2,1,3)
                    input_rs_z_m1 = input_rs_z_m1[0,:,:,:] # (B, L, 3+ks)
                    input_hyper_de_m1 = {
                        "x": input_rs_x_m1,  # (2S, B, dim_z, L)
                        "z": input_rs_z_m1,  # (B, L, 3+ks)
                    }
                    out_f = self.de_m1(input_hyper_de_m1)["x"] # (2S, B, 1, L)
                    out_f = out_f.permute(0,3,1,2).reshape(2*S, L, B)
                else:
                    raise NotImplementedError
            else:
                pass
            
            # print(out_f.shape)
            out_f = out_f.permute(0,2,1)# (2S,B,L) or (S,B,1)
            # print(f'max|out_f|:{th.max(th.abs(out_f))}') # tensor(513.5840, device='cuda:0', grad_fn=<MaxBackward1>) # nan_debug
            # sys.exit()

            if self.config["out_mag_pc"]:
                L = V.shape[0]
                out_f = out_f.reshape(-1,q) # (2S,B,q) -> (2SB,q)
                out_f = th.matmul(out_f, V[:,:q].T) # (2SB,q)@(q,L) = (2SB,L)
                out_f = out_f.reshape(2*S,B,L) # (2S,B,L)
            elif self.config["out_latent"]:
                out_f = self.post_nn(out_f)
                L = self.filter_length
            elif self.config["out_cnn"]:
                out_f = out_f.reshape(2*S*B,1,L) # (2SB,1,Q)
                out_f = self.post_cnn(out_f) # (2SB,1,L)
                out_f = out_f.reshape(2*S,B,-1) # (2S,B,L)
                L = out_f.shape[-1]
            elif self.config["out_cnn_ch"]:
                # print(out_f.shape) # debug_0805 
                out_f = out_f.reshape(2*S*B,cnn_max_ch,-1) # (2SB,ch,L//ch)
                # print(out_f.shape) # debug_0805
                out_f = self.post_cnn(out_f) # (2SB,1,L)
                # print(out_f.shape) # debug_0805
                out_f = out_f.reshape(2*S,B,-1) # (2S,B,L)
                L = out_f.shape[-1]
            
            if self.config["DNN_for_interp_ITD"]:
                out = out_f * ITD_std + ITD_mean # (S,B)
                returns['output'] = out.reshape(S,B).permute(1,0) # (B,S)
            elif self.config["DNN_for_interp_HRIR"]:
                out = th.zeros(S,B,2,L, dtype=th.double, device=out_f.device)
                out[:,:,0,:] = out_f[0*S:1*S,:,:] * HRIR_std + HRIR_mean
                out[:,:,1,:] = out_f[1*S:2*S,:,:] * HRIR_std + HRIR_mean
                returns['output'] = out.permute(1,2,3,0) # (B,2,L,S)
            else:
                # out_f: (2S,B,L) float -> out:(S,B,2,L) complex
                out = th.zeros(S,B,2,L, dtype=th.complex64, device=out_f.device)
                if self.config["out_mag"] and self.config["use_lr_aug"]:
                    out[:,:,0,:] = 10**(out_f[0*S:1*S,:,:] * magdb_std + magdb_mean)
                    out[:,:,1,:] = 10**(out_f[1*S:2*S,:,:] * magdb_std + magdb_mean)
                else:
                    raise NotImplementedError
            
                # print(out.shape)

                returns['output'] = out.permute(1,2,3,0) # (B,2,L,S)
                # print(out.shape)

        # for data_kind in returns:
        #     ic(data_kind)
        #     ic(type(returns[data_kind]))
        
        return returns

class HRTFApproxNetwork_CNN(Net):
    def __init__(self,
                config,
                model_name='hrtf_approx_network',
                use_cuda=True,
                c_std=None,
                c_mean=None):
        super().__init__(model_name, use_cuda)
        self.config = config
        self.max_f = config["max_frequency"]
        self.maxto = config["max_truncation_order"]
        self.fft_length = config["fft_length"]
        self.filter_length = round(self.fft_length/2)
        self.f_arr = (np.linspace(0, self.max_f, self.filter_length+1))[1:] # frequency bin, (filterlengh,)
        self.N_arr_np = calcMaxOrder(f=self.f_arr,maxto=self.maxto)
        # self.coeff_size =  round(2*2*np.sum((self.N_arr_np + 1)**2))
        # self.coeff_size =  round(2*3*np.sum((self.N_arr_np + 1)**2))
        # self.c_std = c_std
        # self.c_mean = c_mean

        self.to_in = round(config["num_pts"]**0.5 - 1)

        #=========== CNN ============
        channel = config["channel"]
        out_h = (self.maxto+1)**2 - (self.to_in+1)**2
        # out_h = (self.maxto+1)**2
        in_h  = (self.to_in+1)**2
        modules = []

        if config["conv_dim"] == 2:
        # (N=154, C=2, H=100, W=128) -> (N=154, C=2, H=361, W=128)
            
            kernel_size = [config["kernel_size"],1]
            num_layers = round(np.ceil((out_h - in_h) / (kernel_size[0]-1)))
            
            for l in range(num_layers):
                in_ch  = 2 if l == 0 else channel
                out_ch = 2 if l == num_layers-1 else channel
                pad_bottom = kernel_size[0]-1 if not l == num_layers-1 else out_h - in_h - (kernel_size[0]-1)*(num_layers-1)

                modules.extend([
                    nn.ZeroPad2d(padding=(0,0,kernel_size[0]-1,pad_bottom)),
                    nn.Conv2d(in_ch, out_ch, kernel_size, padding=0)])
                if not l == num_layers-1:
                    modules.extend([
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU()
                    ])
        elif config["conv_dim"] == 1:
        # (N=154, C=2*128, L=100) -> (N=154, C=2*128, L=361)
            kernel_size = config["kernel_size"]
            # num_layers = round(np.ceil((out_h - in_h) / (kernel_size-1)))
            num_layers = round(np.floor((out_h - in_h) / (kernel_size-1)))
            
            for l in range(num_layers):
                in_ch  = 2*self.filter_length if l == 0 else channel
                out_ch = 2*self.filter_length if l == num_layers-1 else channel

                # pad_right = kernel_size-1 if not l == num_layers-1 else out_h - in_h - (kernel_size-1)*(num_layers-1)

                # modules.extend([
                #     nn.ConstantPad1d(padding=(kernel_size-1,pad_right), value=0.0),
                #     nn.Conv1d(in_ch, out_ch, kernel_size, padding=0)])

                pad_init = out_h - in_h - (kernel_size-1)*num_layers
                if l == 0:
                    modules.extend([
                        nn.ConstantPad1d(padding=(0,pad_init), value=0.0)])
                modules.extend([
                    nn.ConvTranspose1d(in_ch, out_ch, kernel_size, padding=0, output_padding=0)])
                if not l == num_layers-1:
                    modules.extend([
                        nn.BatchNorm1d(out_ch),
                        nn.ReLU()
                    ])

        self.cnn = nn.Sequential(*modules)
        #============================

    
    def forward(self, hrtf_re, hrtf_im, srcpos):
        '''
        [args]
            hrtf_re: (440, 2, 128, 77) float
            hrtf_im: (440, 2, 128, 77) float
            srcpos:  (440, 3,      77) float
        '''
        returns = {}
        B = srcpos.shape[0]
        hrtf = th.complex(hrtf_re, hrtf_im)

        #=== solve linear inverse problem ====
        vec_cart_gt = sph2cart(srcpos[:,1,:],srcpos[:,2,:],srcpos[:,0,:]) # sub==0
        idx_t_des = aprox_t_des(pts=vec_cart_gt[:,:,0]/1.47, t=self.to_in, plot=False)
        pos_t = srcpos[idx_t_des,:,:]
        hrtf_t  = hrtf[idx_t_des,:,:,:]
        #--- Special Function ----
        SpF_t = SpecialFunc(r=pos_t[:,0,0], phi=pos_t[:,1,0], theta=pos_t[:,2,0], maxto=self.to_in, fft_length=self.fft_length, max_f=self.max_f)
        Hank_t = SpF_t.SphHankel()
        Hank_t = th.complex(th.nan_to_num(th.real(Hank_t)), th.nan_to_num(th.imag(Hank_t)))
        Harm_t = SpF_t.SphHarm()
        #---
        B_t = pos_t.shape[0]

        #--- Linear inv problem. ---
        N = self.to_in # fixed truncation
        n_vec,_ = sph_harm_nmvec(N)
        num_base = (N+1)**2

        #--- matrix ---
        mat_base = th.zeros(self.filter_length, B_t, num_base, dtype = th.complex64).to(hrtf.device) # Phi; matrix contains bases. (L, B, (N+1)**2)
        # Hank_t: (B,N,L), Harm_t; (B, (N+1)**2)
        for i,n in enumerate(n_vec):
            mat_base[:,:,i] = Hank_t.permute(2,0,1)[:,:,n] * Harm_t[None,:,i]
        #--- Phi^H
        mat_base_H = th.conj(th.transpose(mat_base,1,2)) 
        #--- Phi^+ = Phi^H * Phi
        mat_base_MPI = th.matmul(mat_base_H, mat_base) 
        #--- regularization matrix in Duraiswaini+04
        #--- D=(1+n(n+1))I
        D = th.diag(1+th.from_numpy((n_vec*(n_vec+1)).astype(np.float32)).clone()).to(hrtf.device)
        mat_reg = self.config["reg_w"] * D # (100,100)
        mat = mat_base_MPI + mat_reg[None,:,:] # (L, B, (N+1)**2)

        #--- vector ---
        # hrtf_t: (100, 2, 128, 77)
        # print(hrtf_t.shape)
        h = hrtf_t.permute(2,0,1,3) # (128, 100, 2, 77)
        h = h.contiguous().view(h.shape[0], h.shape[1], -1) # (128, 100, 154)

        #--- solve ---
        # c: (128, 100, 154)
        c = th.linalg.solve(mat ,th.matmul(mat_base_H, h))
        c = c.view(c.shape[0], c.shape[1], 2, -1) # (128, 100, 2, 77)
        Coeff_in = c.permute(2,1,0,3) # (2, 100, 128, 77)
        returns['coeff_in'] = Coeff_in
        Coeff_in = Coeff_in.permute(3,0,1,2)
        #=====================================

        #========= CNN (inpainting) ==========
        # [in ] Coeff_in: (77, 2, 100, 128) complex64
        # [out] Coeff:    (77, 2, 361, 128) complex64
        #-------------------------------------

        bs = Coeff_in.shape[0] * Coeff_in.shape[1] # 154
        Coeff_in = Coeff_in.contiguous().view(bs, Coeff_in.shape[2], Coeff_in.shape[3]) # (154, 100, 128)
        Coeff_in_re, Coeff_in_im = th.real(Coeff_in).unsqueeze(1), th.imag(Coeff_in).unsqueeze(1)

        input = th.cat((Coeff_in_re, Coeff_in_im), dim=1) 
        if self.config["conv_dim"] == 2:
            input = input # (N=154, C=2, H=100, W=128)
            Coeff = self.cnn(input) # (N=154, C=2, H=361-100, W=128)
        elif self.config["conv_dim"] == 1:
            input_1d = input.permute(0,1,3,2).contiguous().view(input.shape[0],input.shape[1]*input.shape[3],input.shape[2]) # (N=154, C=2*128, L=100)
            Coeff = self.cnn(input_1d) # (N=154, C=2*128, L=361-100)
            Coeff = Coeff.view(Coeff.shape[0],2,-1,Coeff.shape[2]).contiguous().permute(0,1,3,2) # (N=154, C=2, H=361-100, W=128)
        Coeff = th.cat((input,Coeff), dim=2) # (N=154, C=2, H=361, W=128)
        Coeff = th.complex(Coeff[:,0,:,:], Coeff[:,1,:,:]) # (N=154, H=361, W=128) complex64
        Coeff = Coeff.contiguous().view(hrtf.shape[-1],2,Coeff.shape[1],Coeff.shape[2]) # (77, 2, 361, 128)
        #=====================================

        #=== Spherical Wave Function Expansion ====
        out = th.zeros(hrtf.shape[-1], B, 2, self.filter_length, dtype = th.complex64).to(hrtf.device) # (77, B=440, 2, 128)
        #--- special function
        r, phi, theta = srcpos[:,0], srcpos[:,1], srcpos[:,2]
        #---- dammy returns -----
        returns['vec'] = vec_cart_gt
        returns['theta'] = theta # zenith
        returns['phi'] = phi # azimuth
        returns['z'] = th.zeros(2,2,hrtf.shape[-1]).cuda()
        #------------------------
        SpF = SpecialFunc(r=r[:,0], phi=phi[:,0], theta=theta[:,0], maxto=self.maxto, fft_length=self.fft_length, max_f=self.max_f)
        n_vec, _ = SpF.n_vec, SpF.m_vec
        Hank = SpF.SphHankel().to(hrtf.device)
        # Hank = th.complex(th.nan_to_num(th.real(Hank)), th.nan_to_num(th.imag(Hank)))
        Harm = SpF.SphHarm().to(hrtf.device)
        #---
        for i,n in enumerate(n_vec):
            mask = th.from_numpy(self.N_arr_np.astype(np.float32)).clone() >= n
            mask = mask.to(Coeff.device)
            Coeff[:,:,i,:] = Coeff[:,:,i,:] * mask[None,:] # masking
            Coeff_mn = Coeff[:,:,i,:]
            # print(Coeff_mn.shape) # torch.Size([4, 2, 128])
            Hank_n = Hank[:,n,:] * mask[None,:] # masking
            # print(Hank_n.shape) # torch.Size([440, 128])
            Harm_mn = Harm[:,i]
            # print(Harm_mn.shape) # torch.Size([440])
            delta = th.mul(Coeff_mn[:,None,:,:], Hank_n[None,:,None,:]) * Harm_mn[None,:,None,None]
            # print(delta.shape) # torch.Size([4,440, 2, 128])
            out += delta
        
        returns['coeff'] = Coeff.permute(1,2,3,0)
        # coeff: (2, (maxto+1)**2=361, 128, 77)
        returns['output'] = out.permute(1,2,3,0)
        # output: (B=440, 2, 128, 77)
        return returns
        
class HRTFApproxNetwork_CAE(Net):
    def __init__(self,
                config,
                model_name='hrtf_approx_network',
                use_cuda=True):
        super().__init__(model_name, use_cuda)
        self.config = config
        self.max_f = config["max_frequency"]
        self.maxto = config["max_truncation_order"]
        self.fft_length = config["fft_length"]
        self.filter_length = round(self.fft_length/2)
        self.f_arr = (np.linspace(0, self.max_f, self.filter_length+1))[1:] # frequency bin, (filterlengh,)
        self.N_arr_np = calcMaxOrder(f=self.f_arr,maxto=self.maxto)
        
        ### Encoder (common)
        modules = []
        modules.extend([
                        nn.Conv1d(in_channels=4, out_channels=config["channel_En"], kernel_size=config["kernel_size"], stride=2, padding=round(config["kernel_size"]/2-1)),
                        nn.LayerNorm(round(config["fft_length"]/(2**2))),
                        nn.ReLU(), 
                        nn.Dropout(config["droprate"])])
        for l in range(config["hlayers_En"]):
            modules.extend([
                            nn.Conv1d(in_channels=config["channel_En"], out_channels=config["channel_En"], kernel_size=config["kernel_size"], stride=2, padding=round(config["kernel_size"]/2-1)),
                            nn.LayerNorm(round(config["fft_length"]/(2**(3+l)))),
                            nn.ReLU(), 
                            nn.Dropout(config["droprate"])
                            ])
        self.Encoder = nn.Sequential(*modules)

        ### Encoder (z)
        modules = []
        modules.extend([
                        nn.Conv1d(in_channels=config["channel_En"], out_channels=config["channel_En_z"], kernel_size=config["kernel_size"], stride=2, padding=round(config["kernel_size"]/2-1)),
                        nn.LayerNorm(round(config["fft_length"]/(2**(3+config["hlayers_En"])))),
                        nn.ReLU(), 
                        nn.Dropout(config["droprate"])
                        ])
        for l in range(config["hlayers_En_z"]):
            modules.extend([
                            nn.Conv1d(in_channels=config["channel_En_z"], out_channels=config["channel_En_z"], kernel_size=config["kernel_size"], stride=2, padding=round(config["kernel_size"]/2-1)),
                            nn.LayerNorm(round(config["fft_length"]/(2**(4+config["hlayers_En"]+l)))),
                            nn.ReLU(), 
                            nn.Dropout(config["droprate"])
                            ])
        self.Encoder_z = nn.Sequential(*modules)

        ### Encoder (v)
        modules = []
        modules.extend([nn.Flatten(),
                        nn.Linear(round(config["channel_En"]*config["fft_length"]/(2 * 2**(config["hlayers_En"]+1))), config["channel_En_v"]),
                        nn.LayerNorm(config["channel_En_v"]),
                        nn.ReLU(), 
                        nn.Dropout(config["droprate"])])
        for l in range(config["hlayers_En_v"]):
            modules.extend([nn.Linear(config["channel_En_v"], config["channel_En_v"]),
                            nn.LayerNorm(config["channel_En_v"]),
                            nn.ReLU(), 
                            nn.Dropout(config["droprate"])])
        modules.extend([nn.Linear(config["channel_En_v"], 3)])
        self.Encoder_v = nn.Sequential(*modules)

        ### Decoder(z)
        modules = []
        modules.extend([
                nn.ConvTranspose1d(in_channels=config["channel_En_z"], out_channels=config["channel_De_z"], kernel_size=config["kernel_size"], stride=2, padding=round(config["kernel_size"]/2-1)),
                nn.LayerNorm(round(config["fft_length"]/(2**(2+config["hlayers_En"]+config["hlayers_En_z"])))),
                nn.ReLU(), 
                nn.Dropout(config["droprate"])])
        for l in range(config["hlayers_En"]+config["hlayers_En_z"]):
            modules.extend([
                    nn.ConvTranspose1d(in_channels=config["channel_De_z"], out_channels=config["channel_De_z"], kernel_size=config["kernel_size"], stride=2, padding=round(config["kernel_size"]/2-1)),
                    nn.LayerNorm(round(config["fft_length"]/(2**(1+config["hlayers_En"]+config["hlayers_En_z"]-l)))),
                    nn.ReLU(), 
                    nn.Dropout(config["droprate"])])
        modules.extend([
                nn.ConvTranspose1d(in_channels=config["channel_De_z"], out_channels=4*(self.maxto+1)**2, kernel_size=config["kernel_size"], stride=2, padding=round(config["kernel_size"]/2-1))
                ])
        self.Decoder_z = nn.Sequential(*modules)
        
    def forward(self, input, use_cuda_forward=True, use_srcpos=True, srcpos=None):
        '''
        :param hrtf: the input as a B x ch=4 x L complex tensor
        :return: out: HRTF produce by the network, B x ch x L complex tensor
        '''
        returns = {}
        # print(input.shape) # torch.Size([440, 4, 128])
        zz = self.Encoder(input)
        # print(zz.shape) # torch.Size([440, 128, 8])
        z = self.Encoder_z(zz)
        # print(z.shape) # torch.Size([440, 128, 2])
        returns['z'] = z
        Coeff = self.Decoder_z(th.mean(z, dim=0).view(1,z.shape[1],z.shape[2]))
        # Coeff *= 10
        # print(Coeff.shape) # torch.Size([1, 1444, 128])
        Coeff = Coeff.view(Coeff.shape[0],2,-1,Coeff.shape[-1])
        # print(Coeff.shape) # torch.Size([1, 2, 722, 128])
        Coeff = th.complex(Coeff[:,:,:(self.maxto+1)**2,:], Coeff[:,:,(self.maxto+1)**2:,:])
        # print(Coeff.shape) # 

        vec_cart = self.Encoder_v(zz) # torch.Size([B, 3])
        vec_sph = cart2sph(vec_cart[:,0], vec_cart[:,1], vec_cart[:,2])
        phi = vec_sph[:,1]
        theta = vec_sph[:,2]
        r = 1.47 * th.ones(theta.shape[0]) # fixed in HUTUBS
        
        returns['vec'] = vec_cart
        returns['theta'] = theta # zenith
        returns['phi'] = phi # azimuth
        
        if use_srcpos:
            srcpos = srcpos.cuda()
            r = srcpos[:,0]
            phi = srcpos[:,1]
            theta = srcpos[:,2]
        SpF = SpecialFunc(r=r, phi=phi, theta=theta, maxto=self.maxto, fft_length=self.fft_length, max_f=self.max_f)
        n_vec, m_vec = SpF.n_vec, SpF.m_vec
        Hank = SpF.SphHankel()
        Hank = th.complex(th.nan_to_num(th.real(Hank)), th.nan_to_num(th.imag(Hank))).to(Coeff.device)
        Harm = SpF.SphHarm().to(Coeff.device)

        # #### if use sph Bessel
        # Bessel = SpF.SphBessel()[None,None,:,:].to(Coeff.device)
        # Coeff = Coeff * 100 * Bessel
        # ####
        
        out = th.zeros(input.shape[0], 2, SpF.filter_length, dtype = th.complex64)
        if use_cuda_forward:
            out = out.cuda()
            # print(Hank[0,:,0])
        for i,n in enumerate(n_vec):
            mask = th.from_numpy(self.N_arr_np.astype(np.float32)).clone() >= n
            if use_cuda_forward:
                mask = mask.cuda()
            Coeff[0,:,i,:] = Coeff[0,:,i,:] * mask[None,:] # masking
            Coeff_mn = Coeff[0,:,i,:] 
            # print(Coeff_mn.shape) # torch.Size([2, 128])
            Hank_n = Hank[:,n,:] * mask[None,:] # masking
            # print(Hank_n.shape) # torch.Size([440, 128])
            Harm_mn = Harm[:,i]
            # print(Harm_mn.shape) # torch.Size([440])
            delta = th.mul(Coeff_mn[None,:,:], Hank_n[:,None,:]) * Harm_mn[:,None,None]
            # print(delta.shape) # torch.Size([440, 2, 128])
            out += delta
        returns['coeff'] = Coeff
        returns['output'] = out
        return returns
    
class HRTFApproxNetwork_HyperCAE(Net):
    def __init__(self,
                config,
                model_name='hrtf_approx_network',
                use_cuda=True):
        super().__init__(model_name, use_cuda)
        self.config = config
        self.max_f = config["max_frequency"]
        self.maxto = config["max_truncation_order"]
        self.fft_length = config["fft_length"]
        self.filter_length = round(self.fft_length/2)
        self.f_arr = (np.linspace(0, self.max_f, self.filter_length+1))[1:] # frequency bin, (filterlengh,)
        self.N_arr_np = calcMaxOrder(f=self.f_arr,maxto=self.maxto)
        
        ### Encoder
        modules = []
        modules.extend([HyperConvSingleKernelBlock(ch_in=4, ch_out=config["channel_En"], input_size=3)])
        self.Encoder1 = nn.Sequential(*modules) # nn.Sequential(*modules)

        modules = []
        modules.extend([
                        nn.LayerNorm([config["channel_En"],self.filter_length]),
                        nn.ReLU(), 
                        nn.Dropout(config["droprate"])])
        self.Encoder2 = nn.Sequential(*modules)

        modules = []
        for l in range(round(config["hlayers_En"]/2)):
            modules.extend([ResHCSKB(channel=config["channel_En"], droprate=config["droprate"], filter_length=self.filter_length, input_size=3)])
        self.Encoder3 = nn.Sequential(*modules)

        modules = []
        modules.extend([HyperConvSingleKernelBlock(ch_in=config["channel_En"], ch_out=1, input_size=3)])
        self.Encoder4 = nn.Sequential(*modules)
        
        modules = []
        modules.extend([
                        nn.LayerNorm(self.filter_length),
                        nn.ReLU(), 
                        nn.Dropout(config["droprate"])])
        self.Encoder5 = nn.Sequential(*modules)

        ### Decoder
        out_L = (self.maxto+1)**2
        # stride = 2
        # num_layers = round(np.floor(np.log(out_L)/np.log(stride)))

        # modules = []
        # for l in range(num_layers):   
        #     in_channel = 1 if l == 0 else config["channel_De"]
        #     modules.extend([
        #             nn.ConvTranspose1d(in_channels=in_channel, out_channels=config["channel_De"], kernel_size=config["kernel_size"], stride=stride, padding=round(config["kernel_size"]/2-stride/2)),
        #             nn.LayerNorm(stride**(l+1)),
        #             nn.ReLU(), 
        #             nn.Dropout(config["droprate"])])
        # modules.extend([
        #         nn.ConvTranspose1d(in_channels=config["channel_De"], out_channels=4, kernel_size=config["kernel_size"], stride=stride, padding=round((stride**(num_layers+1) - out_L + config["kernel_size"] - stride) / 2))
        #         ])
        # self.Decoder = nn.Sequential(*modules)

        modules = []
        modules.extend([
                        nn.Linear(1, config["channel_De"]),
                        nn.LayerNorm(config["channel_De"]),
                        nn.ReLU(), 
                        nn.Dropout(config["droprate"])])
        modules.extend([ResLinearBlock(config["channel_De"], config["droprate"])])
        for l in range(1):
            modules.extend([ResLinearBlock(config["channel_De"], config["droprate"])])
        modules.extend([nn.Linear(config["channel_De"], 4*out_L)])
        self.Decoder = nn.Sequential(*modules)
            
        
    def forward(self, input, use_cuda_forward=True, use_srcpos=True, srcpos=None):
        '''
        :param hrtf: the input as a B x ch=4 x L complex tensor
        :return: out: HRTF produce by the network, B x ch x L complex tensor
        '''
        returns = {}
        # print(input.shape) # torch.Size([440, 4, 128])
        srcpos = srcpos.to(input.device)
        phi = srcpos[:,1]
        theta = srcpos[:,2]
        r = srcpos[:,0]
        vec_cart_gt = sph2cart(phi,theta,r)
        # print(input.shape) # torch.Size([440, 4, 128])
        # print(vec_cart_gt.shape) # torch.Size([440, 3])
        
        in1 = {"x":input,"z":vec_cart_gt}
        z = self.Encoder1(in1) 
        z = self.Encoder2(z)
        in3 = {"x":z,"z":vec_cart_gt}
        z = self.Encoder3(in3)["x"]
        in4 = {"x":z,"z":vec_cart_gt}
        z = self.Encoder4(in4) 
        z = self.Encoder5(z) # (B,1,128)
        # print(z.shape) # torch.Size([440, 1, 128])

        returns['z'] = z 
        z = th.mean(z, dim=0) # (1,128)
        z = z.permute(1,0)
        # z = z.unsqueeze(2) # (128, 1, 1)
        # print(z.shape) # torch.Size([128, 1, 1])
        Coeff = self.Decoder(z) # (128, 4, (N+1)**2)

        Coeff = Coeff.view(self.filter_length,4,-1)

        # print(Coeff.shape)
        Coeff = th.cat((
            th.complex(Coeff[:,0:1,:],Coeff[:,1:2,:]),
            th.complex(Coeff[:,2:3,:],Coeff[:,3:4,:])), dim=1) # (128, 2, (N+1)**2)
        Coeff = Coeff.view(1, 2, -1, self.filter_length)
        # print(Coeff.shape) # torch.Size([1, 2, 400, 128])
        
        ### dammy
        returns['vec'] = vec_cart_gt # dammy
        returns['theta'] = theta # zenith
        returns['phi'] = phi # azimuth
        ###
        
        SpF = SpecialFunc(r=r, phi=phi, theta=theta, maxto=self.maxto, fft_length=self.fft_length, max_f=self.max_f)
        n_vec, m_vec = SpF.n_vec, SpF.m_vec
        Hank = SpF.SphHankel()
        Hank = th.complex(th.nan_to_num(th.real(Hank)), th.nan_to_num(th.imag(Hank))).to(Coeff.device)
        Harm = SpF.SphHarm().to(Coeff.device)

        # #### if use sph Bessel
        # Bessel = SpF.SphBessel()[None,None,:,:].to(Coeff.device)
        # Coeff = Coeff * 100 * Bessel
        # ####
        returns['coeff'] = Coeff
        
        out = th.zeros(input.shape[0], 2, SpF.filter_length, dtype = th.complex64)
        if use_cuda_forward:
            out = out.cuda()
            # print(Hank[0,:,0])
        for i,n in enumerate(n_vec):
            mask = th.from_numpy(self.N_arr_np.astype(np.float32)).clone() >= n
            if use_cuda_forward:
                mask = mask.cuda()
            Coeff_mn = Coeff[0,:,i,:] * mask[None,:] # masking
            # print(Coeff_mn.shape) # torch.Size([2, 128])
            Hank_n = Hank[:,n,:] * mask[None,:] # masking
            # print(Hank_n.shape) # torch.Size([440, 128])
            Harm_mn = Harm[:,i]
            # print(Harm_mn.shape) # torch.Size([440])
            delta = th.mul(Coeff_mn[None,:,:], Hank_n[:,None,:]) * Harm_mn[:,None,None]
            # print(delta.shape) # torch.Size([440, 2, 128])
            out += delta
        returns['output'] = out
        return returns

class HRTFApproxNetwork_AE_PINN(Net):
    def __init__(self,
                config,
                model_name='hrtf_approx_network',
                use_cuda=True):
        super().__init__(model_name, use_cuda)
        self.config = config
        self.max_f = config["max_frequency"]
        self.fft_length = config["fft_length"]
        self.filter_length = round(self.fft_length/2)
        self.f_arr = (np.linspace(0, self.max_f, self.filter_length+1))[1:] # frequency bin, (filterlengh,)
        self.k = th.from_numpy(self.f_arr*2*np.pi/343.18)
        
        ### Encoder (common)
        modules = []
        modules.extend([
                        nn.Conv1d(in_channels=4, out_channels=config["channel_En"], kernel_size=config["kernel_size"], stride=1, padding='same'),
                        nn.LayerNorm(self.filter_length),
                        nn.ReLU(), 
                        nn.Dropout(config["droprate"])])
        for l in range(config["hlayers_En"]):
            modules.extend([
                            nn.Conv1d(in_channels=config["channel_En"], out_channels=config["channel_En"], kernel_size=config["kernel_size"], stride=1, padding='same'),
                            nn.LayerNorm(self.filter_length),
                            nn.ReLU(), 
                            nn.Dropout(config["droprate"])
                            ])
        modules.extend([
                        nn.Conv1d(in_channels=config["channel_En"], out_channels=2, kernel_size=config["kernel_size"], stride=1, padding='same'),
                        nn.LayerNorm(self.filter_length),
                        nn.ReLU(), 
                        nn.Dropout(config["droprate"])
                        ])
        self.Encoder = nn.Sequential(*modules)

        ### Encoder (z)
        modules = []
        modules.extend([nn.Flatten(),
                        nn.Linear(self.filter_length, config["channel_En_z"]),
                        nn.LayerNorm(config["channel_En_z"]),
                        nn.ReLU(), 
                        nn.Dropout(config["droprate"])
                        ])
        for l in range(round(config["hlayers_En_z"]/2)):
            modules.extend([ResLinearBlock(config["channel_En_z"],config["droprate"])])
        modules.extend([nn.Linear(config["channel_En_z"], config["dim_z"]),
                        nn.LayerNorm(config["dim_z"]),
                        nn.ReLU(), 
                        ])
        self.Encoder_z = nn.Sequential(*modules)

        ### Encoder (v)
        modules = []
        modules.extend([nn.Flatten(),
                        nn.Linear(self.filter_length, config["channel_En_v"]),
                        nn.LayerNorm(config["channel_En_v"]),
                        nn.ReLU(), 
                        nn.Dropout(config["droprate"])])
        for l in range(config["hlayers_En_v"]):
            modules.extend([nn.Linear(config["channel_En_v"], config["channel_En_v"]),
                            nn.LayerNorm(config["channel_En_v"]),
                            nn.ReLU(), 
                            nn.Dropout(config["droprate"])])
        modules.extend([nn.Linear(config["channel_En_v"], 3)])
        self.Encoder_v = nn.Sequential(*modules)

        ### Decoder(z)
        modules = []
        modules.extend([
                        nn.Linear(config["dim_z"], config["channel_En_z"]),
                        nn.LayerNorm(config["channel_En_z"]),
                        nn.ReLU(), 
                        nn.Dropout(config["droprate"])
                        ])
        for l in range(config["hlayers_En_z"]):
            modules.extend([ResLinearBlock(config["channel_En_z"],config["droprate"])])
        modules.extend([nn.Linear(config["channel_En_z"], self.filter_length),
                        nn.LayerNorm(self.filter_length),
                        nn.ReLU()
                        ])
        self.Decoder_z = nn.Sequential(*modules)

        ### Decoder (common)
        modules = []
        modules.extend([
                        nn.ConvTranspose1d(in_channels=4, out_channels=config["channel_En"], kernel_size=config["kernel_size"], stride=1, padding=round((config["kernel_size"]-1)/2)),
                        nn.LayerNorm(self.filter_length),
                        nn.ReLU(), 
                        nn.Dropout(config["droprate"])])
        for l in range(config["hlayers_En"]):
            modules.extend([
                            nn.ConvTranspose1d(in_channels=config["channel_En"], out_channels=config["channel_En"], kernel_size=config["kernel_size"], stride=1, padding=round((config["kernel_size"]-1)/2)),
                            nn.LayerNorm(self.filter_length),
                            nn.ReLU(), 
                            nn.Dropout(config["droprate"])
                            ])
        modules.extend([
                        nn.ConvTranspose1d(in_channels=config["channel_En"], out_channels=4, kernel_size=config["kernel_size"], stride=1, padding=round((config["kernel_size"]-1)/2))
                        ])
        self.Decoder = nn.Sequential(*modules)
        
    def forward(self, input, use_cuda_forward=True, use_srcpos=True, srcpos=None):
        '''
        :param hrtf: the input as a B x ch=4 x L complex tensor
        :return: out: HRTF produce by the network, B x ch x L complex tensor
        '''
        returns = {}

        '''
        self.Encoder
            in: (B,4,L), out: (B,2,L)
        self.Encoder_z
            in: (B,L), out: (B,dim_z)
        self.Encoder_v
            in: (B,L), out: (B,3)
        self.Decoder_z
            in: (1,dim_z), out: (1,L)
        self.Decoder
            in: (B,4,L), out: (B,4,L)
        '''
        # print(input.shape) # torch.Size([440, 4, 128])
        B = input.shape[0]
        zz = self.Encoder(input)
        # print(zz.shape) # torch.Size([440, 2, 128])
        z = self.Encoder_z(zz[:,0,:])
        # print(z.shape) # torch.Size([440, dim_z=16])
        returns['z'] = z
        zd = self.Decoder_z(z)
        # print(zd.shape) # torch.Size([440, 128])
        zd = th.tile(th.mean(zd,dim=0),(B,1))
        # print(zd.shape) # torch.Size([440, 128])

        vec_cart = self.Encoder_v(zz[:,1,:]) # torch.Size([B, 3])
        vec_sph = cart2sph(vec_cart[:,0], vec_cart[:,1], vec_cart[:,2])
        phi   = vec_sph[:,1]   # azimuth
        theta = vec_sph[:,2]   # zenith
        r     = 1.47 * th.ones(theta.shape[0]) # fixed in HUTUBS
        
        returns['vec'] = vec_cart
        returns['phi'] = phi # azimuth
        returns['theta'] = theta # zenith

        ### Load Ground Truth
        srcpos = srcpos.to(input.device)
        L = self.filter_length
        ## flatten
        rf  = th.flatten(th.tile(srcpos[:,0], (L,1)))
        phf = th.flatten(th.tile(srcpos[:,1], (L,1)))
        thf = th.flatten(th.tile(srcpos[:,2], (L,1)))

        ## Calc. Laplacian in Spherial coord.
        # rf.requires_grad  = True
        # phf.requires_grad = True
        # thf.requires_grad = True
        # returns['rf'] = rf
        # returns['thf'] = thf
        # returns['phf'] = phf
        # r     = rf.view(L,B).permute(1,0)
        # phi   = phf.view(L,B).permute(1,0)
        # theta = thf.view(L,B).permute(1,0)
        # input_de = th.cat((zd.unsqueeze(dim=1), r.unsqueeze(dim=1), phi.unsqueeze(dim=1), theta.unsqueeze(dim=1)), dim=1)
        ##

        ## Calc. Laplacian in Cartesian coord.
        xf  = rf * th.sin(thf) * th.cos(phf)
        yf  = rf * th.sin(thf) * th.sin(phf)
        zf  = rf * th.cos(thf) 
        xf.requires_grad  = True
        yf.requires_grad = True
        zf.requires_grad = True
        returns['xf'] = xf
        returns['yf'] = yf
        returns['zf'] = zf
        x_v = xf.view(L,B).permute(1,0)
        y_v = yf.view(L,B).permute(1,0)
        z_v = zf.view(L,B).permute(1,0)
        # print(x_v)
        # sys.exit()
        input_de = th.cat((zd.unsqueeze(dim=1), x_v.unsqueeze(dim=1), y_v.unsqueeze(dim=1), z_v.unsqueeze(dim=1)), dim=1)
        ##
        
        # print(input_de.shape) # torch.Size([440, 4, 128])
        out = self.Decoder(input_de)
        # print(out.shape) # torch.Size([440, 4, 128])
        # print(out.device) # cuda:0
        out_c = th.cat((th.complex(out[:,0,:], out[:,1,:]).unsqueeze(dim=1), th.complex(out[:,2,:], out[:,3,:]).unsqueeze(dim=1)), dim=1)
        returns['output'] = out_c
        returns['output_4ch'] = out

        
        # print(rf.device) # cuda:0


        returns['k'] = self.k.to(input.device)
        returns['L'] = L
        returns['B'] = B
        return returns

class HRTFApproxNetwork_PINN(Net):
    def __init__(self,
                 model_name='hrtf_approx_network',
                 maxto=15,
                 fft_length=128,
                 max_f=16000,
                 channel = 256,
                 layers = 3,
                 droprate = 0.2,
                 use_cuda=True):
        super().__init__(model_name, use_cuda)

        self.max_f = max_f
        self.maxto = maxto
        self.fft_length = fft_length
        self.filter_length = round(self.fft_length/2)# + 1)
        self.f_np = (np.linspace(0, self.max_f, self.filter_length+1))[1:] # frequency bin, (filterlengh,)
        self.f_th = th.from_numpy(self.f_np.astype(np.float32)).clone()
        self.out_channel =  round(2*2*self.filter_length)
        self.channel = channel

        modules = []
        modules.extend([nn.Linear(5,self.channel),
                            nn.ReLU(), 
                            nn.Dropout(droprate)])
        for l in range(layers):
                modules.extend([nn.Linear(self.channel,self.channel),
                                nn.ReLU(),
                                nn.Dropout(droprate)])
        modules.append(nn.Linear(self.channel,self.out_channel))

        self.layers = nn.Sequential(*modules)

        for layer in self.layers:
            if hasattr(layer, 'weight'):
                nn.init.normal_(layer.weight, 0.0, 1e-3)
            if hasattr(layer, 'bias'):
                nn.init.normal_(layer.bias, 0.0, 1e-3)

    def forward(self, pos, use_cuda_forward=True, use_coeff = False, coeff=None):
        '''
        :param pos: the input as a (Bx) 3 tensor ([r,phi,theta])
        :return: out: HRTF produce by the network (Bx)2xfilter_length
        '''
        # print(pos.shape) # B x 3
        B = pos.shape[0]
        # print(pos.device) # cuda:0
        pos = pos.cuda()
        r, phi, theta = pos[:,0], pos[:,1], pos[:,2]
        # print(th.cos(theta).shape)
        # print(r.shape)
        pos = th.stack((th.sin(r*2*np.pi), th.cos(phi),th.sin(phi), th.cos(theta),th.sin(theta)), 1)
        # print(pos.shape) # Bx5
        pos = pos.to(th.float32)
        # print(pos.device)
        out = self.layers(pos)
        out_f = out.view(B,2,-1)
        out_c = th.complex(out_f[:,:,:round(out_f.shape[-1]/2)], out_f[:,:, round(out_f.shape[-1]/2):])

        # out_l = th.complex(out_f[:,0,:round(out_f.shape[-1]/2)], out[:, round(out.shape[1]/4):round(out.shape[1]/2)]).view(B,1,-1)
        # # print(out_l.dtype) # torch.complex64
        # out_r = th.complex(out[:,round(out.shape[1]/2):round(out.shape[1]*3/4)], out[:, round(out.shape[1]*3/4):]).view(B,1,-1)
        # out_c = th.cat((out_l,out_r),dim=1)
        # print(out.shape) # B x 2 x filterlength
        # print(out.dtype) # torch.complex64
        return {"output": out_c, "output_f":out_f, "f":self.f_th}

class HRTFApproxNetwork_old(Net):
    def __init__(self,
                 model_name='hrtf_approx_network',
                 maxto=15,
                 fft_length=128,
                 max_f=16000,
                 channel = 256,
                 use_cuda=True):
        super().__init__(model_name, use_cuda)

        self.max_f = max_f
        self.maxto = maxto
        self.fft_length = fft_length
        self.filter_length = round(self.fft_length/2)# + 1)
        self.f_arr = (np.linspace(0, self.max_f, self.filter_length+1))[1:] # frequency bin, (filterlengh,)
        self.N_arr = calcMaxOrder(f=self.f_arr,maxto=self.maxto)
        self.output_size =  round(2*2*np.sum((self.N_arr + 1)**2))

        self.channel = channel
        droprate = 0.2
        self.layers = nn.Sequential(nn.Linear(5,self.channel),
                                    nn.ReLU(), # nn.GELU(),
                                    nn.Dropout(droprate),
                                    nn.Linear(self.channel,self.channel),
                                    nn.ReLU(),
                                    nn.Dropout(droprate),
                                    nn.Linear(self.channel,self.channel),
                                    nn.ReLU(),
                                    nn.Dropout(droprate),
                                    nn.Linear(self.channel,self.channel),
                                    nn.ReLU(),
                                    nn.Dropout(droprate),
                                    nn.Linear(self.channel,self.output_size),
                                    )
        ## init
        # for layer in self.layers:
        #     if hasattr(layer, 'weight'):
        #         nn.init.normal_(layer.weight, 0.0, 1e-3)
        #     if hasattr(layer, 'bias'):
        #         nn.init.normal_(layer.bias, 0.0, 1e-3)
    def forward(self, pos):
        '''
        :param pos: the input as a (Bx) 3 tensor ([r,phi,theta])
        :return: out: HRTF produce by the network (Bx)2xfilter_length
        '''
        # print(pos.shape) # B x 3
        # B = pos.shape[0]
        # print(pos.device) # cuda:0
        r, phi, theta = pos[:,0], pos[:,1], pos[:,2]
        # print(th.cos(theta).shape)
        # print(r.shape)
        pos = th.stack((th.sin(r*2*np.pi), th.cos(phi),th.sin(phi), th.cos(theta),th.sin(theta)), 1)
        # print(pos.shape) # Bx5
        SpF = SpecialFunc(r=r, phi=phi, theta=theta, maxto=self.maxto, fft_length=self.fft_length, max_f=self.max_f)
        n_vec, m_vec = SpF.n_vec, SpF.m_vec
        Hank = SpF.SphHankel().cuda()
        # print(Hank.shape) # (B, maxto+1, filter_length)
        Harm = SpF.SphHarm().cuda()
        # print(Harm.shape) # (B, (maxto+1)**2,)
        # Coeff = SpF.RandomInput() # (2, (maxto+1)**2, filter_length)
        pos = pos.to(th.float32)
        # print(pos.device) # cuda:0
        # print(pos) # contains 0
        # print(pos.dtype)
        # if th.any(th.isnan(pos)):
        #     print("pos:NaN") # not NaN

        Coeff = self.layers(pos)

        # for l, layer in enumerate(self.layers):
        #     print(l)
        #     if hasattr(layer, 'weight'):
        #         if th.any(th.isnan(layer.weight)):
        #             print(layer.weight[:5])
        #             print(f"NaN at {l} th layer's weight")
        #             # print(pos.grad)
        #             sys.exit()
        #     if hasattr(layer, 'bias'):
        #         if th.any(th.isnan(layer.bias)):
        #             print(layer.bias[:5])
        #             print(f"NaN at {l} th layer's bias")
        #             sys.exit()
        #     if l == 0:
        #         Coeff = layer(pos)
        #     else:
        #         Coeff = layer(Coeff)
        #     if th.any(th.isnan(Coeff)):
        #         print(f"NaN at {l} th layer's output") # > NaN
        #         sys.exit()
        # print(Coeff.device) # cuda:0
        
        Coeff = transformCoeff(Coeff, n_vec, self.N_arr, self.maxto, self.filter_length)
        # print(Coeff.shape) # B,2,(maxto+1)**2,filter_length
        out = th.zeros(Coeff.shape[0], 2, SpF.filter_length, dtype = th.complex64).cuda()
        for i,n in enumerate(n_vec):
            # m = m_vec[i]
            Coeff_mn = Coeff[:,:,i,:]
            Hank_n = Hank[:,n]
            Harm_mn = Harm[:,i]
            # if th.any(th.isnan(Coeff_mn)):
            #     print(f"NaN in Coeff_mn") # > NaN
            #     sys.exit()
            # if th.any(th.isnan(Hank_n)):
            #     print(f"NaN in Hank_n") # > NaN
            #     sys.exit()
            # if th.any(th.isnan(Harm_mn)):
            #     print(f"NaN in Hanrm_mn") # > NaN
            #     sys.exit()
            # print(Coeff_mn.shape)
            # print(Hank_n[0])
            # print(th.mul(Coeff_mn, Hank_n[:,None]).shape) # B,2,filter_length
            # if th.any(th.isnan(th.mul(Coeff_mn, Hank_n[:,None]))):
            #     print("NaN")
            delta = th.mul(Coeff_mn, Hank_n[:,None]) * Harm_mn[:,None,None]
            # if th.any(th.isnan(delta)):
            #     print(f"NaN in delta") # > NaN
            #     sys.exit()
            out += delta
        # print(out.device) # cuda:0
        if th.any(th.isnan(out)):
                print(f"NaN in output") # > NaN
                sys.exit()
        return {"output": out}

class HRTFApproxNetwork_Lemaire05(Net):
    # [Lemaire+05]
    def __init__(self,
                config,
                model_name='hrtf_approx_network',
                use_cuda=True):
        super().__init__(model_name, use_cuda)
        self.config = config
        self.max_f = config["max_frequency"]
        self.fft_length = config["fft_length"]
        self.filter_length = round(self.fft_length/2)
        self.f_arr = (np.linspace(0, self.max_f, self.filter_length+1))[1:] # frequency bin, (filterlengh,)
        self.to_in = round(config["num_pts"]**0.5 - 1)

        
        in_size = 2*self.filter_length + 6 # left, right, pos_in, pos_desired
        out_size = 2*self.filter_length # left, right
        modules = []
        modules.extend([nn.Linear(in_size,config["dim_z"]),
                        nn.LayerNorm(config["dim_z"]),
                        nn.Sigmoid(),
                        nn.Linear(config["dim_z"],out_size),
                        nn.Sigmoid()
                        ])
        self.layers = nn.Sequential(*modules)

    def forward(self, input, srcpos):
        '''
        :param input : (B, 2L, S) float tensor
        :param srcpos: (B,  3, S) float tensor
        :return out  : (B,  2, L, S) float tensor
        '''
        # print(input.shape)  # torch.Size([440, 256, 77])
        # print(srcpos.shape) # torch.Size([440, 3, 77])

        returns = {}
        L = self.filter_length
        # B = input.shape[0]
        # S = input.shape[-1]

        # mean_logmag = th.mean(input)
        # std_logmag = th.std(input, dim=[0,1,2])

        # max_in = th.max(input) 
        # min_in = th.min(input) 
        # print([max_in,min_in]) # [tensor(22.5611, device='cuda:0'), tensor(-57.7401, device='cuda:0')]
        max_in = 30
        min_in = -70
        
        # input = (input - mean_logmag) / std_logmag
        input = (input - min_in) / (max_in - min_in) 
        # print([th.max(input),th.min(input)])
        # sys.exit()

        srcpos = srcpos.cuda()
        vec_cart_all = sph2cart(srcpos[:,1],srcpos[:,2],srcpos[:,0]) # all 440 pts
        # print(vec_cart_all.shape) # torch.Size([440, 3, 77])

        idx_t_des = aprox_t_des(pts=vec_cart_all[:,:,0]/srcpos[0,0,0], t=self.to_in, plot=False)
        vec_cart_tdes = vec_cart_all[idx_t_des,:,:] # sampled (t+1)^2 pts
        # print(vec_cart_tdes.shape) # torch.Size([169, 3, 77])

        idx_in = nearest_pts(vec_cart_tdes[:,:,0], vec_cart_all[:,:,0])
        vec_cart_in = vec_cart_tdes[idx_in,:,:]
        vec_sphe_in = srcpos[idx_in,:,:]
        hrtf_in = input[idx_in,:,:] # nearest 440 pts
        # print(vec_cart_in.shape) # torch.Size([440, 3, 77])
        # print(vec_cart_in[:5,:,0])
        # tensor([[-2.5526e-01, -2.2316e-08, -1.4477e+00],
        # [ 2.5526e-01,  0.0000e+00, -1.4477e+00],
        # [ 1.2763e-01,  2.2106e-01, -1.4477e+00],
        # [ 1.2763e-01,  2.2106e-01, -1.4477e+00],
        # [-2.5526e-01, -2.2316e-08, -1.4477e+00]], device='cuda:0')

        # identity dammy
        # network_in = th.cat((input,vec_cart_all,vec_cart_all),dim=1).permute(2,0,1)
        
        # network_in = th.cat((hrtf_in,vec_cart_in,vec_cart_all),dim=1).permute(2,0,1)
        network_in = th.cat((hrtf_in,vec_sphe_in,srcpos),dim=1).permute(2,0,1)
        # # print(network_in.shape) # torch.Size([77, 440, 262])
        out = self.layers(network_in)

        # print([th.max(out),th.min(out)])
        # out = (out * std_logmag) + mean_logmag
        out = out * (max_in-min_in) + min_in
        # print(out.shape) # torch.Size([77, 440, 256])
        # print([th.max(out),th.min(out)])

        out_l = out[:,:,:L].unsqueeze(2)
        out_r = out[:,:,L:].unsqueeze(2)
        out_lr = th.cat((out_l,out_r),dim=2).to(th.complex64)
        # print(out_lr.shape) # torch.Size([77, 440, 2, 128])
        returns['output'] = out_lr.permute(1,2,3,0)

        return returns
