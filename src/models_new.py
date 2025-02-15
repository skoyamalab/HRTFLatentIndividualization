# from termios import B115200
# from turtle import forward
import math
import numpy as np
import torch as th
import torchaudio as ta
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from torch.distributions.multivariate_normal import MultivariateNormal
from src.utils import Net, SpecialFunc, calcMaxOrder, transformCoeff, transformCoeff_MagdBPhase, sph_harm_nmvec, cart2sph, sph2cart, aprox_t_des, aprox_reg_poly, nearest_pts, transformCoeff_MagdBPhase_CNN, plane_sample, parallel_planes_sample, replace_activation_function
from src.lebedev import genGrid
from src.arcfacemetrics import ArcMarginProduct
import sys
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
        #------ Fourier Feature Mapping -----
        if len(self.config["data_kind_ffm"]) > 0:
            # self.ffm = {}
            for data_kind in self.config["data_kind_ffm"]:
                # self.ffm[data_kind] = FourierFeatureMapping(num_features=self.config["num_ff"][data_kind], dim_data=self.config["dim_data_hyper"][data_kind], trainable=self.config["ffm_trainable"])
                exec(f'self.ffm_{data_kind} = FourierFeatureMapping(num_features=self.config["num_ff"][data_kind], dim_data=self.config["dim_data_hyper"][data_kind], trainable=self.config["ffm_trainable"])')

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
                    modules.extend([
                        HyperLinearBlock(ch_in=ch_in, ch_out=ch_out, ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"], input_size=input_size_en_0, post_prcs=post_prcs),
                    ])
                else:
                    raise NotImplementedError

            self.en_0 = nn.Sequential(*modules)
        # x:(2S,L,B',1)

        #------ Decoder -------------------------

        ch_in = config["dim_z"]
        
        if self.config["hlayers_De_-1"] > 0:
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
        returns = {}
        db_name = data["db_name"][0]

        for data_kind in self.config['data_kind_interp']:
            if data_kind in ['HRTF_mag','HRTF']:
                LL = self.filter_length
            elif data_kind in ['HRIR']:
                LL = self.filter_length * 2

        S,B,_ = data["SrcPos"].shape
        
        #===v sampling v====
        if self.config["pln_smp"]:
            idx_mes_pos = plane_sample(pts=data['SrcPos_Cart'][0], axes=self.config["pln_smp_axes"], thr=0.01)
        elif self.config["pln_smp_paral"]:
            idx_mes_pos = parallel_planes_sample(pts=data['SrcPos_Cart'][0], values=th.tensor([-.5,0.0,.5]).to(data['SrcPos'].device)*(data['SrcPos'][0,0,0]), axis=self.config['pln_smp_paral_axis'], thr=0.01)
        elif self.config["num_pts"] >= min(B,362):
            idx_mes_pos = range(0,B) 
        elif self.config["random_sample"] and mode == 'train':
            perm = th.randperm(B)
            idx_mes_pos = perm[:self.config["num_pts"]]
        elif self.config["num_pts"] < 9:
            idx_mes_pos = aprox_reg_poly(pts=data['SrcPos_Cart'][0]/(data['SrcPos'][0,0,0]), num_pts=self.config["num_pts"], db_name=db_name)
        else:
            self.t = round(self.config["num_pts"]**0.5-1)
            idx_mes_pos = aprox_t_des(pts=data['SrcPos_Cart'][0]/(data['SrcPos'][0,0,0]), t=self.t, plot=False, db_name=db_name)

        returns["idx_mes_pos"] = idx_mes_pos
        B_mes = len(idx_mes_pos)
        data_mes = {}

        #=== Standardize ====
        for data_kind in self.config['data_kind_interp']:
            if data_kind in ['HRTF_mag','HRTF','HRIR','ITD']:
                data[data_kind] = (data[data_kind] - self.config["Val_Standardize"][data_kind][db_name]["mean"]) / (self.config["Val_Standardize"][data_kind][db_name]["std"])
        data['SrcPos_Cart'] = data['SrcPos_Cart']/(self.config["SrcPos_Cart_norm"])
        data['Freq'] = (th.arange(self.filter_length+1)[1:]/self.filter_length).to(data['HRTF_mag'].device).unsqueeze(-1)

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
            for data_str in ['data','data_mes']:
                if data_kind in eval(data_str):
                    # eval(data_str)[data_kind] = self.ffm[data_kind](eval(data_str)[data_kind])
                    exec(f'eval(data_str)[data_kind] = self.ffm_{data_kind}(eval(data_str)[data_kind])')
                    if data_kind == 'SrcPos_Cart':
                        # eval(data_str)['SrcPos_Cart_lr_flip'] = self.ffm[data_kind](eval(data_str)['SrcPos_Cart_lr_flip'])
                        exec(f"eval(data_str)['SrcPos_Cart_lr_flip'] = self.ffm_{data_kind}(eval(data_str)['SrcPos_Cart_lr_flip'])")

        device = data['HRTF_mag'].device
        hyper_en_x = th.zeros(S,B_mes,0,1, device=device, dtype=th.float32)
        for data_kind in self.config["data_kind_interp"]:
            if data_kind in ['ITD']:
                hyper_en_x = th.cat((hyper_en_x, data_mes[data_kind][:,:,None,None]), dim=2) # (S,B_mes,1,1)
            elif data_kind in ['HRTF_mag','HRIR']:
                hyper_en_x = th.cat((hyper_en_x, data_mes[data_kind][:,:,0,:,None], data_mes[data_kind][:,:,1,:,None]), dim=2) # (S,B_mes,2L,1)
        
        assert self.config["data_kind_interp"] == ['HRTF_mag','ITD']
        
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

        latents = self.en_0({
            "x": hyper_en_x, # (S,B_mes, 1 or 2LL or 2LL+1, 1)
            "z": hyper_en_z, # (S,B_mes, 1 or 2LL or 2LL+1, *)
        })["x"]
        # (S, B_mes, 2 * LL + 1, d)

        returns["z_bm"] = latents
        latents = th.mean(latents, dim=self.config["mid_mean_dim"], keepdim=True)  # (S, 1, 2 * LL + 1, d)
        returns["z"] = latents
        assert self.config["mid_mean_dim"] == (1,)
        latents = latents.tile(1,B,1,1)  # (S, B, 2 * LL + 1, d)

        assert self.config["data_kind_interp"] == ['HRTF_mag','ITD']
        hyper_de_z = th.zeros(S, B, 2 * LL + 1, 0, device=device, dtype=th.float32)
        for data_kind in self.config["data_kind_hyper_en"]:
            if data_kind in ['SrcPos_Cart']:
                # data[data_kind]:               (S, B, 3) -> (S, B, 1, 3) -> (S, B, LL, 3)
                # data[f'{data_kind}_lr_flip']:  (S, B, 3) -> (S, B, 1, 3) -> (S, B, LL, 3)
                # data[data_kind]:               (S, B, 3) -> (S, B, 1, 3)
                # cat                         :  (S, B, 2 * LL + 1, 3)
                hyper_de_z = th.cat((hyper_de_z, th.cat((data[data_kind].unsqueeze(2).tile(1,1,LL,1), data[f'{data_kind}_lr_flip'].unsqueeze(2).tile(1,1,LL,1), data[data_kind].unsqueeze(2)), dim=2)), dim=3) # (S,B,2*LL+1,3 or num_ff)
            elif data_kind in ['Freq']:
                freq_tensor = data[data_kind].tile(2,1)
                freq_dammy  = th.zeros_like(freq_tensor)[0:1,:]
                hyper_de_z = th.cat((hyper_de_z, th.cat((freq_tensor,freq_dammy), dim=0)[None,None,:,:].tile(S,B,1,1)), dim=3)  # (S,B,2*LL+1,1 or num_ff)

        delta = th.cat((th.zeros(2*LL, device=device, dtype=th.float32), th.ones(1, device=device, dtype=th.float32)), dim=0)
        hyper_de_z = th.cat((hyper_de_z, delta[None,None,:,None].tile(S,B,1,1)), dim=3)

        out_f = self.de_m1({
            "x": latents,    # (S, B, 2LL+1, d)
            "z": hyper_de_z, # (S, B, 2LL+1, *)
        })["x"]


        assert self.config["data_kind_interp"] == ['HRTF_mag','ITD']
        for data_kind in self.config["data_kind_interp"]:
            if data_kind == 'ITD':
                returns['ITD'] = out_f[:,:,0,0]
                out_f = out_f[:,:,0:,:]
            else:
                returns[data_kind] = th.cat((out_f[:,:,None,:LL,0], out_f[:,:,None,LL:2*LL,0]), dim=2) # (S,B,2,LL)
                out_f = out_f[:,:,2*LL:,:]
        
        for data_kind in self.config["data_kind_interp"]:
            returns[data_kind] = returns[data_kind] * self.config["Val_Standardize"][data_kind][db_name]["std"] + self.config["Val_Standardize"][data_kind][db_name]["mean"]

        return returns
