import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

#src.
from utils import Net, sph_harm_nmvec

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
                use_cuda=True,
                device='cuda:0'):
        super().__init__(model_name, use_cuda, device)
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
        db_name = data["db_name"][0]
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

        # else:
        

        #     # 'SrcPos_Cart':  (S,B,3) torch.float32 [x,y,z]
        #     # 'HRTF':         (S,B,2,L,2) torch.float32 not implemented yet
        #     # 'HRTF_mag':     (S,B,2,L) torch.float32
        #     # 'HRIR':         (S,B,2,2L) torch.float32
        #     input_rs = input_rs.permute(2,0,1) # (S,B',2L+3) or (S,B',1+3)
        #     # B_p = input_rs.shape[1] # B'
        #     if self.config["DNN_for_interp_ITD"]:
        #         input_rs_x = input_rs[:,:,0:1] # (S,B',1)
        #         input_rs_z = input_rs[:,:,-3:]  # (S,B',3)
        #     elif self.config["use_lr_aug"]:
        #         input_rs_x = th.cat((input_rs[:,:,0*L:1*L],input_rs[:,:,1*L:2*L]),dim=0) # (2S,B',L)
        #         pos_l = input_rs[:,:,-3:]
        #         pos_r = pos_l * th.tensor([1,-1,1],device=pos_l.device)[None, None, :] # y座標 反転
        #         input_rs_z = th.cat((pos_l, pos_r), dim=0)  # (2S,B',3)
        #     else:
        #         raise NotImplementedError
            
        #     if self.config["use_ffm_for_hyper_cartpos"]:
        #         input_rs_z = self.ffm_cartpos(input_rs_z/self.config["ffm_norm_cartpos"])

        #     if self.config["in_mag_pc"]:
        #         V = th.load("v_mag_train.pt").to(input_rs_x.device) # (L,L)
        #         q = self.config['num_pc']
        #         input_rs_x = input_rs_x.reshape(-1,L) # (2S,B',L) -> (2S*B',L)
        #         input_rs_x = th.matmul(input_rs_x, V[:,:q]) # (2S*B',L)@(L,q) = (2S*B',q)
        #         input_rs_x = input_rs_x.reshape(2*S,B_p,q) # (2S, B',q)
        #         L = q
        #         # print(input_rs_x.shape)
        #     elif self.config["in_latent"]:
        #         input_rs_x = self.pre_nn(input_rs_x)
        #         q = self.config["num_latents"]
        #         L = q
        #     elif self.config["in_cnn"]:
        #         input_rs_x = input_rs_x.reshape(2*S*B_p,1,L)
        #         input_rs_x = self.pre_cnn(input_rs_x) # (2SB',1, Q)
        #         input_rs_x = input_rs_x.reshape(2*S,B_p,-1)  # (2S, B', Q)
        #         L = input_rs_x.shape[-1]
        #     elif self.config["in_cnn_ch"]:
        #         # print(input_rs_x.shape) # debug_0805
        #         input_rs_x = input_rs_x.reshape(2*S*B_p,1,L)
        #         # print(input_rs_x.shape) # debug_0805
        #         input_rs_x = self.pre_cnn(input_rs_x) # (2SB',ch, L//ch)
        #         cnn_max_ch = input_rs_x.shape[1]
        #         # print(input_rs_x.shape) # debug_0805
        #         input_rs_x = input_rs_x.reshape(2*S,B_p,-1)  # (2S, B', L)
        #         L = input_rs_x.shape[-1]
        #         # print("------") # debug_0805
            


        #     input_rs_x = input_rs_x.permute(0,2,1)  # (2S,L,B') or (S,1,B')
        #     if self.config["DNN_for_interp_ITD"]:
        #         # input_rs_z = input_rs_z.unsqueeze(1) #  (S,1,B',3)
        #         pass # input_rs_z (S,B',3)
        #     else:
        #         if not self.config["hyperconv_en_0"]:
        #             input_rs_z = th.tile(input_rs_z.unsqueeze(1), (1,L,1,1)) #  (2S,L,B',3)

        #         if self.config["use_RSH_for_hyper_en"]:
        #             r_en = th.tile(r[idx_t_des,0], (2,))
        #             theta_en = th.tile(theta[idx_t_des,0], (2,))
        #             phi_en = th.cat((phi[idx_t_des,0], 2*np.pi-phi[idx_t_des,0]))
        #             SpF_en = SpecialFunc(r=r_en, phi=phi_en, theta=theta_en, maxto=self.config["max_truncation_order"], fft_length=self.fft_length, max_f=self.max_f)
        #             RSH_en = SpF_en.RealSphHarm().to(input_rs_x.device)
        #             # print(SH_en.shape) # (2*B',dimz=(N+1)^2) # torch.Size([880, 64])
        #             RSH_en = th.tile(RSH_en.reshape(1,1,2*B_p,-1), (S,L,1,1))
        #             # print(RSH_en.shape) # (S,L,2B',(N+1)^2) # torch.Size([16, 128, 880, 64])
        #             RSH_en = th.cat((RSH_en[:,:,:B_p,:], RSH_en[:,:,B_p:,:]), dim=0)
        #             # print(RSH_en.shape) # (2S,L,B',(N+1)^2)
        #             # sys.exit()
        #             input_rs_z = RSH_en.to(dtype=input_rs_z.dtype)
                    
        #         if not self.config["hyperconv_en_0"] and self.config["use_freq_for_hyper_en"]:
        #             # freq_th = th.from_numpy(self.f_arr.astype(np.float32)).clone().to(input_rs_z.device) # (L)
        #             # freq_th = freq_th / (freq_th[-1]/2) -1  # (0,1]
        #             freq_th = (th.arange(L+1)[1:]/(L/2)-1).to(input_rs_z.device)
        #             if not self.config["hyperconv_en_0"]:
        #                 freq_th = th.tile(freq_th.view(1,L,1,1), (2*S,1,B_p,1)) # (2S,L,B',1)
        #                 if self.config["use_ffm_for_hyper_freq"]:
        #                     freq_th = self.ffm_freq(freq_th)
        #                 input_rs_z = th.cat((input_rs_z,freq_th), dim=-1) #  (2S,L,B',4)
        #             else:
        #                 if self.config["hyperconv_en_0_pad"] == 'same':
        #                     freq_th = freq_th.reshape(1,L).tile(L,1) # (L,L)
        #                     idx_gather = th.arange(L).to(freq_th.device)
        #                     idx_offset = th.arange(L).to(freq_th.device)
        #                     idx_gather = (idx_gather[None,:] + idx_offset[:,None]) % L
        #                     ks = self.config["hyperconv_en_0_ks"]
        #                     freq_th = th.gather(freq_th, -1, idx_gather)[:,:ks] # (L, ks)
        #                     freq_th = freq_th.reshape(1,L,1,ks).tile(2*S,1,B_p,1) # (2S, L, B', ks)
        #                     input_rs_z = th.cat((input_rs_z,freq_th), dim=-1) # (2S, L, B', 3+ks)
        #                 else:
        #                     raise NotImplementedError
            
        #     if self.config["use_Bp_for_hyper_en"]:
        #         Bp_norm = self.config["Bp_norm"] if "Bp_norm" in self.config.keys() else B
        #         if self.config["DNN_for_interp_ITD"]:
        #             B_p_th = th.tensor([B_p/Bp_norm*2-1]).reshape(1,1,1).tile(S,B_p,1).to(input_rs_z.device) # (S,B',1)
        #         elif self.config["hyperconv_en_0"]:
        #             B_p_th = th.tensor([B_p/Bp_norm*2-1]).reshape(1,1,1).tile(2*S,B_p,1).to(input_rs_z.device)
        #         else:
        #             B_p_th = th.tensor([B_p/Bp_norm*2-1]).reshape(1,1,1,1).tile(2*S,L,B_p,1).to(input_rs_z.device)
        #         if self.config["use_ffm_for_hyper_Bp"]:
        #             B_p_th = self.ffm_Bp(B_p_th)
        #         # print(input_rs_z.shape)
        #         # print(B_p_th.shape)
        #         # print(B_p_th[0,0,0,0])
        #         input_rs_z = th.cat((input_rs_z, B_p_th), dim=-1)  #  (2S,L,B',3/4) or (2S,B',3/4) or (S,B',3/4)

        #     #==============================
        #     if self.config["hlayers_En_0"]>0:
        #         # input_rs_x,  # (2S,L,B')
        #         # input_rs_z,  # (2S,L,B', 3or4)
        #         if self.config["DNN_for_interp_ITD"]:
        #             input_rs_x_0 = input_rs_x.reshape(S, B_p, 1)
        #             input_rs_z_0 = input_rs_z.reshape(S, B_p,-1)
        #             input_hyper_en_0 = {
        #                 "x": input_rs_x_0,  # (S,B',1)
        #                 "z": input_rs_z_0,  # (S,B', 3or4)
        #             }
        #             input_rs_x = self.en_0(input_hyper_en_0)["x"] # x:(S,1,B',1 or d)
        #             input_rs_x = input_rs_x.reshape(S, 1, B_p, -1)
        #         elif self.config["hyperlinear_en_0"]:
        #             input_rs_x_0 = input_rs_x.reshape(2*S, L*B_p, 1)
        #             input_rs_z_0 = input_rs_z.reshape(2*S, L*B_p,-1)
        #             input_hyper_en_0 = {
        #                 "x": input_rs_x_0,  # (2S,LB',1)
        #                 "z": input_rs_z_0,  # (2S,LB', 3or4)
        #             }
        #             # print(input_hyper_en_0["x"].dtype)
        #             # print(input_hyper_en_0["z"].dtype)
        #             input_rs_x = self.en_0(input_hyper_en_0)["x"] # x:(2S,L,B',1 or d)
        #             input_rs_x = input_rs_x.reshape(2*S, L, B_p, -1)
        #         elif self.config["hyperconv_en_0"]:
        #             input_rs_x_0 = input_rs_x.permute(0,2,1).reshape(2*S*B_p, 1, L) # (2S,L,B') -> (2S,B',L) -> (2SB',1,L)
        #             input_rs_z_0 = input_rs_z.reshape(2*S*B_p,-1) # (2S,B',3/4) -> (2SB',3/4)
        #             input_hyper_en_0 = {
        #                 "x": input_rs_x_0,  # (2SB',1or2,L)
        #                 "z": input_rs_z_0,  # (2SB',3or4)
        #             }
        #             if self.config["use_freq_for_hyper_en"]:
        #                 freq_th = (th.arange(L+1)[1:]/(L/2)-1).to(input_rs_x_0.device)
        #                 freq_th = th.tile(freq_th.view(1,1,L), (2*S*B_p,1,1)) # (2SB',1,L)
        #                 input_hyper_en_0["f"] = freq_th

        #             input_rs_x = self.en_0(input_hyper_en_0)["x"] # (2SB',d, L)
        #             input_rs_x = input_rs_x.reshape(2*S, B_p, -1, L)
        #             input_rs_x = input_rs_x.permute(0,3,1,2) # (2S, L, B',d)

        #             # print(f"input_rs_x.shape: {input_rs_x.shape}") # dbg

        #         elif self.config["hyperconv_FD_en_0"]:
        #             input_rs_x_0 = input_rs_x.permute(0,2,1) # (2S,B',L)
        #             input_rs_x_0 = input_rs_x_0.reshape(2*S, B_p, 1, L) # (2S,B',1, L)
        #             input_rs_z_0 = input_rs_z.permute(0,2,1,3) # (2S, B', L, 3+ks)
        #             input_rs_z_0 = input_rs_z_0[0,:,:,:] # (B', L, 3+ks)
        #             input_hyper_en_0 = {
        #                 "x": input_rs_x_0,  # (2S,B',1, L)
        #                 "z": input_rs_z_0,  # (B', L, 3+ks)
        #             }
        #             input_rs_x = self.en_0(input_hyper_en_0)["x"] # (2S, B', d, L)
        #             input_rs_x = input_rs_x.permute(0,3,1,2)  # (2S, L, B', d)
        #         if not self.config["aggregation_mean"]:
        #             input_rs_x = input_rs_x.reshape(2*S, L, B_p)
        #     else:
        #         pass
            

        #     if self.config["aggregation_mean"]:
        #         # print(input_rs_x.shape) # torch.Size([8, 128, 440, 441])
        #         if self.config["z_norm"]:
        #             input_rs_x = F.normalize(input_rs_x, dim=-1)
        #         returns["z_bm"] = input_rs_x # (2S, L, B', d)
        #         latents = th.mean(input_rs_x, dim=2) # (2S, L, d)
        #         if self.config["z_norm"]:
        #             latents = F.normalize(latents, dim=-1)
        #         # print(latents.shape) # torch.Size([8, 128, 441])
        #     else:
        #         # print(f'max|input_rs_x|:{th.max(th.abs(input_rs_x))}') # tensor(9.4058, device='cuda:0') # nan_debug
        #         # print(th.max(th.abs(input_rs_z))) # tensor(1.4700, device='cuda:0')
        #         input_hyper_en_1 = {
        #             "x": input_rs_x,  # (2S,L,B')
        #             "z": input_rs_z,  # (2S,L,B', 3or4)
        #         }
        #         # print(input_rs_x.shape) # torch.Size([64, 128, 440])
        #         # print(input_rs_z.shape) # torch.Size([64, 128, 440, 3])
        #         latents = self.en_1(input_hyper_en_1)
        #         returns["weight_en_1"] = latents["weight"] # (S,L,B',ch_en)
        #         if self.config["weight_en_sph_harm"]:
        #             # flip のこと考えてなかった．要修正
        #             SpF_en = SpecialFunc(r=r[idx_t_des,0], phi=phi[idx_t_des,0], theta=theta[idx_t_des,0], maxto=self.config["max_truncation_order"], fft_length=self.fft_length, max_f=self.max_f)
        #             RSH_en = SpF_en.RealSphHarm().to(input_rs_x.device)
        #             # print(SH_en.shape) # (B',dimz=(N+1)^2) # torch.Size([440, 64])
        #             if self.config["pinv_FIAE_en"]:
        #                 returns["RSH_en"] = RSH_en
        #             else:
        #                 returns["RSH_en"] = RSH_en / (4*np.pi)

        #         latents = latents["x"]
        #         # print(latents.shape) # (2S,L,ch_en or dim_z) # torch.Size([64, 128, 128])
        #         # print(f'max|latents|:{th.max(th.abs(latents))}') # tensor(281.0440, device='cuda:0', grad_fn=<MaxBackward1>) # nan_debug
            

        #     if self.config["hlayers_En_z"] > 0:
        #         if self.config["use_freq_for_hyper_en"]:
        #             input_hyper_en_2 = {
        #                 "x": latents,  # (2S,L,ch_en)
        #                 "z": freq_th[:,:,0,:],  # (2S,L, 1)
        #             }
        #             latents = self.en_2(input_hyper_en_2)["x"]
        #         else:
        #             latents = self.en_2(latents)

        #     #======== attention to reflect all mes.pos. info. ====================
        #     # print(th.max(th.abs(latents)))
        #     if self.config["pos_all_en_attn"]:
        #         #=== key ====
        #         if self.config["use_freq_for_hyper_en"]:
        #             input_rs_z_attn = input_rs_z[0,:,:,:] #  (2S,L,B',4) -> (L,B',4)
        #         else:
        #             input_rs_z_attn = input_rs_z[0,0,:,:] #  (2S,L,B',4) -> (B',4)
        #         mat_en_k = self.en_attn_k(input_rs_z_attn) # (L,B',d_z) or (B',d_z)
        #         if self.config["pos_all_en_attn_gram"]:
        #             mat_en_q = self.en_attn_k(input_rs_z_attn) # (L,B',d_z) or (B',d_z)
        #         else:
        #             mat_en_q = self.en_attn_q(input_rs_z_attn) # (L,B',d_z) or (B',d_z)

        #         if not self.config["use_freq_for_hyper_en"]:
        #             mat_en_k = th.tile(mat_en_k.unsqueeze(0), (L,1,1)) # (L,B',d_z)
        #             mat_en_q = th.tile(mat_en_q.unsqueeze(0), (L,1,1)) # (L,B',d_z)
                
        #         mat_en_kq = th.bmm(mat_en_k.permute(0,2,1), mat_en_q) # (L,d_z,d_z)
        #         mat_en_kq = th.tile(mat_en_kq.unsqueeze(0), (2*S,1,1,1))  # (2*S,L,d_z,d_z)
        #         mat_en_kq = th.reshape(mat_en_kq, (2*S*L,self.config["dim_z"],self.config["dim_z"])) # (2*S*L,d_z,d_z)
        #         mat_en_kq += self.config["pos_all_en_attn_eye_coeff"]*th.eye(self.config["dim_z"])[None,:,:].to(mat_en_kq.device)
        #         latents = th.reshape(latents, (2*S*L,self.config["dim_z"],1)) # # (2S,L,d_z) -> (2SL, d_z, 1)
        #         latents = th.bmm(mat_en_kq, latents) # (2SL, d_z, 1)
        #         latents = th.reshape(latents, (2*S,L,self.config["dim_z"]))
        #     else:
        #         pass

        #     #===== Mid-CNN -------------------------------
        #     if self.config["use_mid_conv"]:
        #         if self.config["res_mid_conv"]:
        #             latents_res = latents.clone()
        #         latents = latents.permute(0,2,1) # 2S, d_z, L
        #         if ["use_freq_for_mid_conv"]:
        #             # freq_th = th.from_numpy(self.f_arr.astype(np.float32)).clone().to(latents.device) # (L)
        #             # freq_th = freq_th / (freq_th[-1]/2) -1  # (0,1]
        #             freq_th = (th.arange(L+1)[1:]/(L/2)-1).to(latents.device)
        #             freq_th = th.tile(freq_th.view(1,1,L), (2*S,1,1)) # (2S,1,L)
        #             latents = th.cat((latents,freq_th),dim=1) # (2S,d_z+1,L)
        #         latents = self.mid_cnn(latents)
        #         latents = latents.permute(0,2,1) # 2S, L, d_z
        #         if self.config["res_mid_conv"]:
        #             latents = latents + latents_res

        #     if self.config["use_mid_conv_simple"]:
        #         latents = latents.permute(0,2,1) # 2S, d_z, L
        #         if self.config["ch_mid_conv_simple"] == 1:
        #             latents = latents.reshape(-1,1, latents.shape[-1]) # 2S, 1, L
        #         if self.config["use_freq_for_mid_conv"]:
        #             freq_th = (th.arange(L+1)[1:]/(L/2)-1).to(latents.device)
        #             freq_th = th.tile(freq_th.view(1,1,L), (latents.shape[0],1,1)) # (2S,1,L) or (2S*d,1,L)
        #             latents = th.cat((latents,freq_th),dim=1) # (2S,1+d,L) or  # (2S*d,1+1,L)
        #         latents = self.mid_cnn(latents)
        #         latents = latents.reshape(2*S,-1,L)  # 2S, d, L
        #         latents = latents.permute(0,2,1)  # 2S, L, d_z

        #     #=============================================
        #     if self.config["use_mid_linear"]:
        #         latents = latents.permute(0,2,1) # 2S, d_z, L
        #         latents = self.mid_linear(latents) # 2S, d_z, 1
        #         latents = latents.tile(1,1,self.config['fft_length']//2).permute(0,2,1)  # 2S, L, d_z

        #     # print(th.max(th.abs(latents)))
        #     # sys.exit()
        #     # print(latents.shape) # (2S,L,dim_z) # torch.Size([64, 128, 64]) # torch.Size([4, 128, 441])
        #     # sys.exit()
        #     returns["z"] = latents

        #     if self.config["hlayers_De_z"] > 0:
        #         if self.config["use_freq_for_hyper_de"]:
        #             input_hyper_de_1 = {
        #                 "x": latents,  # (2S,L,dim_z)
        #                 "z": freq_th[:,:,0,:],  # (2S,L, 1)
        #             }
        #             output = self.de_1(input_hyper_de_1)["x"]
        #         else:
        #             output = self.de_1(latents)
        #         # output (2S,L,ch_de)
        #     else:
        #         output = latents # (2S, L, d) or (S,1,d)
        #     if self.config["DNN_for_interp_ITD"]:
        #         input_rs_z_out = vec_cart_gt.permute(2,0,1) # (S,B,3)
        #     else:
        #         pos_l_out = vec_cart_gt.permute(2,0,1) # (S,B,3)
        #         pos_r_out = pos_l_out * th.tensor([1,-1,1],device=pos_l_out.device)[None,None, :] # y座標 反転
        #         input_rs_z_out = th.cat((pos_l_out, pos_r_out), dim=0)  # (2S,B,3)
        #         if not self.config["hyperconv_de_-1"]:
        #             input_rs_z_out = th.tile(input_rs_z_out.unsqueeze(1), (1,L,1,1)) #  (2S,L,B,3)
        #     if self.config["use_ffm_for_hyper_cartpos"]:
        #         input_rs_z_out = self.ffm_cartpos(input_rs_z_out/self.config["ffm_norm_cartpos"])

        #     if self.config["use_RSH_for_hyper_de"]:
        #         r_de = th.tile(r[:,0], (2,))
        #         theta_de = th.tile(theta[:,0], (2,))
        #         phi_de = th.cat((phi[:,0], 2*np.pi-phi[:,0]))
        #         SpF_de = SpecialFunc(r=r_de, phi=phi_de, theta=theta_de, maxto=self.config["max_truncation_order"], fft_length=self.fft_length, max_f=self.max_f)
        #         RSH_de = SpF_de.RealSphHarm().to(output.device)
        #         # print(RSH_de.shape) # (2*B,dimz=(N+1)^2) # torch.Size([880, 64])
        #         RSH_de = th.tile(RSH_de.reshape(1,1,2*B,-1), (S,L,1,1))
        #         # print(RSH_de.shape) # (S,L,2B,(N+1)^2) # torch.Size([16, 128, 880, 64])
        #         RSH_de = th.cat((RSH_de[:,:,:B,:], RSH_de[:,:,B:,:]), dim=0)
        #         # print(RSH_de.shape) # (2S,L,B,(N+1)^2)
        #         input_rs_z_out = RSH_de.to(dtype=input_rs_z_out.dtype)
        #         # sys.exit()

        #     if self.config["use_freq_for_hyper_de"]:
        #         freq_th_out = (th.arange(L+1)[1:]/(L/2)-1).to(input_rs_z_out.device)
        #         if not self.config["hyperconv_de_-1"]:
        #             freq_th_out = th.tile(freq_th_out.view(1,L,1,1), (2*S,1,B,1)) # (2S,L,B,1)
        #             if self.config["use_ffm_for_hyper_freq"]:
        #                 freq_th_out = self.ffm_freq(freq_th_out)
        #             input_rs_z_out = th.cat((input_rs_z_out,freq_th_out), dim=-1) #  (2S,L,B,4)

        #         else:
        #             if self.config["hyperconv_de_-1_pad"] == 'same':
        #                 freq_th_out = freq_th_out.reshape(1,L).tile(L,1) # (L,L)
        #                 idx_gather = th.arange(L).to(freq_th_out.device)
        #                 idx_offset = th.arange(L).to(freq_th_out.device)
        #                 idx_gather = (idx_gather[None,:] + idx_offset[:,None]) % L
        #                 ks = self.config["hyperconv_de_-1_ks"]
        #                 freq_th_out = th.gather(freq_th_out, -1, idx_gather)[:,:ks] # (L, ks)
        #                 freq_th_out = freq_th_out.reshape(1,L,1,ks).tile(2*S,1,B,1) # (2S, L, B, ks)
        #                 input_rs_z_out = th.cat((input_rs_z_out,freq_th_out), dim=-1) # (2S, L, B, 3+ks)
        #             else:
        #                 raise NotImplementedError

        #     # print(input_rs_z_out.shape)
        #     if not self.config["de_2_skip"]:
        #         input_hyper_de_2 = {
        #             "x": output,  # (2S,L,ch_de or dim_z)
        #             "z": input_rs_z_out,  # (2S,L,B,3 or 4)
        #         }
        #         out_f = self.de_2(input_hyper_de_2) # (2S,L,B)
        #         returns["weight_de_2"] = out_f["weight"] # (S,L,B,ch_de)
        #         # print(returns["weight_en_1"].shape)
        #         # print(returns["weight_de_2"].shape)
        #         # sys.exit()
        #         out_f = out_f["x"]
        #         # print(out_f.shape)
        #     else:
        #         out_f = output #self.de_2(output)

        #     if self.config["weight_de_sph_harm"]:
        #         # flip のこと考えてなかった．要修正
        #         SpF_de = SpecialFunc(r=r[:,0], phi=phi[:,0], theta=theta[:,0], maxto=self.config["max_truncation_order"], fft_length=self.fft_length, max_f=self.max_f)
        #         RSH_de = SpF_de.RealSphHarm().to(returns["weight_de_2"].device)
        #         # print(SH_en.shape) # (B,dimz=(N+1)^2) # torch.Size([440, 64])
        #         returns["RSH_de"] = RSH_de
            
        #     if self.config["hlayers_De_-1"]>0:
        #         if self.config["de_2_skip"]:
        #             input_rs_x_m1 = th.tile(out_f.unsqueeze(2),(1,1,B,1)) # (2S,L,B,ch_de or dim_z) or (S,1,B,ch_de or dim_z)
        #         else:
        #             input_rs_x_m1 = out_f.unsqueeze(3) # (2S,L,B,1)
        #         # print(input_rs_x_m1.shape)
        #         if self.config["DNN_for_interp_ITD"]:
        #             input_rs_x_m1 = input_rs_x_m1.reshape(S,B,-1)
        #             input_rs_z_m1 = input_rs_z_out  # (S,1,B,3 or 4)
        #             input_rs_z_m1 = input_rs_z_m1.reshape(S,B,-1)
        #             input_hyper_de_m1 = {
        #                 "x": input_rs_x_m1,  # (S,B, dim_z)
        #                 "z": input_rs_z_m1,  # (S,B, 3or4)
        #             }
        #             # print(input_rs_x_m1.shape)
        #             # print(input_rs_z_m1.shape)
        #             out_f = self.de_m1(input_hyper_de_m1)["x"] # x:(S,B,1)
        #             out_f = out_f.reshape(S, 1, B) # x:(S,1,B)
        #         elif self.config["hyperlinear_de_-1"]:
        #             input_rs_x_m1 = input_rs_x_m1.reshape(2*S,L*B,-1)
        #             input_rs_z_m1 = input_rs_z_out  # (2S,L,B,3 or 4)
        #             input_rs_z_m1 = input_rs_z_m1.reshape(2*S,L*B,-1)
        #             input_hyper_de_m1 = {
        #                 "x": input_rs_x_m1,  # (2S,L*B, dim_z)
        #                 "z": input_rs_z_m1,  # (2S,L*B, 3or4)
        #             }
        #             # print(input_rs_x_m1.shape)
        #             # print(input_rs_z_m1.shape)
        #             out_f = self.de_m1(input_hyper_de_m1)["x"] # x:(2S,L*B,1)
        #             out_f = out_f.reshape(2*S, L, B) # x:(2S,L,B)
        #             # print(out_f.shape)
        #             # sys.exit()
        #         elif self.config["hyperconv_de_-1"]:
        #             # print(f"input_rs_x_m1: {input_rs_x_m1.shape}") # dbg
        #             input_rs_x_m1 = input_rs_x_m1.permute(0,2,3,1).reshape(2*S*B, -1, L) # (2S,L,B,d) -> (2S,B,d,L) -> (2SB,d,L)
        #             input_rs_z_m1 = input_rs_z_out  # (2S,B,3)
        #             input_rs_z_m1 = input_rs_z_m1.reshape(2*S*B,-1) # (2S,B,3) -> (2SB,3)
        #             input_hyper_de_m1 = {
        #                 "x": input_rs_x_m1,  # (2SB,1or2,L)
        #                 "z": input_rs_z_m1,  # (2SB',3)
        #             }
        #             if self.config["use_freq_for_hyper_de"]:
        #                 freq_th = (th.arange(L+1)[1:]/(L/2)-1).to(input_rs_x_0.device)
        #                 freq_th = th.tile(freq_th.view(1,1,L), (2*S*B,1,1)) # (2SB,1,L)
        #                 input_hyper_de_m1["f"] = freq_th  # (2SB,1,L)
                    
        #             out_f = self.de_m1(input_hyper_de_m1)["x"] # x:(2SB,1,L)
        #             out_f = out_f.reshape(2*S, B, L).permute(0,2,1) # x:(2S,L,B)
        #             # print(out_f.shape) # dbg
        #         elif self.config["hyperconv_FD_de_-1"]:
        #             # input_rs_x_m1: (2S,L,B,dim_z) -> (2S,B,dim_z,L)
        #             input_rs_x_m1 = input_rs_x_m1.permute(0,2,3,1)
        #             input_rs_z_m1 = input_rs_z_out  # (2S,L,B,3+ks)
        #             # input_rs_z_m1: (2S,L,B,3+ks) -> (2S,B,L,3+ks)
        #             input_rs_z_m1 = input_rs_z_m1.permute(0,2,1,3)
        #             input_rs_z_m1 = input_rs_z_m1[0,:,:,:] # (B, L, 3+ks)
        #             input_hyper_de_m1 = {
        #                 "x": input_rs_x_m1,  # (2S, B, dim_z, L)
        #                 "z": input_rs_z_m1,  # (B, L, 3+ks)
        #             }
        #             out_f = self.de_m1(input_hyper_de_m1)["x"] # (2S, B, 1, L)
        #             out_f = out_f.permute(0,3,1,2).reshape(2*S, L, B)
        #         else:
        #             raise NotImplementedError
        #     else:
        #         pass
            
        #     # print(out_f.shape)
        #     out_f = out_f.permute(0,2,1)# (2S,B,L) or (S,B,1)
        #     # print(f'max|out_f|:{th.max(th.abs(out_f))}') # tensor(513.5840, device='cuda:0', grad_fn=<MaxBackward1>) # nan_debug
        #     # sys.exit()

        #     if self.config["out_mag_pc"]:
        #         L = V.shape[0]
        #         out_f = out_f.reshape(-1,q) # (2S,B,q) -> (2SB,q)
        #         out_f = th.matmul(out_f, V[:,:q].T) # (2SB,q)@(q,L) = (2SB,L)
        #         out_f = out_f.reshape(2*S,B,L) # (2S,B,L)
        #     elif self.config["out_latent"]:
        #         out_f = self.post_nn(out_f)
        #         L = self.filter_length
        #     elif self.config["out_cnn"]:
        #         out_f = out_f.reshape(2*S*B,1,L) # (2SB,1,Q)
        #         out_f = self.post_cnn(out_f) # (2SB,1,L)
        #         out_f = out_f.reshape(2*S,B,-1) # (2S,B,L)
        #         L = out_f.shape[-1]
        #     elif self.config["out_cnn_ch"]:
        #         # print(out_f.shape) # debug_0805 
        #         out_f = out_f.reshape(2*S*B,cnn_max_ch,-1) # (2SB,ch,L//ch)
        #         # print(out_f.shape) # debug_0805
        #         out_f = self.post_cnn(out_f) # (2SB,1,L)
        #         # print(out_f.shape) # debug_0805
        #         out_f = out_f.reshape(2*S,B,-1) # (2S,B,L)
        #         L = out_f.shape[-1]
            
        #     if self.config["DNN_for_interp_ITD"]:
        #         out = out_f * ITD_std + ITD_mean # (S,B)
        #         returns['output'] = out.reshape(S,B).permute(1,0) # (B,S)
        #     elif self.config["DNN_for_interp_HRIR"]:
        #         out = th.zeros(S,B,2,L, dtype=th.double, device=out_f.device)
        #         out[:,:,0,:] = out_f[0*S:1*S,:,:] * HRIR_std + HRIR_mean
        #         out[:,:,1,:] = out_f[1*S:2*S,:,:] * HRIR_std + HRIR_mean
        #         returns['output'] = out.permute(1,2,3,0) # (B,2,L,S)
        #     else:
        #         # out_f: (2S,B,L) float -> out:(S,B,2,L) complex
        #         out = th.zeros(S,B,2,L, dtype=th.complex64, device=out_f.device)
        #         if self.config["out_mag"] and self.config["use_lr_aug"]:
        #             out[:,:,0,:] = 10**(out_f[0*S:1*S,:,:] * magdb_std + magdb_mean)
        #             out[:,:,1,:] = 10**(out_f[1*S:2*S,:,:] * magdb_std + magdb_mean)
        #         else:
        #             raise NotImplementedError
            
        #         # print(out.shape)

        #         returns['output'] = out.permute(1,2,3,0) # (B,2,L,S)
        #         # print(out.shape)

        # for data_kind in returns:
        #     ic(data_kind)
        #     ic(type(returns[data_kind]))
        
        return returns
