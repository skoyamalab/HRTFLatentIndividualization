import torch as th
import torchaudio as ta

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

        # B, 1, L, S
        ILD_AE = th.abs(th.diff(data, dim=dim) - th.diff(target, dim=dim))
        if mean:
            ILD_AE = th.mean(ILD_AE)
        return ILD_AE

class LSD(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data, target, dim=2, data_kind="HRTF", mean=True):
        """
        :param data:   (B, 2, L, S) complex (or float) tensor
        :param target: (B, 2, L, S) complex (or float) tensor
        :return: a scalar or (B, 2, S) tensor
        """
        if data_kind == "HRTF":
            mag2db = ta.transforms.AmplitudeToDB(stype="magnitude")
            data = mag2db(th.abs(data))
            target = mag2db(th.abs(target))
        elif data_kind == "HRTF_mag":
            pass

        LSD = th.sqrt(th.mean((data - target).pow(2), dim=dim))
        if mean:
            LSD = th.mean(LSD)

        return LSD
