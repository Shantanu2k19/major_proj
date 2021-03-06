import torch
import torch.nn as nn

class Encoder2(nn.Module):
    def __init__(self):
        super().__init__()

        self.inorm = InstanceNorm()

        self.l11 = nn.Conv2d(1, 64, 3, 1, 1)
        self.l111 = nn.BatchNorm2d(64)
        self.l12 = nn.ReLU(inplace=True)
        self.l13 = nn.MaxPool2d(2, 2)

        self.l21 = nn.Conv2d(64, 128, 3, 1, 1)
        self.l222 = nn.BatchNorm2d(128)
        self.l22 = nn.ReLU(inplace=True)
        self.l23 = nn.MaxPool2d(2, 2)

        self.l31 = nn.Conv2d(128, 256, 3, 1, 1)
        self.l333 = nn.BatchNorm2d(256)
        self.l32 = nn.ReLU(inplace=True)

        self.l41 = nn.Conv2d(256, 256, 3, 1, 1)
        self.l444 = nn.BatchNorm2d(256)
        self.l42 = nn.ReLU(inplace=True)
        self.l43 = nn.MaxPool2d(2, 2)

        self.l51 = nn.Conv2d(256, 512, 3, 1, 1)
        self.l555 = nn.BatchNorm2d(512)
        self.l52 = nn.ReLU(inplace=True)

        self.l61 = nn.Conv2d(512, 512, 3, 1, 1)
        self.l666 = nn.BatchNorm2d(512)
        self.l62 = nn.ReLU(inplace=True)
        self.l63 = nn.MaxPool2d(2, 2)
        # self.embeddings = nn.Sequential(
        #     nn.Linear(8, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 128),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        mns = []
        sds = []
        def calc(y):

            y, mn, sd = self.inorm(y, return_mean_std=True)
            mns.append(mn)
            sds.append(sd)

        y = x
        y = self.l11(y)
        y = self.l111(y)
        y = self.l12(y)
        y = self.l13(y)

        y = self.l21(y)
        y = self.l222(y)
        y = self.l22(y)
        y = self.l23(y)

        calc(y)

        y = self.l31(y)
        y = self.l333(y)
        y = self.l32(y)

        calc(y)

        y = self.l41(y)
        y = self.l444(y)
        y = self.l42(y)
        y = self.l43(y)

        calc(y)

        y = self.l51(y)
        y = self.l555(y)
        y = self.l52(y)

        calc(y)

        y = self.l61(y)
        y = self.l66(y)
        y = self.l62(y)
        y = self.l63(y)

        calc(y)


        
        return y, mns, sds


class InstanceNorm(nn.Module):
    def _init_(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def calc_mean_std(self, x, mask=None):
        B, C = x.shape[:2]

        mn = x.view(B, C, -1).mean(-1)
        sd = (x.view(B, C, -1).var(-1) + 1e-5).sqrt()
        mn = mn.view(B, C, *((len(x.shape) - 2) * [1]))
        sd = sd.view(B, C, *((len(x.shape) - 2) * [1]))
        
        return mn, sd


    def forward(self, x, return_mean_std=False):
        mean, std = self.calc_mean_std(x)
        x = (x - mean) / std
        if return_mean_std:
            return x, mean, std
        else:
            return x

class Decoder2(nn.Module):
    def __init__(
        self, c_in=0, c_h=0, c_out=0, 
        n_conv_blocks=6, upsample=1
    ):
        super().__init__()
        # self.embeddings = nn.Sequential(
        #     nn.Linear(128,1024),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(1024, 1024),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(1024,8),
        #     nn.ReLU(inplace=True),
        # )
        self.l00 = nn.BatchNorm2d(512)
        self.l11 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.l111 = nn.BatchNorm2d(512)
        self.l12 = nn.ReLU(inplace=True)
        self.l21 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.l22 = nn.ReLU(inplace=True)
        self.l31 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.l333 = nn.BatchNorm2d(512)
        self.l32 = nn.ReLU(inplace=True)
        self.l41 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.l42 = nn.ReLU(inplace=True)
        self.l51 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.l52 = nn.ReLU(inplace=True)
        self.l6 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.rnn = nn.GRU(128,128,num_layers = 2)
    def forward(self,y,mns,sds):
        y = self.l00(y)
        y = y*sds[-1] + mns[-1]
        y = self.l11(y)
        y = self.l111(y)
        y = self.l12(y)
        y = y*sds[-2] + mns[-2]

        y = self.l21(y)
        y = self.l22(y)
        
        y = y*sds[-3] + mns[-3]
        y = self.l31(y)
        y = self.l32(y)
        y = y*sds[-4] + mns[-4]

        y = self.l41(y)
        y = self.l42(y)
        y = y*sds[-5] + mns[-5]

        y = self.l51(y)
        y = self.l52(y)

        y = self.l6(y)

        y = y.squeeze(1)
        y,_ = self.rnn(y)

        return y;
class VariantSigmoid(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        y = 1 / (1+torch.exp(-self.alpha*x))
        return y
class model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder2()
        self.dec = Decoder2()
        self.act = VariantSigmoid(0.1)
    def forward(self,x):
        y,mns,sds = self.enc(x)
        y = self.act(y)
        y = self.dec(y,mns,sds)
        return y;
kk = torch.rand(32,1,80,128)
obj = model1()
obj.forward(kk)