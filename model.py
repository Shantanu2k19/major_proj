import torch
import torch.nn as nn
import torch.nn.functional as F
from util.mytorch import np2pt

def get_model(build_config, device, mode):
    model = Model(**build_config.model.params).to(device)
    # model = model1().to(device)
    # print(model)
    if mode == 'train':
        # model_state, step_fn, save, load
        optimizer = torch.optim.Adam(model.parameters(), **build_config.optimizer.params)
        criterion_l1 = nn.L1Loss()
        model_state = {
            'model': model,
            'optimizer': optimizer,
            'steps': 0,
            # static, no need to be saved
            'criterion_l1': criterion_l1,
            'device': device,
            'grad_norm': build_config.optimizer.grad_norm,
            # this list restores the dynamic states
            '_dynamic_state': [
                'model',
                'optimizer',
                'steps'
            ]
        }
        return model_state, train_step
    elif mode == 'inference':
        model_state = {
            'model': model,
            # static, no need to be saved
            'device': device,
            '_dynamic_state': [
                'model'
            ]
        }
        return model_state, inference_step
    else:
        raise NotImplementedError

# For training and evaluating 
def train_step(model_state, data, train=True):
    meta = {}
    model = model_state['model']
    optimizer = model_state['optimizer']
    criterion_l1 = model_state['criterion_l1']
    device = model_state['device']
    grad_norm = model_state['grad_norm']

    if train:
        optimizer.zero_grad()
        model.train()
    else:
        model.eval()

    x = data['mel'].to(device)

    dec = model(x)
    loss_rec = criterion_l1(dec, x)

    loss = loss_rec

    if train:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
            max_norm=grad_norm)
        optimizer.step()

    meta['log'] = {
        'loss_rec': loss_rec.item(),
    }

    return meta

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# For inference
def inference_step(model_state, data):
    meta = {}
    model = model_state['model']
    device = model_state['device']
    model.to(device)
    model.eval()

    source = data['source']['mel']
    target = data['target']['mel']

    source = np2pt(source).to(device)
    target = np2pt(target).to(device)
    
    dec = model.inference(source, target)
    meta = {
        'dec': dec
    }
    return meta


# ====================================
#  Modules
# ====================================
class InstanceNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def calc_mean_std(self, x, mask=None):
        B, C = x.shape[:2]

        mn = x.view(B, C, -1).mean(-1)
        sd = (x.view(B, C, -1).var(-1) + self.eps).sqrt()
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




class DecConvBlock(nn.Module):
    def __init__(self, c_in, c_h, c_out, upsample=1):
        super().__init__()
        self.dec_block = nn.Sequential(
                ConvNorm(c_in, c_h, kernel_size=3, stride=1),
                nn.BatchNorm1d(c_h),
                nn.LeakyReLU(),
                ConvNorm(c_h, c_in, kernel_size=3),
                )
        self.gen_block = nn.Sequential(
                ConvNorm(c_in, c_h, kernel_size=3, stride=1),
                nn.BatchNorm1d(c_h),
                nn.LeakyReLU(),
                ConvNorm(c_h, c_in, kernel_size=3),
                )
        self.upsample = upsample
    def forward(self, x):
        y = self.dec_block(x)
        if self.upsample >  1:
            x = F.interpolate(x, scale_factor=self.upsample)
            y = F.interpolate(y, scale_factor=self.upsample)
        y = y + self.gen_block(y)
        return x + y


# ====================================
#  Model
# ====================================

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
  

    def forward(self, x):
        mns = []
        sds = []
        # ll = [2]
        def calc(y):

            y, mn, sd = self.inorm(y, return_mean_std=True)
            mns.append(mn)
            sds.append(sd)
            # print(y.shape,ll[0])
            # ll[0]+=1

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
        y = self.l666(y)
        y = self.l62(y)
        y = self.l63(y)

        calc(y)

        # y = self.embeddings(y)

        # print(y.shape,'after embeddings')
        
        return y, mns, sds


class Decoder2(nn.Module):
    def __init__(
        self, c_in=0, c_h=0, c_out=0, 
        n_conv_blocks=6, upsample=1
    ):
        super().__init__()
        self.l00 = nn.BatchNorm2d(512);
        self.l11 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.l111 = nn.BatchNorm2d(512)
        self.l12 = nn.LeakyReLU(inplace=True)
        self.l21 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.l22 = nn.LeakyReLU(inplace=True)
        self.l31 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.l333 = nn.BatchNorm2d(256)
        self.l32 = nn.LeakyReLU(inplace=True)
        self.l41 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.l42 = nn.LeakyReLU(inplace=True)
        self.l51 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.l555 = nn.BatchNorm2d(64)
        self.l52 = nn.LeakyReLU(inplace=True)
        self.l6 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.rnn = nn.GRU(128,128,num_layers = 2)
    def forward(self,y,mns,sds):
        # print("Now Decoding")
        # y = self.embeddings(y)
        # print(y.shape,"after embeddings")
        y = y*sds[-1] + mns[-1]
        y = self.l11(y)
        y = self.l111(y)
        y = self.l12(y)
        y = y*sds[-2] + mns[-2]
        # print(y.shape)
        y = self.l21(y)
        y = self.l22(y)
        
        y = y*sds[-3] + mns[-3]
        y = self.l31(y)
        y = self.l333(y)
        y = self.l32(y)
        y = y*sds[-4] + mns[-4]
        # print(y.shape)
        y = self.l41(y)
        y = self.l42(y)
        y = y*sds[-5] + mns[-5]
        # print(y.shape)
        y = self.l51(y)
        y = self.l555(y)
        y = self.l52(y)
        
        y = self.l6(y)
        y = y.squeeze(1)
        # print(y.shape)
        y,_ = self.rnn(y)
        return y;



class VariantSigmoid(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        y = 1 / (1+torch.exp(-self.alpha*x))
        return y

class NoneAct(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


class Activation(nn.Module):
    dct = {
        'none': NoneAct,
        'sigmoid': VariantSigmoid,
        'tanh': nn.Tanh
    }
    def __init__(self, act, params=None):
        super().__init__()
        self.act = Activation.dct[act](**params)

    def forward(self, x):
        return self.act(x)


class model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder2()
        self.dec = Decoder2()
        self.act = VariantSigmoid(0.1)
    def forward(self,x):
        x = x[:,None,:,:]
        y,mns,sds = self.enc(x)
        y = self.act(y)
        y = self.dec(y,mns,sds)
        return y;
    def inference(self,source,target):
        original_source_len = source.size(-1)
        original_target_len = target.size(-1)

        if original_source_len % 8 != 0:
            source = F.pad(source, (0, 8 - original_source_len % 8), mode='reflect')
        if original_target_len % 8 != 0:
            target = F.pad(target, (0, 8 - original_target_len % 8), mode='reflect')

        x, x_cond = source, target
        x,x_cond = x[:,None,:,:],x_cond[:,None,:,:]
        y,_,_,=self.enc(x)
        y1,mns,sds = self.enc(x)
        y = self.act(y);
        y = self.dec(y,mns,sds)
        dec = y[:,:,:original_source_len]
        return dec;