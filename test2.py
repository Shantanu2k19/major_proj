# import torch
from torch.nn import ConvTranspose2d,GRU
# in,out,kernel,stride,padding,opp
from torch import randn
# Hout = (Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
"""
torch.Size([32, 64, 40, 64]) 1
torch.Size([32, 128, 20, 32]) 2
torch.Size([32, 256, 20, 32]) 3
torch.Size([32, 256, 10, 16]) 4
torch.Size([32, 512, 10, 16]) 5
torch.Size([32, 512, 5, 8]) 6
torch.Size([32, 512, 5, 128]) after embeddings


src = torch.Size([32, 256, 20, 32]) after embeddings
trgt = torch.Size([32, 64, 40, 64])
"""
def get(hin,stride,padding,dilation,kernel,opp):
	return ( ((hin - 1)*stride) - 2*padding) + dilation*(kernel-1) + opp 
h = randn(32, 64, 40, 64)
upsample = ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1,dilation=1)
obj = GRU(128,128,num_layers = 2)
_,cc,H,W = h.shape
h = upsample(h)
h = h.squeeze(1)
print(h.shape)
h_0 = randn(2,80,128)
h = obj(h)
# print(h.shape)
	# ,get(H,2,1,1,4,1),get(W,2,1,1,4,1))