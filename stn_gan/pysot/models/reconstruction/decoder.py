from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from IPython import embed
import torch.nn.functional as F
# class Bottleneck(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1,
#                  downsample=None, dilation=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         padding = 2 - stride
#         if downsample is not None and dilation > 1:
#             dilation = dilation // 2
#             padding = dilation

#         assert stride == 1 or dilation == 1, \
#             "stride and dilation must have one equals to zero at least"

#         if dilation > 1:
#             padding = dilation
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=padding, bias=False, dilation=dilation)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual

#         out = self.relu(out)

#         return out

# class Decoder(nn.Module):
#     def __init__(self,in_channel,out_channel,kernel_size=3):
#         super(Decoder,self).__init__()
#         self.reconstruct = nn.Sequential(
            
#             nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(in_channel),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(in_channel, in_channel//2, kernel_size=kernel_size, stride=2, bias=False),
#             nn.BatchNorm2d(in_channel//2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channel//2, in_channel//4, kernel_size=kernel_size, stride=1, bias=False),
#             nn.BatchNorm2d(in_channel//4),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(in_channel // 4, in_channel // 8, kernel_size=kernel_size, stride=2, bias=False),
#             nn.BatchNorm2d(in_channel // 8),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channel // 8, in_channel // 16, kernel_size=kernel_size, stride=1, bias=False),
#             nn.BatchNorm2d(in_channel // 16),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(in_channel // 16, in_channel // 32, kernel_size=kernel_size, stride=2, bias=False),
#             nn.BatchNorm2d(in_channel // 32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channel // 32, in_channel//64, kernel_size=kernel_size, stride=1, bias=False),
#             nn.BatchNorm2d(in_channel//64),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(in_channel//64, out_channel, kernel_size=kernel_size, stride=2, bias=False),
#             nn.Conv2d(out_channel, out_channel, kernel_size=5, stride=1, bias=False),
#             nn.Sigmoid()
#         )
#     def forward(self, input):
#         # noise = torch.randn(input.shape[0],input.shape[1],input.shape[2],input.shape[3]).cuda()
        
#         # input = torch.cat([input,noise],dim=1)
#         output = self.reconstruct(input)
#         return output
class Decoder(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=4):
        super(Decoder,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel*2, in_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channel, in_channel, kernel_size=kernel_size, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channel*2,in_channel//2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channel//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channel//2, in_channel//4, kernel_size=kernel_size, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(in_channel//4),
            nn.ReLU(inplace=True),
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channel//2, in_channel // 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channel // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channel // 2, in_channel // 4, kernel_size=kernel_size, stride=2, padding=1,bias=False),
        )
        self.layer3=nn.Sequential(
            # nn.BatchNorm2d(in_channel // 32),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channel // 32, in_channel//64, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(in_channel//64),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channel//64, out_channel, kernel_size=kernel_size, stride=2, bias=False),
            nn.Conv2d(in_channel//4, out_channel, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input, thetas, all_pos, all_features):
        
        noise = torch.randn(input.shape[0],input.shape[1],input.shape[2],input.shape[3]).cuda()
        
        input = torch.cat([input,noise],dim=1)
        output = self.conv1(input)
        output1=output
        for i, theta in enumerate(thetas):
            
            grid=F.affine_grid(theta.unsqueeze(0), all_features[1][i,:,:,:].unsqueeze(0).size())
            replace=F.grid_sample(all_features[1][i,:,:,:].unsqueeze(0),grid)
            if i==0:
                output1=torch.cat([output[i,:,:,:].unsqueeze(0),replace],dim=1)
            else:
                output1=torch.cat([output1,torch.cat([output[i,:,:,:].unsqueeze(0),replace],dim=1)],dim=0)
        
        output1=self.layer1(output1)
        for i, theta in enumerate(thetas):
            grid=F.affine_grid(theta.unsqueeze(0), all_features[0][i,:,:,:].unsqueeze(0).size())
            replace=F.grid_sample(all_features[0][i,:,:,:].unsqueeze(0),grid)
            if i==0:
                output2=torch.cat([output1[i,:,:,:].unsqueeze(0),replace],dim=1)
            else:
                output2=torch.cat([output2,torch.cat([output1[i,:,:,:].unsqueeze(0),replace],dim=1)],dim=0)
        output2=self.layer2(output2)
        # for i, theta in enumerate(thetas):
        #     grid=F.affine_grid(theta.unsqueeze(0), all_features[0][i,:,:,:].unsqueeze(0).size())
        #     replace=F.grid_sample(all_features[0][i,:,:,:].unsqueeze(0),grid)
        #     output2[i,:,all_pos[i][2]:all_pos[i][2]+128,all_pos[i][3]:all_pos[i][3]+128]=replace
        output3=self.layer3(output2)
        return output3

# class Decoder(nn.Module):
#     def __init__(self,in_channel,out_channel):
#         super(Decoder, self).__init__()
#         self.downsample = self._make_block(nn.Conv2d(in_channel, 256, 1, bias=False), 256)
#         self.inplanes = 256
#         self.layer1 = self._make_layer(Bottleneck, 256, 3)
#         self.deconv1 = self._make_block(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
#                                                            bias=False), 128)
#         self.inplanes = 128
#         self.layer2 = self._make_layer(Bottleneck, 32, 2)
#         self.deconv2 = self._make_block(nn.ConvTranspose2d(32, 32, kernel_size=3,
#                                                            stride=2, padding=1, bias=False), 32)
#         self.deconv3 = nn.ConvTranspose2d(32, out_channel, kernel_size=7, stride=2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         x = self.downsample(x) #  256*15*15
#         x = self.layer1(x) # 256*15*15
#         x = self.deconv1(x) # 128*31*31
#         x = self.layer2(x) # 32*31*31
#         x = self.deconv2(x) # 32*61*61
#         x = self.deconv3(x) # 3*127*127
#         x = self.sigmoid(x)
#         return x

#     def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
#         downsample = None
#         dd = dilation
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             if stride == 1 and dilation == 1:
#                 downsample = nn.Sequential(
#                     nn.Conv2d(self.inplanes, planes * block.expansion,
#                               kernel_size=1, stride=stride, bias=False),
#                     nn.BatchNorm2d(planes * block.expansion),
#                 )
#             else:
#                 if dilation > 1:
#                     dd = dilation // 2
#                     padding = dd
#                 else:
#                     dd = 1
#                     padding = 0
#                 downsample = nn.Sequential(
#                     nn.Conv2d(self.inplanes, planes * block.expansion,
#                               kernel_size=3, stride=stride, bias=False,
#                               padding=padding, dilation=dd),
#                     nn.BatchNorm2d(planes * block.expansion),
#                 )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride,
#                             downsample, dilation=dilation))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, dilation=dilation))

#         return nn.Sequential(*layers)

#     def _make_block(self, module, out_channel):
#         return nn.Sequential(
#             module,
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True),
#         )


# if __name__ == "__main__":
#     r_zf = [torch.rand(1, 256, 15, 15)] * 3
#     fg_zf = [zf[:, :zf.size(1) // 2, :, :].contiguous() for zf in r_zf]
#     bg_zf = [zf[:, zf.size(1) // 2:, :, :].contiguous() for zf in r_zf]
#     fg_zf, bg_zf = torch.cat(fg_zf, dim=1), torch.cat(bg_zf, dim=1)
#     print(fg_zf.shape)
#     print(bg_zf.shape)
#     feature = torch.rand(1, 128 * 3, 15, 15)
#     print(feature.shape)
#     net = Decoder()
#     print(net(feature).shape)
