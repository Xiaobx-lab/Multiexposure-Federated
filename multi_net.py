
import torch.nn as nn
import torchvision.transforms as transforms
import torch
from torch.autograd import Function
from multi_LUT import multi_LUT
# from DCNv2.DCN import PCD_Align
# from unet_model import UNet


def getBinaryTensor(imgTensor, boundary = 200):
    one = torch.ones_like(imgTensor)
    zero = torch.zeros_like(imgTensor)
    return torch.where(imgTensor > boundary, one, zero)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel * 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel * 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class PreBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size ):
        super(PreBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

class ResidualBlockNoBN(nn.Module):

    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class Pyramid(nn.Module):
    def __init__(self, in_channels=6, n_feats=64):
        super(Pyramid, self).__init__()
        self.in_channels = in_channels
        self.n_feats = n_feats
        num_feat_extra = 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.n_feats, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        layers = []
        for _ in range(num_feat_extra):
            layers.append(ResidualBlockNoBN())
        self.feature_extraction = nn.Sequential(*layers)

        self.downsample1 = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x):
        x_in = self.conv1(x)
        x1 = self.feature_extraction(x_in)
        x2 = self.downsample1(x1)
        x3 = self.downsample2(x2)
        return [x1, x2, x3]

class align (nn.Module):
    def __init__(self):
        super(align, self).__init__()
        # PCD align module
        self.pyramid_feats = Pyramid(3)
        # self.align_module = PCD_Align()
        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear'),
        )
    def forward(self, image_short, image_medium, image_long):

        shape = image_short.shape
        image_short = self.model(image_short)
        image_medium = self.model(image_medium)
        image_long = self.model(image_long)
        f1_l = self.pyramid_feats(image_short)
        f2_l = self.pyramid_feats(image_medium)
        f3_l = self.pyramid_feats(image_long)
        f2_ = f2_l[0]   # 基准
        # PCD alignment
        f1_aligned_l = self.align_module(f1_l, f2_l)
        f3_aligned_l = self.align_module(f2_l, f3_l)

        f2_ = nn.Upsample(size=(shape[2],shape[3]),mode='bilinear')(f2_)
        f1_aligned_l = nn.Upsample(size=(shape[2],shape[3]),mode='bilinear')(f1_aligned_l)
        f3_aligned_l = nn.Upsample(size=(shape[2],shape[3]),mode='bilinear')(f3_aligned_l)

        return f2_,f1_aligned_l,f3_aligned_l


class multi_net(nn.Module):
    def __init__(self,conv = default_conv, dim = 16):
        super(multi_net, self).__init__()
        kernel_size = 3
        self.dim = 16
        self.block = PreBlock(default_conv, dim, 3).cuda()
        pre_process = [
            conv(3, self.dim, kernel_size)
        ]
        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]
        post_melt = [
        	  conv(9, 3, kernel_size),
        	  ]
        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)
        self.post_melt = nn.Sequential(*post_melt)
        self.lut = multi_LUT().cuda()
        self.sig = nn.Sigmoid().cuda()
        # self.conv1 = nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=True)
        # self.align = align().cuda()
        # self.unet = UNet(3,3)

    def forward(self, image_short, image_medium, image_long):

        # f2_, f1_aligned_l, f3_aligned_l = self.align( image_short, image_medium, image_long)

        image_short = self.lut(image_short)
        image_medium = self.lut(image_medium)
        image_long = self.lut(image_long)

        # melt_img_align = self.conv1(f2_+f1_aligned_l+f3_aligned_l)
        # melt_img = self.post_melt(melt_img)

        melt_img = image_short+image_medium+image_long
        # print(melt_img.shape)
       # melt_img = self.unet(melt_img)
        melt_img = self.pre(melt_img)
        melt_img = self.block(melt_img)
        # melt_img = self.block(melt_img)
        melt_img = self.block(melt_img)
        melt_img = self.post(melt_img)
       # melt_img = self.lut(melt_img)
        out = self.sig(melt_img)
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.Tensor(1,3,256,256).to(device)
    Y = torch.Tensor(1,3,256,256).to(device)
    Z = torch.Tensor(1, 3, 256, 256).to(device)
    net = multi_net().cuda()
    out = net(X,Y,Z)
    print(out.shape)


