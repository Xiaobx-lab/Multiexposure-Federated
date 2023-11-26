import torch.nn as nn
import torch
from models_x import Generator3DLUT_identity, Generator3DLUT_zero, Classifier, TV_3D, TrilinearInterpolation
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else False

class multi_LUT(nn.Module):
    def __init__(self ):
        super(multi_LUT, self).__init__()
        self.LUT0 = Generator3DLUT_identity().cuda()
        self.LUT1 = Generator3DLUT_zero().cuda()
        self.LUT2 = Generator3DLUT_zero().cuda()
        self.classifier = Classifier().cuda()
#        self.TV3 = TV_3D().cuda()
#        self.TV3.weight_b = self.TV3.weight_b.type(Tensor)
#        self.TV3.weight_g = self.TV3.weight_g.type(Tensor)
#        self.TV3.weight_r = self.TV3.weight_r.type(Tensor)

        # self.trilinear_ = TrilinearInterpolation()  # 插值？
    def forward(self, img):
        pred = self.classifier(img).squeeze()
        if len(pred.shape) == 1:
            pred = pred.unsqueeze(0)
        gen_A0 = self.LUT0(img)
        gen_A1 = self.LUT1(img)
        gen_A2 = self.LUT2(img)
        weights_norm = torch.mean(pred ** 2)
        combine_A = img.new(img.size())
        for b in range(img.size(0)):
            combine_A[b, :, :, :] = pred[b, 0] * gen_A0[b, :, :, :] + pred[b, 1] * gen_A1[b, :, :, :] + pred[b, 2] * gen_A2[b, :, :, :]
#        tv0, mn0 = self.TV3(self.LUT0)
#        tv1, mn1 = self.TV3(self.LUT1)
#        tv2, mn2 = self.TV3(self.LUT2)
#        tv_cons = tv0 + tv1 + tv2
#        mn_cons = mn0 + mn1 + mn2

        return combine_A
#        , weights_norm, tv_cons, mn_cons

if __name__ == "__main__":
    X = torch.Tensor(1,3,400,300).cuda()
    Y = torch.Tensor(1,3,400,300).cuda()
    net = multi_LUT().cuda()
    out = net(X)
    print(out[0].shape)