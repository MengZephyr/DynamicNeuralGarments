import torch.nn as nn
import torch
from siren import siren_uniform, Sine
from torch.nn.utils import spectral_norm
import torch.nn.functional as F


class LaplacianPyramid(nn.Module):
    def __init__(self, device):
        super(LaplacianPyramid, self).__init__()
        self.device = device

    def forward(self, layers, pixelMask):
        out = []
        pixelMask = pixelMask.permute(1, 2, 0)
        pixelMask = pixelMask.unsqueeze(0)
        pixelMask = 2. * pixelMask - 1.
        for lgrid in layers:
            y = F.grid_sample(lgrid, pixelMask)
            out.append(y)
        out = torch.cat(out, dim=1)
        return out


class NeuralFeatMap(nn.Module):
    def __init__(self, device):
        super(NeuralFeatMap, self).__init__()
        self.device = device

    def forward(self, H, W, pX, pY, vertMask, hatF, jInfo=None):
        if jInfo is not None:
            vInfo = torch.cat([hatF, jInfo], dim=0)  # jInfo: dimFeat, numVerts
            out = torch.zeros(vInfo.size()[0], H, W).to(self.device)
            out[:, pY, pX] = torch.mm(vInfo, vertMask)
            return out
        else:
            out = torch.zeros(hatF.size()[0], H, W).to(self.device)
            #out[:, pY, pX] = 1.
            out[:, pY, pX] = torch.mm(hatF, vertMask)  # [dimFeat, numVerts] * [numVerts, len(pX)] = [dimFeat, len(pX)]
            return out


class sinConv(nn.Module):
    def __init__(self, inDim, outDim, kernel_size, stride, padding, w0, c, ifActSin=False):
        super(sinConv, self).__init__()
        if ifActSin is True:
            layers = [nn.Conv2d(inDim, outDim, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                      Sine(w0=w0)]
        else:
            layers = [nn.Conv2d(inDim, outDim, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                      nn.LeakyReLU(0.2, True)]
        self.en = nn.Sequential(*layers)
        #self.weight_ini(c)

    def weight_ini(self, c):
        for m in self.en.state_dict():
            k = m.split('.')
            if k[-1] == 'weight':
                siren_uniform(self.en.state_dict()[m], mode='fan_in', c=c)

    def forward(self, x):
        return self.en(x)


class sinConvTranspose(nn.Module):
    def __init__(self, inDim, outDim, kernel_size, stride, padding, w0, c, ifActSin=False):
        super(sinConvTranspose, self).__init__()
        if ifActSin is True:
            layers = [nn.ConvTranspose2d(inDim, outDim,
                                         kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                      Sine(w0)]
        else:
            layers = [nn.ConvTranspose2d(inDim, outDim,
                                         kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                      nn.LeakyReLU(0.2, True)]
        self.de = nn.Sequential(*layers)
        #self.weight_ini(c)

    def weight_ini(self, c):
        for m in self.de.state_dict():
            k = m.split('.')
            if k[-1] == 'weight':
                siren_uniform(self.de.state_dict()[m], mode='fan_in', c=c)

    def forward(self, x):
        return self.de(x)


class OneInputEncoder(nn.Module):
    def __init__(self, featureDim, outDim, w0=1, w0_ini=30, c=6., ifActSin=False):
        super(OneInputEncoder, self).__init__()
        self.encoder = nn.Sequential(
            sinConv(featureDim, 64, kernel_size=4, stride=2, padding=1, w0=w0_ini, c=c, ifActSin=ifActSin),
            sinConv(64, 64, kernel_size=4, stride=2, padding=1, w0=w0_ini, c=c, ifActSin=ifActSin),
            sinConv(64, 128, kernel_size=4, stride=2, padding=1, w0=w0, c=c, ifActSin=ifActSin),
            sinConv(128, 256, kernel_size=4, stride=2, padding=1, w0=w0, c=c, ifActSin=ifActSin),
            sinConv(256, 512, kernel_size=4, stride=2, padding=1, w0=w0, c=c, ifActSin=ifActSin),
            sinConv(512, outDim, kernel_size=3, stride=1, padding=1, w0=w0, c=c, ifActSin=ifActSin)
        )

    def forward(self, X):
        return self.encoder(X)


class MultiLevelEncoder(nn.Module):
    def __init__(self, featureDim, outDim, numLevels, w0, w0_ini, c):
        super(MultiLevelEncoder, self).__init__()
        self.numLevels = numLevels
        self.E0 = sinConv(featureDim, 64, kernel_size=3, stride=2, padding=1, w0=w0_ini, c=c)

        self.F1 = sinConv(featureDim, 64, kernel_size=3, stride=1, padding=1, w0=w0, c=c)
        self.E1 = sinConv(128, 64, kernel_size=3, stride=2, padding=1, w0=w0, c=c)

        self.F2 = sinConv(featureDim, 64, kernel_size=3, stride=1, padding=1, w0=w0, c=c)
        self.E2 = sinConv(128, 128, kernel_size=3, stride=2, padding=1, w0=w0, c=c)

        self.F3 = sinConv(featureDim, 128, kernel_size=3, stride=1, padding=1, w0=w0, c=c)
        self.E3 = sinConv(256, 256, kernel_size=3, stride=2, padding=1, w0=w0, c=c)

        self.F4 = sinConv(featureDim, 256, kernel_size=3, stride=1, padding=1, w0=w0, c=c)
        self.E4 = sinConv(512, 512, kernel_size=3, stride=2, padding=1, w0=w0, c=c)

        self.EM = sinConv(512, outDim, kernel_size=3, stride=1, padding=1, w0=w0, c=c)

    def forward(self, BatchLevelMaps):
        assert (len(BatchLevelMaps) == self.numLevels)
        x = self.E0(BatchLevelMaps[0])

        f = self.F1(BatchLevelMaps[1])
        x = self.E1(torch.cat([x, f], dim=1))

        f = self.F2(BatchLevelMaps[2])
        x = self.E2(torch.cat([x, f], dim=1))

        f = self.F3(BatchLevelMaps[3])
        x = self.E3(torch.cat([x, f], dim=1))

        f = self.F4(BatchLevelMaps[4])
        x = self.E4(torch.cat([x, f], dim=1))

        x = self.EM(x)
        return x


class SPADE(nn.Module):
    def __init__(self, xDim, gDim, w0=1, c=6, ifInsNorm=True):
        super(SPADE, self).__init__()
        self.ifInsNorm = ifInsNorm
        if ifInsNorm is True:
            self.inNorm = nn.InstanceNorm2d(xDim, affine=False)
        nhidden = xDim
        ifActSin = False if ifInsNorm is True else True
        self.share_net = sinConv(gDim, nhidden, kernel_size=3, stride=1, padding=1, w0=w0, c=c, ifActSin=ifActSin)
        self.gamma_net = sinConv(nhidden, xDim, kernel_size=3, stride=1, padding=1, w0=w0, c=c, ifActSin=ifActSin)
        self.beta_net = sinConv(nhidden, xDim, kernel_size=3, stride=1, padding=1, w0=w0, c=c, ifActSin=ifActSin)

    def forward(self, x, g):
        if self.ifInsNorm is True:
            nx = self.inNorm(x)
        else:
            nx = x
        actv = self.share_net(g)
        gamma = self.gamma_net(actv)
        beta = self.beta_net(actv)
        out = nx * gamma + beta
        return out


class BetaSPADE(nn.Module):
    def __init__(self, xDim, gDim, w0, c):
        super(BetaSPADE, self).__init__()
        self.guidNet = nn.Sequential(sinConv(gDim, xDim, kernel_size=3, stride=1, padding=1, w0=w0, c=c),
                                     sinConv(xDim, xDim, kernel_size=3, stride=1, padding=1, w0=w0, c=c))

    def forward(self, x, g):
        beta = self.guidNet(g)
        return x+beta


class SPADEResnetBlock(nn.Module):
    def __init__(self, indim, gdim, w0=1., w0_ini=30., c=6., ifnorm=False, skip=True, ifActSin=False):
        super(SPADEResnetBlock, self).__init__()
        self.skip = skip
        self.spade1 = SPADE(xDim=indim, gDim=gdim, w0=w0, c=c, ifInsNorm=ifnorm)
        self.conv1 = sinConv(indim, indim, kernel_size=3, stride=1, padding=1, w0=w0_ini, c=c, ifActSin=ifActSin)

        self.spade2 = SPADE(xDim=indim, gDim=gdim, w0=w0, c=c, ifInsNorm=ifnorm)
        self.conv2 = sinConv(indim, indim, kernel_size=3, stride=1, padding=1, w0=w0_ini, c=c, ifActSin=ifActSin)

        if self.skip is True:
            self.skip_spade = SPADE(xDim=indim, gDim=gdim, w0=w0, c=c, ifInsNorm=ifnorm)
            self.skip_conv = sinConv(indim, indim, kernel_size=3, stride=1, padding=1, w0=w0_ini, c=c, ifActSin=ifActSin)

    def forward(self, x, g):
        xf = x
        xf = self.spade1(xf, g)
        xf = self.conv1(xf)
        xf = self.spade2(xf, g)
        xf = self.conv2(xf)

        if self.skip is True:
            x = self.skip_spade(x, g)
            x = self.skip_conv(x)

        return xf + x


class CNNGenerator(nn.Module):
    def __init__(self, midDim, bgInDim, outDim, w0=1., w0_ini=30., c=6., ifSine=True):
        super(CNNGenerator, self).__init__()
        self.D_net = nn.Sequential(sinConvTranspose(midDim, 512, kernel_size=4, stride=2, padding=1, w0=w0, c=c, ifActSin=ifSine),
                                   sinConvTranspose(512, 256, kernel_size=4, stride=2, padding=1, w0=w0, c=c, ifActSin=ifSine),
                                   sinConvTranspose(256, 128, kernel_size=4, stride=2, padding=1, w0=w0, c=c, ifActSin=ifSine),
                                   sinConvTranspose(128, 64, kernel_size=4, stride=2, padding=1, w0=w0, c=c, ifActSin=ifSine),
                                   sinConvTranspose(64, 64, kernel_size=4, stride=2, padding=1, w0=w0, c=c, ifActSin=ifSine))
        # self.D5 = sinConvTranspose(midDim, 512, kernel_size=4, stride=2, padding=1, w0=w0, c=c)
        # self.D4 = sinConvTranspose(512, 256, kernel_size=4, stride=2, padding=1, w0=w0, c=c)
        # self.D3 = sinConvTranspose(256, 128, kernel_size=4, stride=2, padding=1, w0=w0, c=c)
        # self.D2 = sinConvTranspose(128, 64, kernel_size=4, stride=2, padding=1, w0=w0, c=c)
        # self.D1 = sinConvTranspose(64, 64, kernel_size=4, stride=2, padding=1, w0=w0, c=c)

        bgDim = 32
        #self.FConv = sinConv(64, bgDim, kernel_size=3, stride=1, padding=1, w0=w0, c=c)
        self.FConv = nn.Conv2d(64, bgDim, kernel_size=3, stride=1, padding=1)
        #self.weight_ini(c, self.FConv)

        self.MConv = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1), nn.Sigmoid())

        #self.BConv = sinConv(bgInDim, bgDim, kernel_size=3, stride=1, padding=1, w0=w0_ini, c=c)
        self.BConv = nn.Conv2d(bgInDim, bgDim, kernel_size=3, stride=1, padding=1)
        #self.weight_ini(c, self.BConv)
        self.BFSin = Sine(w0=w0) if ifSine else nn.LeakyReLU(0.2, True)

        self.ResB_0 = nn.Sequential(sinConv(bgDim, bgDim, kernel_size=3, stride=1, padding=1, w0=w0, c=c, ifActSin=ifSine),
                                    sinConv(bgDim, bgDim, kernel_size=3, stride=1, padding=1, w0=w0, c=c, ifActSin=ifSine))
        self.outConv_0 = sinConv(bgDim, bgDim, kernel_size=3, stride=1, padding=1, w0=w0, c=c, ifActSin=ifSine)

        self.ResB_1 = nn.Sequential(sinConv(bgDim, bgDim, kernel_size=3, stride=1, padding=1, w0=w0, c=c, ifActSin=ifSine),
                                    sinConv(bgDim, bgDim, kernel_size=3, stride=1, padding=1, w0=w0, c=c, ifActSin=ifSine))

        self.outConv_1 = nn.Conv2d(bgDim, outDim, kernel_size=3, stride=1, padding=1)

    def weight_ini(self, c, net):
        for m in net.state_dict():
            k = m.split('.')
            if k[-1] == 'weight':
                siren_uniform(net.state_dict()[m], mode='fan_in', c=c)

    def forward(self, x, bg, ifNeedM=False):
        x = self.D_net(x)
        f = self.FConv(x)
        m = self.MConv(x)
        b = self.BConv(bg)
        x = m * f + (1-m) * b
        x = self.BFSin(x)

        x = x + self.ResB_0(x)
        x = self.outConv_0(x)
        x = x + self.ResB_1(x)
        x = self.outConv_1(x)

        if ifNeedM:
            return x, m
        else:
            return x


class oneInRenderGenerator(nn.Module):
    def __init__(self, inDim, midDim, outDim, bgInDim, device, ifActSin=False):
        super(oneInRenderGenerator, self).__init__()
        self.Encoder = OneInputEncoder(featureDim=inDim, outDim=midDim, ifActSin=ifActSin).to(device)
        ifnorm = False if ifActSin is True else True
        self.LatentSpade = SPADEResnetBlock(indim=midDim, gdim=midDim*2, ifnorm=ifnorm, ifActSin=ifActSin).to(device)
        self.Decoder = CNNGenerator(midDim=midDim, bgInDim=bgInDim, outDim=outDim, ifSine=ifActSin).to(device)
        if ifActSin:
            self.weight_ini(6, self.Encoder)
            self.weight_ini(6, self.LatentSpade)
            self.weight_ini(6, self.Decoder)

    def weight_ini(self, c, net):
        for m in net.state_dict():
            k = m.split('.')
            if k[-1] == 'weight':
                siren_uniform(net.state_dict()[m], mode='fan_in', c=c)

    def forward(self, xLMap, pLMap, bg, ifmask=False):
        x = self.Encoder(xLMap)
        sg = self.Encoder(pLMap)
        sg = torch.cat([x, sg], dim=1)
        x = self.LatentSpade(x, sg)

        if ifmask:
            x, m = self.Decoder(x, bg, ifmask)
            return x, m
        else:
            x = self.Decoder(x, bg)
            return x

    def runBlend(self, xLMap, pLMap, nxLMap, npLMap, bg, alpha=0.3):
        x = self.Encoder(xLMap)
        sg = self.Encoder(pLMap)
        sg = torch.cat([x, sg], dim=1)
        x = self.LatentSpade(x, sg)

        nx = self.Encoder(nxLMap)
        nsg = self.Encoder(npLMap)
        nsg = torch.cat([nx, nsg], dim=1)
        nx = self.LatentSpade(nx, nsg)

        x = (1-alpha)*x + alpha*nx

        x = self.Decoder(x, bg)
        return x


class RenderGenerator(nn.Module):
    def __init__(self, inDim, midDim, outDim, bgInDim, numLevels, w0, w0_ini, c, device):
        super(RenderGenerator, self).__init__()
        self.Encoder = MultiLevelEncoder(featureDim=inDim, outDim=midDim, numLevels=numLevels,
                                         w0=w0, w0_ini=w0_ini, c=c).to(device)
        self.LatentSpade = SPADEResnetBlock(indim=midDim, gdim=midDim*2, w0=w0, w0_ini=w0, c=c).to(device)
        self.Decoder = CNNGenerator(midDim=midDim, bgInDim=bgInDim, outDim=outDim, w0=w0, w0_ini=w0, c=c).to(device)

        self.weight_ini(c, self.Encoder)
        self.weight_ini(c, self.LatentSpade)
        self.weight_ini(c, self.Decoder)

    def weight_ini(self, c, net):
        for m in net.state_dict():
            k = m.split('.')
            if k[-1] == 'weight':
                siren_uniform(net.state_dict()[m], mode='fan_in', c=c)

    def forward(self, xLMap, pLMap, bg):
        x = self.Encoder(xLMap)
        sg = self.Encoder(pLMap)
        sg = torch.cat([x, sg], dim=1)
        x = self.LatentSpade(x, sg)
        x = self.Decoder(x, bg)
        return x


class Siren_DiscriminatorPatchCNN(nn.Module):
    def __init__(self, inDim, w0, w0_ini, c=6):
        super(Siren_DiscriminatorPatchCNN, self).__init__()
        # seq = [sinConv(inDim=inDim, outDim=64, kernel_size=4, stride=2, padding=1, w0=w0_ini, c=c),
        #        sinConv(inDim=64, outDim=128, kernel_size=4, stride=2, padding=1, w0=w0, c=c),
        #        sinConv(inDim=128, outDim=256, kernel_size=4, stride=2, padding=1, w0=w0, c=c),
        #        sinConv(inDim=256, outDim=512, kernel_size=4, stride=2, padding=1, w0=w0, c=c),
        #        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)]

        seq = [nn.Conv2d(inDim, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)] + \
              [nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)] + \
              [nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)] + \
              [nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)] + \
              [nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*seq)

        #self.weight_ini(c, self.model)

    def weight_ini(self, c, net):
        for m in net.state_dict():
            k = m.split('.')
            if k[-1] == 'weight':
                siren_uniform(net.state_dict()[m], mode='fan_in', c=c)

    def getLayer(self, id_list, x):
        out = []
        bInd = 0
        for id in id_list:
            if len(out) == 0:
                out.append(self.model[:id](x))
            else:
                tx = out[-1]
                out.append(self.model[bInd:id](tx))
            bInd = id

        return out

    def forward(self, x):
        return self.model(x)


class DiscriminatorPatchCNN(nn.Module):
    def __init__(self, inDim, ifSec=False):
        super(DiscriminatorPatchCNN, self).__init__()
        '''
        Receptive fild: (output_size - 1) * stride_size + kernel_size
        [70 <-- 34 <-- 16 <-- 7 <-- 4 <-- output_size (1)]
        Output image size: 1 + (in_size + 2*padding - (kernel_size-1) - 1) / stride
        [512-->256 --> 128 --> 64 --> 32 --> output_size(31)]
        '''
        if ifSec is True:
            sequence = [spectral_norm(nn.Conv2d(inDim, 64, kernel_size=4, stride=2, padding=1)),
                        nn.LeakyReLU(0.2, True)] + \
                       [spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
                        nn.LeakyReLU(0.2, True)] + \
                       [spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
                        nn.LeakyReLU(0.2, True)] + \
                       [spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
                        nn.LeakyReLU(0.2, True)] + \
                       [spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))]
        else:
            sequence = [nn.Conv2d(inDim, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)] +\
                       [nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)] +\
                       [nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)] +\
                       [nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)] + \
                       [nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)

    def getLayer(self, id_list, x):
        out = []
        bInd = 0
        for id in id_list:
            if len(out) == 0:
                out.append(self.model[:id](x))
            else:
                tx = out[-1]
                out.append(self.model[bInd:id](tx))
            bInd = id

        return out

    def weight_ini(self, c, net):
        for m in net.state_dict():
            k = m.split('.')
            if k[-1] == 'weight':
                siren_uniform(net.state_dict()[m], mode='fan_in', c=c)

    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)
