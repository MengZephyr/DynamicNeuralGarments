from Models import LaplacianPyramid, NeuralFeatMap, oneInRenderGenerator, DiscriminatorPatchCNN
from DataIO import readUV_OBJFile, ReadSampleMap, readJointFeat
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from vgg import VGG19, vgg_std, vgg_mean


class primUV_Architect(object):
    def __init__(self, device, numFeatures, FeatDim, JointDim, OutDim, viewH, viewW, uvMeshFile,
                 genCKPName=None, disCKPName=None, isTrain=True, isContinue=False, texS = 1):
        self.isTrain = isTrain
        self.device = device
        self.numFeatures = numFeatures
        self.FeatDim = FeatDim
        self.JointDim = JointDim
        self.OutDim = OutDim
        self.viewH = viewH
        self.viewW = viewW

        self.texH = self.viewH // texS
        self.texW = self.viewW // texS
        print("tex size:", self.texH, " ", self.texW)

        if genCKPName is not None:
            gen_ckp = self.load_ckp(genCKPName)
            self.layer_0 = gen_ckp['Text_layer0'].clone().to(device)
            self.layer_1 = gen_ckp['Text_layer1'].clone().to(device)
            self.layer_2 = gen_ckp['Text_layer2'].clone().to(device)
            self.layer_3 = gen_ckp['Text_layer3'].clone().to(device)
        else:
            self.layer_0 = torch.randn(1, self.FeatDim, self.texH, self.texW).to(device)
            self.layer_1 = torch.randn(1, self.FeatDim, self.texH//2, self.texW//2).to(device)
            self.layer_2 = torch.randn(1, self.FeatDim, self.texH//4, self.texW//4).to(device)
            self.layer_3 = torch.randn(1, self.FeatDim, self.texH//8, self.texW//8).to(device)

        self.layers = [self.layer_0, self.layer_1, self.layer_2, self.layer_3]

        self.sampleMap = NeuralFeatMap(self.device).to(device)
        self.textureMap = LaplacianPyramid(self.device).to(device)

        verts, faces = readUV_OBJFile(uvMeshFile, 'v', False)
        self.uvTensor = torch.from_numpy(np.array(verts)).transpose(0, 1).type(torch.FloatTensor).to(device)
        self.uvTensor = self.uvTensor[0:2, :]

        self.ImgToTensor = transforms.Compose([transforms.ToTensor()])
        self.vggImg_transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(vgg_mean + [0], vgg_std + [1.])])

        self.G_net = oneInRenderGenerator(inDim=self.FeatDim*4 + self.JointDim, midDim=512, outDim=self.OutDim,
                                          bgInDim=self.OutDim, device=device).to(device)
        print(self.G_net)
        if genCKPName is not None:
            self.G_net.load_state_dict(gen_ckp['Renderer'])

        self.DataLoss = torch.nn.L1Loss().to(self.device)
        self.criterion = torch.nn.BCELoss().to(self.device)

        if self.isTrain is True:
            self.vgg = VGG19(device).to(device)
            for param in self.vgg.parameters():
                param.requires_grad = False

            self.D_net = DiscriminatorPatchCNN(inDim=self.OutDim * 2, ifSec=True).to(device)
            print(self.D_net)
            if disCKPName is not None:
                dis_ckp = self.load_ckp(disCKPName)
                self.D_net.load_state_dict(dis_ckp['Discriminator'])

            self.optimizer_G = torch.optim.Adam(params=self.layers + list(self.G_net.parameters()),
                                                lr=1.e-4)
            self.optimizer_D = torch.optim.Adam(self.D_net.parameters(), lr=3.e-4)

    def ganLoss(self, x, isReal):
        if isReal:
            target_Label = torch.full(x.size(), 1., dtype=torch.float32, device=self.device)
        else:
            target_Label = torch.full(x.size(), 0., dtype=torch.float32, device=self.device)
        return self.criterion(x, target_Label)

    def ReadImgs(self, filelist, transf):
        ImgTensor = []
        for f in filelist:
            img = Image.open(f)
            img = transf(img).unsqueeze(0)
            ImgTensor.append(img[:, 0:3, :, :].to(self.device))
        ImgTensor = torch.cat(ImgTensor, dim=0)
        return ImgTensor

    def PrepareBatchLevelMap(self, fileList, JFList):
        BatchLevelMaps = []
        for mapFile, JF in zip(fileList, JFList):
            levelH, levelW, levelPixelValidX, levelPixelValidY, levelMap, _ = \
                ReadSampleMap(mapFile, self.numFeatures, 1, False)
            pixelMask = self.sampleMap(H=levelH[0], W=levelW[0], pX=levelPixelValidX[0], pY=levelPixelValidY[0],
                                       vertMask=levelMap[0].type(torch.FloatTensor).to(self.device),
                                       hatF=self.uvTensor)
            out = self.textureMap(layers=self.layers, pixelMask=pixelMask)

            JFeats = readJointFeat(JF)
            JFeats = torch.tensor(JFeats).to(self.device)
            JFeats = torch.transpose(JFeats, 0, 1)
            jcond = self.sampleMap(H=levelH[0], W=levelW[0], pX=levelPixelValidX[0], pY=levelPixelValidY[0],
                                   vertMask=levelMap[0].type(torch.FloatTensor).to(self.device), hatF=JFeats)
            out = torch.cat([out, jcond.unsqueeze(0)], dim=1)
            BatchLevelMaps.append(out)

        BatchLevelMaps = torch.cat(BatchLevelMaps, dim=0)
        return BatchLevelMaps

    def LoadTrainBatch(self, inmapList, prefmapList, YGtList, PrefYList, bgNameList,
                       JMotNList=None, preJMotList=None):
        BatchMaps = self.PrepareBatchLevelMap(inmapList, JMotNList)
        BatchPreMaps = self.PrepareBatchLevelMap(prefmapList, preJMotList)
        BatchGtImgs = self.ReadImgs(YGtList, self.vggImg_transform)
        BatchPreImgs = self.ReadImgs(PrefYList, self.vggImg_transform)
        BatchBgImgs = self.ReadImgs(bgNameList, self.vggImg_transform)

        return BatchMaps, BatchPreMaps, BatchBgImgs, BatchGtImgs, BatchPreImgs

    def DNet_FeatLoss(self, gt, x, Layers):
        LFeatures_gt = self.D_net.getLayer(Layers, gt)
        LFeatures_x = self.D_net.getLayer(Layers, x)
        dist = 0.
        for l in range(len(Layers)):
            dist += self.DataLoss(LFeatures_gt[l], LFeatures_x[l])
        return dist

    def peceptronLoss(self, gt, x, Layers):
        dist = 0.
        if len(Layers) == 0:
            return self.DataLoss(gt, x)

        gtFeats = self.vgg.get_content_actList(gt, Layers)
        xxFeats = self.vgg.get_content_actList(x, Layers)

        for l in range(len(Layers)):
            dist += self.DataLoss(gtFeats[l], xxFeats[l])
        dist += self.DataLoss(gt, x)
        return dist

    def setNet_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def setFeat_requires_grad(self, requires_grad=False):
        for gLayers in self.layers:
            gLayers.requires_grad = requires_grad

    def iter_Train_G(self, inmapList, prefmapList, YGtList, PrefYList, bgNameList, JMotNList=None, preJMotList=None):
        self.setFeat_requires_grad(True)
        self.setNet_requires_grad(self.G_net, True)
        self.setNet_requires_grad(self.D_net, False)
        self.G_net.train()
        self.optimizer_G.zero_grad()

        LMaps, PreLMaps, BgImgs, GtImgs, PreImgs \
            = self.LoadTrainBatch(inmapList, prefmapList, YGtList, PrefYList, bgNameList,
                                  JMotNList, preJMotList)
        out = self.G_net(LMaps, PreLMaps, BgImgs)
        #peceptron_t = self.DataLoss(out, GtImgs)
        peceptron_t = self.peceptronLoss(GtImgs, out, [4, 12, 30])
        exp_Real = self.D_net(torch.cat([out, PreImgs], dim=1))
        discrim_t = self.ganLoss(exp_Real, True)
        dfeat_t = self.DNet_FeatLoss(torch.cat([GtImgs, PreImgs], dim=1),
                                     torch.cat([out, PreImgs], dim=1),
                                     [1, 3, 5])
        LossG = 10 * peceptron_t + 10*dfeat_t + discrim_t
        LossG.backward()
        self.optimizer_G.step()

        return out, GtImgs, PreImgs, LossG, peceptron_t

    def iter_train_D(self, FakeImgs, GtImgs, PreImgs):
        self.setNet_requires_grad(self.D_net, True)
        self.setNet_requires_grad(self.G_net, False)
        self.setFeat_requires_grad(False)
        self.D_net.train()
        self.optimizer_D.zero_grad()

        D_Fake = self.D_net(torch.cat([FakeImgs, PreImgs], dim=1))
        fake_Loss = self.ganLoss(D_Fake, False)
        D_real = self.D_net(torch.cat([GtImgs, PreImgs], dim=1))
        real_Loss = self.ganLoss(D_real, True)
        LossD = fake_Loss + real_Loss
        LossD.backward()
        self.optimizer_D.step()
        return LossD

    def iter_Eval_G(self, inmapList, prefmapList, YGtList, PrefYList, bgNameList, JMotNList=None, preJMotList=None):
        self.setFeat_requires_grad(False)
        self.setNet_requires_grad(self.G_net, False)
        self.setNet_requires_grad(self.D_net, False)
        self.D_net.eval()
        self.G_net.eval()
        with torch.no_grad():
            LMaps, PreLMaps, BgImgs, GtImgs, PreImgs \
                = self.LoadTrainBatch(inmapList, prefmapList, YGtList, PrefYList, bgNameList,
                                      JMotNList, preJMotList)
            out = self.G_net(LMaps, PreLMaps, BgImgs)
            #LossG = self.DataLoss(out, GtImgs)
            LossG = self.peceptronLoss(GtImgs, out, [4, 12, 30])
            return out, GtImgs, LossG

    def run_Gen(self, inmapList, prefmapList, bgNameList, JMotNList, preJMotList):
        LMaps = self.PrepareBatchLevelMap(inmapList, JMotNList)
        preLMaps = self.PrepareBatchLevelMap(prefmapList, preJMotList)
        bgImgs = self.ReadImgs(bgNameList, self.vggImg_transform)
        self.G_net.eval()
        with torch.no_grad():
            out, m = self.G_net(LMaps, preLMaps, bgImgs, True)
            return out, m

    def save_ckp(self, savePref, itt):
        torch.save({'itter': itt, 'Text_layer0': self.layer_0, 'Text_layer1': self.layer_1, 'Text_layer2': self.layer_2,
                    'Text_layer3': self.layer_3, 'Renderer': self.G_net.state_dict(),
                    'optimizer': self.optimizer_G.state_dict()}, savePref+'_Gen.ckp')

        torch.save({'itter': itt, 'Discriminator': self.D_net.state_dict(),
                    'D_optimizer': self.optimizer_D.state_dict()},
                   savePref + '_Dis.ckp')

    def load_ckp(self, fileName):
        ckp = torch.load(fileName, map_location=lambda storage, loc: storage)
        return ckp
