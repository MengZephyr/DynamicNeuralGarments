import torch
from uv_Architecture import primUV_Architect


class Trib_RunGAN(primUV_Architect):
    def __init__(self, device, numFeatures, FeatDim, JointDim, OutDim, viewH, viewW, uvMeshFile,
                 genCKPName=None, disCKPName=None, isTrain=True, isContinue=False):
        super().__init__(device, numFeatures, FeatDim, JointDim, OutDim,
                         viewH, viewW, uvMeshFile, genCKPName, disCKPName, isTrain, isContinue)

    def Trib_loadBatches(self, inmapList, prefmapList, YGtList, PrefYGtList, NextYGtList,
                         bgNameList, JMotNList, preJMotList):
        BatchMaps = super().PrepareBatchLevelMap(inmapList, JMotNList)
        prefBatchMaps = super().PrepareBatchLevelMap(prefmapList, preJMotList)

        BatchBgImgs = super().ReadImgs(bgNameList, self.vggImg_transform)

        BatchGtImgs = super().ReadImgs(YGtList, self.vggImg_transform)
        BatchPreGtImgs = super().ReadImgs(PrefYGtList, self.vggImg_transform)
        BatchNextGtImgs = super().ReadImgs(NextYGtList, self.vggImg_transform)

        return BatchMaps, prefBatchMaps, BatchBgImgs, BatchGtImgs, BatchPreGtImgs, BatchNextGtImgs

    def Trib_iterTrain_GNet(self, inmapList, prefmapList, YGtList, PrefYGtList, NextYGtList,
                            bgNameList, JMotNList, preJMotList):
        # opt generator
        super().setNet_requires_grad(self.D_net, False)
        super().setNet_requires_grad(self.G_net, True)
        super().setFeat_requires_grad(True)
        self.G_net.train()
        self.optimizer_G.zero_grad()
        LMaps, preLMaps, BgImgs, GtImgs, PreGtImgs, NextGtImgs \
            = self.Trib_loadBatches(inmapList, prefmapList, YGtList, PrefYGtList, NextYGtList,
                                    bgNameList, JMotNList, preJMotList)
        out = self.G_net(LMaps, preLMaps, BgImgs)

        exp_Real = self.D_net(torch.cat([out, PreGtImgs], dim=1))
        next_exp_Real = self.D_net(torch.cat([NextGtImgs, out], dim=1))
        discrim_t = super().ganLoss(exp_Real, True) + super().ganLoss(next_exp_Real, True)
        peceptron_t = super().peceptronLoss(GtImgs, out, [4, 12, 30])
        dfeat_t = super().DNet_FeatLoss(torch.cat([out, PreGtImgs], dim=1),
                                        torch.cat([GtImgs, PreGtImgs], dim=1), [1, 3, 5]) + \
                  super().DNet_FeatLoss(torch.cat([NextGtImgs, out], dim=1),
                                        torch.cat([NextGtImgs, GtImgs], dim=1), [1, 3, 5])

        LossG = 10 * peceptron_t + 5 * dfeat_t + 0.5 * discrim_t
        LossG.backward()
        self.optimizer_G.step()
        return out, GtImgs, PreGtImgs, NextGtImgs, LossG, peceptron_t

    def Trib_iterTrain_DNet(self, fake, gtImgs, preGtImgs, NextGtImgs):
        # opt discriminator
        super().setNet_requires_grad(self.D_net, True)
        super().setNet_requires_grad(self.G_net, False)
        super().setFeat_requires_grad(False)
        self.D_net.train()
        self.optimizer_D.zero_grad()

        D_Fake = self.D_net(torch.cat([fake, preGtImgs], dim=1))
        next_D_Fake = self.D_net(torch.cat([NextGtImgs, fake], dim=1))
        fake_Loss = super().ganLoss(D_Fake, False) + super().ganLoss(next_D_Fake, False)

        D_real = self.D_net(torch.cat([gtImgs, preGtImgs], dim=1))
        next_D_real = self.D_net(torch.cat([NextGtImgs, gtImgs], dim=1))
        real_Loss = super().ganLoss(D_real, True) + super().ganLoss(next_D_real, True)

        LossD = 0.5 * fake_Loss + 0.5 * real_Loss
        LossD.backward()
        self.optimizer_D.step()
        return LossD

    def Trib_iterEval_GNet(self, inmapList, prefmapList, YGtList, PrefYGtList, NextYGtList,
                           bgNameList, JMotNList, preJMotList):
        super().setNet_requires_grad(self.G_net, False)
        super().setFeat_requires_grad(False)
        self.G_net.eval()
        with torch.no_grad():
            LMaps, preLMaps, BgImgs, GtImgs, PreGtImgs, NextGtImgs \
                = self.Trib_loadBatches(inmapList, prefmapList, YGtList, PrefYGtList, NextYGtList,
                                        bgNameList, JMotNList, preJMotList)
            out = self.G_net(LMaps, preLMaps, BgImgs)
            peceptron_t = super().peceptronLoss(GtImgs, out, [4, 12, 30])

            return out, GtImgs, peceptron_t