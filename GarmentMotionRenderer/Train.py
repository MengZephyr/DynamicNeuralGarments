from __future__ import division
from __future__ import print_function

import time
import torch
from random import shuffle
#from Architect import NetArchitecture
#from Architect_WGAN_GP import WGAN_GP_Architecture
from uv_Architecture import primUV_Architect
from trib_Architecture import Trib_RunGAN
from torchvision.utils import save_image
from vgg import vgg_std, vgg_mean

USE_CUDA = torch.cuda.is_available()
print('balin-->', USE_CUDA)
device = torch.device("cuda:1" if USE_CUDA else "cpu")

prefRoot = '../Data/Ver_5/'
mapPref = 'C_map_2287/'
yPref = 'ver_hdr/multiLaces/'
bgPref = 'ver_hdr/bg/'
JointPref = '/30_JFeat_2_10/'
uvName = prefRoot + 'case_1/30_uvMesh.obj'
ifTrib = True

numFeats = 2287
DimFeat = 16
numLevels = 5
view_H = 512
view_W = 512
BatchSize = 2
DimJoint = 19 * 5
DimOut = 3

iterations = 100000

'''-----------------------------------------------------------------------------------------'''
TrainList = []
for i in range(2, 846):
    TrainList = TrainList + [['case_1/', 'V_train/', i, 2, j] for j in range(10)]

evalList = [['case_1/', 'V_train/', i, 2, 10] for i in range(802, 846)]

print(len(TrainList))
'''-----------------------------------------------------------------------------------------'''

shuffle(TrainList)
shuffle(evalList)
Train_KK = len(TrainList) // BatchSize
Eval_KK = len(evalList) // BatchSize

NetModel = Trib_RunGAN(device=device, numFeatures=numFeats, FeatDim=DimFeat, JointDim=DimJoint,
                       OutDim=DimOut, viewH=view_H, viewW=view_W, uvMeshFile=uvName,
                       genCKPName='../ckp_A/multi_2Run/t_80000_Gen.ckp',
                       disCKPName='../ckp_A/multi_2Run/t_80000_Dis.ckp', isTrain=True, isContinue=False)


def prepareFileList(ListIDs):
    mapNList = [prefRoot + i[0] + i[1] + mapPref + str(i[2]) + '_' + str(i[4]) + '_m.txt' for i in ListIDs]
    premapNList = [prefRoot + i[0] + i[1] + mapPref + 'p_' + str(i[2]) + '_' + str(i[4]) + '_m.txt' for i in ListIDs]

    YGtNList = [prefRoot + i[0] + i[1] + yPref + str(i[2]).zfill(7) + '_' + str(i[4]) + '.png' for i in ListIDs]
    prevYGtNList = [prefRoot + i[0] + i[1] + yPref + 'p_' + str(i[2]).zfill(7) + '_' + str(i[4]) + '.png'
                    for i in ListIDs]
    NextYGtNList = [prefRoot + i[0] + i[1] + yPref + 'a_' + str(i[2]).zfill(7) + '_' + str(i[4]) + '.png'
                    for i in ListIDs]

    JMoNList = [prefRoot + i[0] + JointPref + str(i[2]).zfill(7) + '.txt' for i in ListIDs]
    preJMoNList = [prefRoot + i[0] + JointPref + str(max(i[2]-1, i[3])).zfill(7) + '.txt' for i in ListIDs]

    bgNList = [prefRoot + i[0] + i[1] + bgPref + str(i[2]).zfill(7) + '_' + str(i[4]) + '.png' for i in ListIDs]

    return mapNList, premapNList, YGtNList, prevYGtNList, NextYGtNList, bgNList, JMoNList, preJMoNList


def draw_iterResult(X, Gt, iter, nB, prefN):
    simg = []
    for i in range(nB):
        x = X[i, :, :, :] * torch.tensor(vgg_std).view(-1, 1, 1).to(device) + \
            torch.tensor(vgg_mean).view(-1, 1, 1).to(device)
        g = Gt[i, :, :, :] * torch.tensor(vgg_std).view(-1, 1, 1).to(device) + \
            torch.tensor(vgg_mean).view(-1, 1, 1).to(device)
        c = torch.cat([g, x], dim=1)
        simg.append(c)
    simg = torch.cat(simg, dim=2)
    save_image(simg, fp=prefN + str(iter) + '.png')


IFSumWriter = False
if IFSumWriter:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

betit = -1
for itt in range(betit+1, iterations+1):
    t = time.time()
    k = itt % Train_KK
    if k == 0:
        shuffle(TrainList)
    idList = TrainList[k*BatchSize: (k+1)*BatchSize]
    mapNList, premapNList, YGtNList, prevYGtNList, nextYGtNList, \
    bgNList, JMoNList, preJMoNList = prepareFileList(idList)
    if ifTrib:
        out, GtImgs, PreImgs, NextGtImgs, LossG, peceptron_t \
            = NetModel.Trib_iterTrain_GNet(inmapList=mapNList, prefmapList=premapNList,
                                           YGtList=YGtNList, PrefYGtList=prevYGtNList, NextYGtList=nextYGtNList,
                                           bgNameList=bgNList, JMotNList=JMoNList, preJMotList=preJMoNList)
        LossD = NetModel.Trib_iterTrain_DNet(fake=out.detach(), gtImgs=GtImgs, preGtImgs=PreImgs, NextGtImgs=NextGtImgs)
        print("Tri_Train Iter: {}, Dist: {:.4f}, Generator Loss: {:.4f}, Discriminator Loss: {:.4f}, time:{:.4f}s".
              format(itt, peceptron_t.item(), LossG.item(), LossD.item(), time.time() - t))
    else:
        out, GtImgs, PreImgs, LossG, peceptron_t \
            = NetModel.iter_Train_G(inmapList=mapNList, prefmapList=premapNList,
                                    YGtList=YGtNList, PrefYList=prevYGtNList, bgNameList=bgNList,
                                    JMotNList=JMoNList, preJMotList=preJMoNList)

        LossD = NetModel.iter_train_D(FakeImgs=out.detach(), GtImgs=GtImgs, PreImgs=PreImgs)
        print("Train Iter: {}, Dist: {:.4f}, Generator Loss: {:.4f}, Discriminator Loss: {:.4f}, time:{:.4f}s".
              format(itt, peceptron_t.item(), LossG.item(), LossD.item(), time.time() - t))

    if itt % 100 == 0:
        if IFSumWriter:
            writer.add_scalar('G_Loss', LossG, itt)
            writer.add_scalar('P_Loss', peceptron_t, itt)
            writer.add_scalar('D_Loss', LossD, itt)
    if itt % 500 == 0:
        draw_iterResult(out, GtImgs, itt, BatchSize, '../test_B/t_')

    if itt % 1000 == 0:
        ckpName = '../ckp_B/c_1000'
        if itt % 10000 == 0:
            ckpName = '../ckp_B/t_' + str(itt)
        t = time.time()
        k = (itt // 1000) % Eval_KK
        if k == 0:
            shuffle(evalList)
        idList = evalList[k * BatchSize: (k + 1) * BatchSize]
        mapNList, premapNList, YGtNList, prevYGtNList, nextYGtNList, \
        bgNList, JMoNList, preJMoNList = prepareFileList(idList)
        out, GtImgs, recLoss \
            = NetModel.iter_Eval_G(inmapList=mapNList, prefmapList=premapNList,
                                   YGtList=YGtNList, PrefYList=prevYGtNList, bgNameList=bgNList,
                                   JMotNList=JMoNList, preJMotList=preJMoNList)
        draw_iterResult(out, GtImgs, itt, BatchSize, '../test_B/r_')
        if IFSumWriter:
            writer.add_scalar('unSeen', recLoss, itt)
        NetModel.save_ckp(ckpName, itt)





















