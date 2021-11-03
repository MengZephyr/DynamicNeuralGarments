from __future__ import division
from __future__ import print_function

import time
import torch
from random import shuffle
from uv_Architecture import primUV_Architect
from torchvision.utils import save_image
from vgg import vgg_std, vgg_mean

#USE_CUDA = False
USE_CUDA = torch.cuda.is_available()
print('balin-->', USE_CUDA)
device = torch.device("cuda:1" if USE_CUDA else "cpu")

prefRoot = '../test/'
viewRoot = '/'
mapPref = 'C_map_2287/'
bgPref = 'bg/'
JointPref = '30_JFeat_2_10/'
uvName = prefRoot + '30_uvMesh.obj'

saveRoot = '../rst/'

numFeats = 2287
DimFeat = 16
numLevels = 5
view_H = 512
view_W = 512
BatchSize = 4
DimJoint = 19*5
DimOut = 3

Frame0 = 1

FrameList = [[j, 0] for j in range(335, 541)]

KK = len(FrameList) // BatchSize
KK = KK if len(FrameList) % BatchSize == 0 else KK+1

Models = primUV_Architect(device=device, numFeatures=numFeats, FeatDim=DimFeat, JointDim=DimJoint,
                          OutDim=DimOut, viewH=view_H, viewW=view_W, uvMeshFile=uvName,
                          genCKPName='../ckp/MultiLayers/t_80000_Gen.ckp',
                          disCKPName=None, isTrain=False, isContinue=False)


def prepareFileList(ListIDs):
    mapNList = [prefRoot + viewRoot + mapPref + str(i[0]) + '_' + str(i[1]) + '_m.txt' for i in ListIDs]
    prevmapNList = [prefRoot + viewRoot + mapPref + 'p_' + str(i[0]) + '_' + str(i[1]) + '_m.txt' for i in ListIDs]
    #prevmapNList = [prefRoot + viewRoot + mapPref + str(max(i-1, Frame0)) + '_m.txt' for i in ListIDs]
    JMoNList = [prefRoot + JointPref + str(i[0]).zfill(7) + '.txt' for i in ListIDs]
    preJMoNList = [prefRoot + JointPref + str(max(i[0]-1, Frame0)).zfill(7) + '.txt' for i in ListIDs]
    bgNList = [prefRoot + viewRoot + bgPref + str(i[0]).zfill(7) + '_' + str(i[1]) + '.png' for i in ListIDs]
    return mapNList, prevmapNList, bgNList, JMoNList, preJMoNList


def draw_iterResult(X, prefN, idList):
    numI = X.size()[0]
    for i in range(numI):
        x = X[i, :, :, :] * torch.tensor(vgg_std).view(-1, 1, 1).to(device) + \
            torch.tensor(vgg_mean).view(-1, 1, 1).to(device)
        save_image(x, fp=prefN + str(idList[i][0]) + '_' + str(idList[i][1]) + '.png')


def draw_iterMask(X, prefN, idList):
    numI = X.size()[0]
    for i in range(numI):
        x = X[i, :, :, :]
        save_image(x, fp=prefN + str(idList[i][0]) + '_' + str(idList[i][1]) + '_m.png')


for i in range(KK):
    begK = BatchSize * i
    endK = BatchSize * (i + 1) if BatchSize * (i + 1) < len(FrameList) else len(FrameList)
    idList = FrameList[begK:endK]
    print(idList)
    mapNList, prevmapNList, bgNList, JMoNList, preJMoNList = prepareFileList(idList)
    out, mask = Models.run_Gen(inmapList=mapNList, prefmapList=prevmapNList, bgNameList=bgNList,
                               JMotNList=JMoNList, preJMotList=preJMoNList)
    draw_iterResult(out, saveRoot, idList)
    #draw_iterMask(mask, saveRoot, idList)
