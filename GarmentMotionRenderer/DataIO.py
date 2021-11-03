import torch
import os
import numpy as np


def readUV_OBJFile(fname, vFlag='v', ifLeft = False):
    if not(os.path.exists(fname)):
        return None, None
    vertArray = []
    faceArray = []

    if vFlag == 'v':
        faceS = 0
    elif vFlag == 'vn':
        faceS = 1
    elif vFlag == 'vt':
        faceS = 2
    else:
        return [], []

    file = open(fname, "r")
    for line in file:
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == vFlag:
            v = [float(x) for x in values[1:4]]
            if ifLeft:
                vertArray.append([v[0], -v[2], v[1]])
            else:
                vertArray.append([v[0], 1.-v[1], v[2]])

        if values[0] == 'f':
            f = [int(x.split('/')[faceS])-1 for x in values[1:4]]
            faceArray.append(f)
    return vertArray, faceArray


def readFileLine(file):
    line = file.readline()
    values = line.split()
    return values


def readJointFeat(fileName):
    file = open(fileName, "r")
    numV = int(readFileLine(file)[0])
    dimF = int(readFileLine(file)[0])
    FeatArray = []
    for vi in range(numV):
        Feat = [float(x) for x in readFileLine(file)]
        FeatArray.append(Feat)
    file.close()
    return FeatArray


def ReadSampleMap(fileName, numV, outNumLevel=5, ifColor=False):
    file = open(fileName, "r")
    if ifColor is True:
        colorV = [float(x) for x in readFileLine(file)]
    else:
        colorV = [-1., -1., -1.]
    numLevel = int(readFileLine(file)[0])
    levelH = [int(x) for x in readFileLine(file)]
    levelW = [int(x) for x in readFileLine(file)]
    levelPixelValidX = []
    levelPixelValidY = []
    levelMap = []
    lc = 0
    for le in range(numLevel):
        if lc >= outNumLevel:
            break

        levelInfo = readFileLine(file)
        levelID = int(levelInfo[0])
        numValid = int(levelInfo[1])
        PixelValidX = []
        PixelValidY = []
        Map = torch.zeros(numV, numValid)
        for vi in range(numValid):
            values = readFileLine(file)
            PixelValidX.append(int(values[0]))
            PixelValidY.append(int(values[1]))
            ind = [int(values[2]), int(values[3]), int(values[4])]
            u = float(values[5])
            v = float(values[6])
            k = torch.tensor([1.-u-v, u, v])
            Map[ind, vi] = k
        levelMap.append(Map)
        levelPixelValidX.append(PixelValidX)
        levelPixelValidY.append(PixelValidY)
        lc += 1

    file.close()

    return levelH, levelW, levelPixelValidX, levelPixelValidY, levelMap, colorV
