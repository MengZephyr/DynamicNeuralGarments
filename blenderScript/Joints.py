import bpy
import mathutils
import numpy as np
import os


def writeBonesFramePosition(numBones, poseArray, fileName):
    with open(fileName, 'w') as f:
        f.write(str(numBones) + "\n")
        for p in poseArray:
            f.write(str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n")
        f.close

KeyBonesNames = ["mixamorig:Head", "mixamorig:Neck", "mixamorig:Spine2", "mixamorig:Spine1", "mixamorig:Spine", "mixamorig:RightShoulder", "mixamorig:RightArm", "mixamorig:RightForeArm", "mixamorig:RightHand", "mixamorig:LeftShoulder", "mixamorig:LeftArm", "mixamorig:LeftForeArm", "mixamorig:LeftHand", "mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot", "mixamorig:RightUpLeg", "mixamorig:RightLeg", "mixamorig:RightFoot"]

D = bpy.data
C = bpy.context
Arma = D.objects['Armature']
sce = C.scene

saveRoot = "D:/models/MD/DataModel/Motions/JJ/case_3/"

C.view_layer.objects.active = Arma
#bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.mode_set(mode='POSE')

frame0 = 1
frame1 = 863
for f in range(frame0, frame1+1):
    sce.frame_set(f)
    ArrayPos = []
    for bname in KeyBonesNames:
        pb = C.object.pose.bones[bname]
        mm_world = C.object.matrix_world
        mm_pb = pb.matrix
        pb_position = mm_world @ mm_pb @ pb.location
        ArrayPos.append(pb_position)

#    bpy.ops.object.mode_set(mode='OBJECT')
#    for pos in ArrayPos:
#        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=pos)

    writeBonesFramePosition(len(KeyBonesNames), ArrayPos, saveRoot+str(f).zfill(7)+ '.txt')