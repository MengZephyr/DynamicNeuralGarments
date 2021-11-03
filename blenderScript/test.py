import bpy
import bmesh
import os
import random
from math import *
from mathutils import *
import mathutils

C = bpy.context
D = bpy.data

sk = D.scenes.keys()[0]
myCamera = D.scenes[sk].camera
myViewer = C.view_layer
pmm = myCamera.calc_matrix_camera(C.evaluated_depsgraph_get())
sce = C.scene
Arma = D.objects['Armature']


def randFloat(a, b):
    return random.random()*(b-a) + a


def readOBJFile(fname):
    if not(os.path.exists(fname)):
        return None, None
    vertArray = []
    faceArray = []
    file = open(fname, "r")
    for line in file:
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'v':
            v = [float(x) for x in values[1:4]]
            vertArray.append([v[0], v[1], v[2]])
        if values[0] == 'f':
            f = [int(x.split('/')[0])-1 for x in values[1:4]]
            faceArray.append(f)
    file.close()
    return vertArray, faceArray


def writeMatrix(wmm, pmm, cpos, fileName):
    with open(fileName, 'w') as f:
        for c in range(len(wmm)):
            f.write(str(wmm[c][0]) + " " + str(wmm[c][1]) + " " + str(wmm[c][2]) + " " + str(wmm[c][3]) + "\n")
        f.write("\n")
        for c in range(len(pmm)):
            f.write(str(pmm[c][0]) + " " + str(pmm[c][1]) + " " + str(pmm[c][2]) + " " + str(pmm[c][3]) + "\n")
        f.write("\n")
        f.write(str(cpos[0]) + " " + str(cpos[1]) + " " + str(cpos[2]) + " " + "1." + "\n")
        f.write("\n")
        f.close()

        
def writeList(dlist, fileName):
    with open(fileName, 'w') as f:
        f.write(str(len(dlist)) + "\n")
        for d in dlist:
            f.write(str(d) + " ")
        f.write("\n")
        f.close()

        
def readList(fileName):
    if not(os.path.exists(fileName)):
        return None
    
    with open(fileName, 'r') as f:
        l = f.readline()
        d = l.split()
        dLen = int(d[0])
        
        l = f.readline()
        d = l.split()
        dlist = [float(x) for x in d[0:dLen]]
        f.close()
    return dlist

#-----------------------------------------------------------------

#pre_FileName = 'D:/models/MD/DataModel/DressOri/case_1/ShortDress/chessboard/PD10_0000001.obj'
#bpy.ops.import_scene.obj(filepath=pre_FileName, split_mode='OFF')

#premyobj = C.selected_objects[0]
#premyobj.name = 'pre_Dress_obj'
#premymeshes = premyobj.data
#premymeshes.name = 'pre_Dress_mesh'
#mat = D.materials.get('Dress_mat')
#premyobj.active_material = mat

#-----------------------------------------------------------------

#FileName = 'D:/models/MD/DataModel/DressOri/case_1/ShortDress/chessboard/PD10_0000002.obj'
#bpy.ops.import_scene.obj(filepath=FileName, split_mode='OFF')

#myobj = C.selected_objects[0]
#myobj.name = 'Dress_obj'
#mymeshes = myobj.data
#mymeshes.name = 'Dress_mesh'
#mat = D.materials.get('Dress_mat')
#myobj.active_material = mat

#-----------------------------------------------------------------

#FileName = 'D:/models/MD/DataModel/DressOri/case_1/ShortDress/chessboard/PD10_0000002.obj'
#bpy.ops.import_scene.obj(filepath=FileName, split_mode='OFF')

#next_myobj = C.selected_objects[0]
#next_myobj.name = 'next_Dress_obj'
#nextmymeshes = next_myobj.data
#nextmymeshes.name = 'next_Dress_mesh'
#mat = D.materials.get('Dress_mat')
#next_myobj.active_material = mat

#-----------------------------------------------------------------

pre_obj = D.objects['pre_Dress_obj']
pre_meshes = pre_obj.data

cur_obj = D.objects['Dress_obj']
cur_meshes = cur_obj.data

next_obj = D.objects['next_Dress_obj']
next_meshes = next_obj.data

minFrame = 1
maxFrame = 849
iniRz = -25

Frame0 = 11
Frame1 = 848
FrameID = [i for i in range(Frame0, Frame1+1)]
ifGarment = True
ifSaveCamera = False

numView = 1
delta = 1./float(numView)

ObjRoot = 'D:/models/MD/DataModel/DressOri/case_1/Chamuse_Complex/10_L/'
saveRoot = 'D:/models/NR/Data/ver_6/case_1/multiLaces/v_tt_Comp/'


CameraBox = D.objects['Empty']
    

for fID in FrameID:
    if fID >= maxFrame:
        break
    
    if ifGarment is True:
        ObjFile = ObjRoot + 'PD10_' + str(fID+1).zfill(7) + '.obj'
        vertArra, faceArray = readOBJFile(ObjFile)
        C.view_layer.objects.active = cur_obj
        bpy.ops.object.mode_set(mode='EDIT')
        cur_bm = bmesh.new()
        cur_bm.from_mesh(cur_meshes)
        i = 0
        for v in cur_bm.verts:
            v.co = Vector(vertArra[i])
            i += 1
        bpy.ops.object.mode_set(mode='OBJECT')
        cur_bm.to_mesh(cur_meshes)
        cur_meshes.update()
        cur_bm.free()
        
#    offSetList = readList(saveRoot + 'cList/' + str(fID+1).zfill(7) + '.txt')
#    numView = len(offSetList)
    
    for vID in range(numView):
        C.view_layer.objects.active = CameraBox
            
        C.object.constraints["Follow Path"].offset_factor = 0.
        bpy.data.worlds["World"].node_tree.nodes.get("Mapping").rotation[2] = radians(C.object.constraints["Follow Path"].offset_factor*360 + iniRz)
        
        sce.frame_set(fID)
        next_obj.hide_render=True
        cur_obj.hide_render=True
        pre_obj.hide_render = True
        D.scenes[sk].render.filepath = saveRoot + 'bg/' + str(fID+1).zfill(7) + '_' + str(vID) + '.png'
        bpy.ops.render.render(write_still=True)
        
        next_obj.hide_render=True    
        cur_obj.hide_render=False
        pre_obj.hide_render = True
        D.scenes[sk].render.filepath = saveRoot + 'render_1/' + str(fID+1).zfill(7) + '_' + str(vID) + '.png'
        bpy.ops.render.render(write_still=True)
        
        if ifSaveCamera is True:
            wmm = myCamera.matrix_world.inverted()
            npos = myCamera.matrix_world.translation
            writeMatrix(wmm, pmm, npos, saveRoot + 'cameras/' + str(fID+1).zfill(7) + '_' + str(vID) + '_c.txt')
